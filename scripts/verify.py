#!/usr/bin/env python3
"""Independent end-to-end verification for the local max-flow server."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "fixtures"
RESULT_DIR = ROOT / "test-results"


class VerificationFailure(AssertionError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(base_url: str) -> None:
    deadline = time.time() + 20
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise VerificationFailure(f"server did not become healthy: {last_error}")


@contextlib.contextmanager
def server_context(use_existing: bool) -> Any:
    if use_existing:
        host = os.environ.get("MAXFLOW_HOST", "127.0.0.1")
        port = os.environ.get("MAXFLOW_PORT", "8080")
        yield f"http://{host}:{port}"
        return

    port = free_port()
    env = {**os.environ, "MAXFLOW_HOST": "127.0.0.1", "MAXFLOW_PORT": str(port)}
    process = subprocess.Popen(
        [str(ROOT / "target" / "release" / "maxflow-server")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for_health(base_url)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def fixtures() -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(FIXTURE_DIR.glob("*.json"))]


def cut_capacity(fixture: dict[str, Any], source_side: set[int]) -> int:
    return sum(
        edge["capacity"]
        for edge in fixture["edges"]
        if edge["from"] in source_side and edge["to"] not in source_side
    )


def enumerate_min_cut(fixture: dict[str, Any]) -> tuple[int, list[int], list[str]]:
    source = fixture["source"]
    sink = fixture["sink"]
    optional = [
        node
        for node in range(fixture["node_count"])
        if node not in (source, sink)
    ]
    best: tuple[int, set[int]] | None = None
    for mask in range(1 << len(optional)):
        side = {source}
        for index, node in enumerate(optional):
            if mask & (1 << index):
                side.add(node)
        value = cut_capacity(fixture, side)
        if best is None or value < best[0]:
            best = (value, side)
    assert best is not None
    edges = sorted(
        edge["id"]
        for edge in fixture["edges"]
        if edge["from"] in best[1] and edge["to"] not in best[1]
    )
    return best[0], sorted(best[1]), edges


def check_flow(fixture: dict[str, Any], certificate: dict[str, Any]) -> tuple[list[str], int]:
    reasons: list[str] = []
    flow_by_id: dict[str, int] = {}
    edge_by_id = {edge["id"]: edge for edge in fixture["edges"]}

    if len(certificate["edge_flows"]) != len(fixture["edges"]):
        reasons.append("edge flow count differs from original edge count")

    for item in certificate["edge_flows"]:
        edge_id = item["edge_id"]
        flow = item["flow"]
        if edge_id in flow_by_id:
            reasons.append(f"duplicate flow entry: {edge_id}")
        if edge_id not in edge_by_id:
            reasons.append(f"unknown edge: {edge_id}")
        elif flow < 0 or flow > edge_by_id[edge_id]["capacity"]:
            reasons.append(f"flow outside [0, capacity]: {edge_id}")
        flow_by_id[edge_id] = flow

    balance = [0] * fixture["node_count"]
    for edge in fixture["edges"]:
        flow = flow_by_id.get(edge["id"])
        if flow is None:
            reasons.append(f"missing flow: {edge['id']}")
            continue
        balance[edge["from"]] += flow
        balance[edge["to"]] -= flow

    source = fixture["source"]
    sink = fixture["sink"]
    for node, net in enumerate(balance):
        if node not in (source, sink) and net != 0:
            reasons.append(f"node {node} violates conservation: net={net}")
    value = balance[source]
    if value < 0:
        reasons.append("source has net inflow")
    if balance[sink] != -value:
        reasons.append("sink inflow does not match source outflow")
    if certificate["value"] != value:
        reasons.append(
            f"declared value {certificate['value']} != source net outflow {value}"
        )
    return reasons, value


def wait_job(client: httpx.Client, job_id: str) -> dict[str, Any]:
    for _ in range(200):
        response = client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        body = response.json()
        if body["state"] != "running":
            return body
        time.sleep(0.02)
    raise VerificationFailure(f"job did not finish: {job_id}")


def submit(client: httpx.Client, fixture: dict[str, Any], max_phases: int | None = None):
    payload = {
        "node_count": fixture["node_count"],
        "source": fixture["source"],
        "sink": fixture["sink"],
        "edges": fixture["edges"],
    }
    if max_phases is not None:
        payload["max_phases"] = max_phases
    response = client.post("/jobs", json=payload)
    response.raise_for_status()
    return wait_job(client, response.json()["id"])


def assert_completed(fixture: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    certificate = body["certificate"]
    reasons, value = check_flow(fixture, certificate)
    oracle_value, oracle_side, oracle_edges = enumerate_min_cut(fixture)
    if reasons:
        raise VerificationFailure("; ".join(reasons))
    if value != fixture["expected_value"] or oracle_value != value:
        raise VerificationFailure(
            f"{fixture['name']}: flow={value}, expected={fixture['expected_value']}, cut={oracle_value}"
        )
    if not body["optimal"] or body["state"] != "completed":
        raise VerificationFailure(f"{fixture['name']}: completed result is not marked optimal")
    cut = certificate.get("cut")
    if cut is None:
        raise VerificationFailure(f"{fixture['name']}: missing cut")
    source_side = set(cut["source_side"])
    actual_edges = sorted(
        edge["id"]
        for edge in fixture["edges"]
        if edge["from"] in source_side and edge["to"] not in source_side
    )
    if cut["source_side"] != fixture["expected_source_side"]:
        raise VerificationFailure(f"{fixture['name']}: source side mismatch")
    if cut["sink_side"] != fixture["expected_sink_side"]:
        raise VerificationFailure(f"{fixture['name']}: sink side mismatch")
    if actual_edges != sorted(fixture["expected_cut_edges"]) or actual_edges != oracle_edges:
        raise VerificationFailure(f"{fixture['name']}: cut edge mismatch")
    return {
        "flow_value": value,
        "enumerated_cut_capacity": oracle_value,
        "source_side": oracle_side,
        "cut_edges": oracle_edges,
        "judgment": "PASS: flow conservation, cut partition, saturating cut, and equality checked independently",
    }


def check_error(
    client: httpx.Client,
    name: str,
    payload: Any,
    status: int,
    category: str,
    code: str,
) -> dict[str, Any]:
    response = client.post("/jobs", json=payload)
    body = response.json()
    if response.status_code != status or body["category"] != category or body["code"] != code:
        raise VerificationFailure(
            f"{name}: expected {status} {category}/{code}, got {response.status_code} {body}"
        )
    return {"name": name, "status": status, "category": category, "code": code}


def verify(base_url: str, run_id: str) -> dict[str, Any]:
    report = {
        "run_id": run_id,
        "started_at": now(),
        "base_url": base_url,
        "fixture_results": [],
        "error_cases": [],
        "judgments": [],
    }
    with httpx.Client(base_url=base_url, timeout=5) as client:
        client.get("/health").raise_for_status()
        for fixture in fixtures():
            body = submit(client, fixture)
            result = assert_completed(fixture, body)
            report["fixture_results"].append(
                {
                    "name": fixture["name"],
                    "server_run_id": body["run_id"],
                    "server_events": body["events"],
                    "independent_result": result,
                }
            )
            report["judgments"].append(
                f"PASS {fixture['name']}: max_flow=min_cut={result['flow_value']}"
            )

        reverse = next(item for item in fixtures() if item["name"] == "reverse residual reroute")
        canceled = submit(client, reverse, max_phases=1)
        reasons, value = check_flow(reverse, canceled["certificate"])
        if reasons or canceled["optimal"] or canceled["certificate"].get("cut") is not None:
            raise VerificationFailure(f"canceled result is invalid: {reasons}")
        report["fixture_results"].append(
            {
                "name": "phase-budget cancellation",
                "server_run_id": canceled["run_id"],
                "server_events": canceled["events"],
                "independent_result": {
                    "flow_value": value,
                    "judgment": "PASS: feasible current flow returned; no cut and optimal=false",
                },
            }
        )

        report["error_cases"].append(
            check_error(
                client,
                "negative capacity",
                {
                    "node_count": 2,
                    "source": 0,
                    "sink": 1,
                    "edges": [{"id": "bad", "from": 0, "to": 1, "capacity": -1}],
                },
                422,
                "input_error",
                "capacity_out_of_range",
            )
        )
        report["error_cases"].append(
            check_error(
                client,
                "source equals sink",
                {"node_count": 1, "source": 0, "sink": 0, "edges": []},
                422,
                "input_error",
                "source_equals_sink",
            )
        )
        report["error_cases"].append(
            check_error(
                client,
                "graph too large",
                {"node_count": 1025, "source": 0, "sink": 1024, "edges": []},
                503,
                "resource_exhausted",
                "graph_too_large",
            )
        )
        missing = client.get("/jobs/missing")
        if missing.status_code != 404 or missing.json()["category"] != "not_found":
            raise VerificationFailure(f"unknown job classification failed: {missing.text}")
        report["error_cases"].append(
            {"name": "unknown job", "status": 404, "category": "not_found", "code": "not_found"}
        )
    report["finished_at"] = now()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", action="store_true", help="use MAXFLOW_HOST:MAXFLOW_PORT")
    args = parser.parse_args()
    run_id = f"verify-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    try:
        with server_context(args.existing) as base_url:
            report = verify(base_url, run_id)
    except Exception as exc:
        report = {
            "run_id": run_id,
            "finished_at": now(),
            "judgment": f"FAIL: {type(exc).__name__}: {exc}",
        }
        RESULT_DIR.mkdir(exist_ok=True)
        (RESULT_DIR / f"{run_id}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    RESULT_DIR.mkdir(exist_ok=True)
    output_path = RESULT_DIR / f"{run_id}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"verification report: {output_path}")
    for judgment in report["judgments"]:
        print(judgment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
