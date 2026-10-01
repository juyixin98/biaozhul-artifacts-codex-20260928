"""Convenience driver that runs two real local worker subprocesses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx

from .worker import main as worker_main


def _load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_local_example(base_url: str, payload_path: Path) -> int:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=10.0) as client:
        created = client.post("/runs", json=_load_payload(payload_path))
        created.raise_for_status()
        run_id = created.json()["run"]["run_id"]
        print(f"created {run_id}")

        env = os.environ.copy()
        source_root = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = str(source_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        processes = [
            subprocess.Popen(
                [sys.executable, "-m", "admm_consensus.worker", base_url, run_id, str(worker_id), "--poll-seconds", "0.01"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for worker_id in (1, 2)
        ]
        try:
            while True:
                run = client.get(f"/runs/{run_id}").json()["run"]
                if run["status"] != "running":
                    break
                pending = run["pending_worker_ids"]
                if len(pending) == 2:
                    client.post(f"/runs/{run_id}/complete", timeout=10.0).raise_for_status()
                time.sleep(0.01)
        finally:
            output = []
            for process in processes:
                process.terminate()
                output.append(process.communicate(timeout=5))
                if process.poll() is None:
                    process.kill()

        run = client.get(f"/runs/{run_id}").json()["run"]
        latest = run["trajectory"][-1]
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": run["status"],
                    "rounds": len(run["trajectory"]),
                    "final_rho": run["rho"],
                    "primal_residual": latest["primal_residual"],
                    "dual_residual": latest["dual_residual"],
                    "z": run["z"],
                    "reference": run["reference"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        for worker_id, (stdout, stderr) in enumerate(output, start=1):
            print(f"--- worker {worker_id} stdout ---")
            print(stdout)
            if stderr:
                print(f"--- worker {worker_id} stderr ---")
                print(stderr)
        return 0 if run["status"] == "converged" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the supplied example using two local worker processes.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--payload", type=Path, default=Path("examples/asymmetric_problem.json"))
    args = parser.parse_args(argv)
    try:
        return run_local_example(args.base_url, args.payload)
    except httpx.HTTPError as exc:
        print(f"local runner HTTP failure: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
