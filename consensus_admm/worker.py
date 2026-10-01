"""Local compute-process worker CLI."""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx
import numpy as np

from .kernel import solve_local
from .parser import parse_worker


def run_worker(
    base_url: str,
    run_id: str,
    worker_id: str,
    *,
    poll_seconds: float = 0.05,
    max_wait_seconds: float = 60.0,
) -> dict:
    deadline = time.time() + max_wait_seconds
    last_iteration = -1

    def get_task_with_retry() -> httpx.Response:
        for attempt in range(20):
            response = client.get(f"/runs/{run_id}/workers/{worker_id}/task")
            if response.status_code != 409:
                return response
            time.sleep(max(poll_seconds, 0.02))
        return response

    with httpx.Client(base_url=base_url, timeout=5.0) as client:
        while time.time() < deadline:
            task_response = get_task_with_retry()
            task_response.raise_for_status()
            task = task_response.json()
            status = task["status"]
            if status in {"converged", "max_iterations_reached"}:
                return task
            if task.get("already_submitted"):
                time.sleep(poll_seconds)
                continue
            if status == "timed_out" or task.get("expired"):
                if task.get("expired"):
                    timeout_response = client.post(
                        f"/runs/{run_id}/timeout",
                        json={"reason": "worker_observed_deadline"},
                    )
                    if timeout_response.status_code not in {200, 409}:
                        timeout_response.raise_for_status()
                resume_response = client.post(
                    f"/runs/{run_id}/resume",
                    json={
                        "deadline_ms": None,
                        "reason": "worker_retry_after_timeout",
                    },
                )
                if resume_response.status_code not in {200, 409}:
                    resume_response.raise_for_status()
                time.sleep(poll_seconds)
                continue

            run = client.get(f"/runs/{run_id}").json()
            worker_payload = next(
                worker
                for worker in run["problem"]["workers"]
                if worker["id"] == worker_id
            )
            worker = parse_worker(worker_payload, len(task["z"]))
            result = solve_local(
                worker=worker,
                iteration=int(task["iteration"]),
                rho=float(task["rho"]),
                z_pre=np.asarray(task["z"], dtype=float),
                u_pre=np.asarray(task["u"], dtype=float),
            )
            if result.iteration != last_iteration:
                print(
                    json.dumps(
                        {
                            "event": "local_update",
                            "run_id": run_id,
                            "worker_id": worker_id,
                            "iteration": result.iteration,
                            "rho": result.rho,
                            "objective": result.objective,
                            "stationarity_residual": result.stationarity_residual,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                last_iteration = result.iteration
            response = client.post(
                f"/runs/{run_id}/workers/{worker_id}/submissions",
                json={
                    "round_token": task["round_token"],
                    "rho": result.rho,
                    "z_pre": result.z_pre.tolist(),
                    "u_pre": result.u_pre.tolist(),
                    "x": result.x.tolist(),
                    "objective": result.objective,
                    "stationarity_residual": result.stationarity_residual,
                    "solve_seconds": result.solve_seconds,
                },
            )
            if response.status_code == 409:
                # Another process can complete a round while this request races.
                terminal = get_task_with_retry()
                terminal.raise_for_status()
                terminal_task = terminal.json()
                if terminal_task["status"] in {
                    "converged",
                    "max_iterations_reached",
                }:
                    return terminal_task
                time.sleep(poll_seconds)
                continue
            response.raise_for_status()
            time.sleep(poll_seconds)
        raise TimeoutError("worker stopped before the run reached a terminal state")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one local ADMM compute process")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--max-wait-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    try:
        result = run_worker(
            args.base_url,
            args.run_id,
            args.worker_id,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_seconds,
        )
    except httpx.HTTPError as exc:
        print(json.dumps({"event": "worker_failed", "reason": str(exc)}), file=sys.stderr)
        return 2
    except TimeoutError as exc:
        print(json.dumps({"event": "worker_failed", "reason": str(exc)}), file=sys.stderr)
        return 3
    print(json.dumps({"event": "worker_terminal", "task": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
