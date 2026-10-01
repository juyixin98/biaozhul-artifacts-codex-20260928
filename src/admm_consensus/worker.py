"""Command-line local compute process for one ADMM worker.

Example:
    python -m admm_consensus.worker http://127.0.0.1:8000 <run-id> 1
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx
import numpy as np

from .kernel import solve_local_update
from .models import QuadraticModel


def run_worker(base_url: str, run_id: str, worker_id: int, *, poll_seconds: float = 0.05) -> int:
    submitted_rounds: set[int] = set()
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=5.0) as client:
        while True:
            response = client.get(f"/runs/{run_id}")
            response.raise_for_status()
            run = response.json()["run"]
            if run["status"] != "running":
                print(f"worker {worker_id}: terminal status {run['status']}", flush=True)
                return 0

            task_response = client.get(f"/runs/{run_id}/task")
            task_response.raise_for_status()
            task = task_response.json()["task"]
            round_number = int(task["round_number"])
            if round_number in submitted_rounds or worker_id in task["submitted_worker_ids"]:
                time.sleep(poll_seconds)
                continue

            payload = next(w for w in task["workers"] if int(w["worker_id"]) == worker_id)
            model = QuadraticModel(Q=np.asarray(payload["Q"], dtype=float), q=np.asarray(payload["q"], dtype=float))
            x = solve_local_update(
                model,
                z=np.asarray(payload["z"], dtype=float),
                u=np.asarray(payload["u"], dtype=float),
                rho=float(task["rho"]),
                worker_id=worker_id,
            )
            submit = client.post(
                f"/runs/{run_id}/submit",
                json={
                    "worker_id": worker_id,
                    "round_number": round_number,
                    "rho": task["rho"],
                    "x": x.tolist(),
                },
            )
            if submit.status_code == 409:
                # The other process can trigger a timeout and discard a late submission.
                print(
                    f"worker {worker_id}: round {round_number} submission rejected: {submit.text}",
                    flush=True,
                )
                time.sleep(poll_seconds)
                continue
            submit.raise_for_status()
            submitted_rounds.add(round_number)
            print(f"worker {worker_id}: submitted round {round_number}", flush=True)
            time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one local consensus ADMM compute process.")
    parser.add_argument("base_url")
    parser.add_argument("run_id")
    parser.add_argument("worker_id", choices=(1, 2), type=int)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        return run_worker(
            args.base_url,
            args.run_id,
            args.worker_id,
            poll_seconds=args.poll_seconds,
        )
    except httpx.HTTPError as exc:
        print(f"worker {args.worker_id}: HTTP failure: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
