#!/usr/bin/env python3
"""Print a persisted run and the intermediate states needed to replay it."""

from __future__ import annotations

import argparse
import json
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="Client-supplied or generated run identifier")
    parser.add_argument("--database", default="stl_runs.sqlite3")
    parser.add_argument("--report", help="Optional JSON path for replay output")
    args = parser.parse_args()

    with sqlite3.connect(args.database) as conn:
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (args.run_id,)
        ).fetchone()
        if run is None:
            print(f"unknown run_id: {args.run_id}")
            return 2
        events = conn.execute(
            "SELECT * FROM run_events WHERE run_id = ? ORDER BY id",
            (args.run_id,),
        ).fetchall()

    payload = {
        "run_number": run["id"],
        "run_id": run["run_id"],
        "status": run["status"],
        "observations": run["observations"],
        "request_fingerprint": run["request_fingerprint"],
        "kernel_version": run["kernel_version"],
        "schema_version": run["schema_version"],
        "error": {
            "category": run["error_category"],
            "code": run["error_code"],
            "message": run["error_message"],
        }
        if run["status"] == "failed"
        else None,
        "events": [
            {
                "event_id": event["id"],
                "run_number": event["run_number"],
                "iteration": event["iteration"],
                "stage": event["stage"],
                "judgment": event["judgment"],
                "state": json.loads(event["state_json"]),
                "created_at": event["created_at"],
            }
            for event in events
        ],
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
