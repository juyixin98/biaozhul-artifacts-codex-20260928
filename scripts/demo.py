#!/usr/bin/env python3
"""Create the bundled example and print a reproducible run id."""

from __future__ import annotations

import json
from pathlib import Path

import httpx


def main() -> int:
    payload_path = Path(__file__).with_name("..") / "examples" / "create_run.json"
    payload = json.loads(payload_path.resolve().read_text(encoding="utf-8"))
    response = httpx.post("http://127.0.0.1:8000/runs", json=payload, timeout=10)
    response.raise_for_status()
    body = response.json()
    print(json.dumps({"run_id": body["run_id"], "status": body["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
