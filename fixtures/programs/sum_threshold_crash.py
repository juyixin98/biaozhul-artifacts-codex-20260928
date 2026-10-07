#!/usr/bin/env python3
"""Synthetic controlled program with multiple trigger conditions.

Reads a structured input ({"records": [...]}) from stdin.

Exit codes:
  0  success
  2  other error (malformed input)
  3  TARGET error: "accumulator overflow", raised only when ALL of the
     following hold (multiple trigger conditions):
       - a record {"kind": "mode", "value": 1} is present
       - at least 2 records with kind == "add" are present
       - the sum of their "value" fields is >= 100
"""
import json
import sys


def main() -> int:
    try:
        doc = json.load(sys.stdin)
    except Exception:
        return 2
    records = doc.get("records")
    if not isinstance(records, list):
        return 2
    adds = [r for r in records if isinstance(r, dict) and r.get("kind") == "add"]
    mode_on = any(
        isinstance(r, dict) and r.get("kind") == "mode" and r.get("value") == 1
        for r in records
    )
    total = sum(r.get("value", 0) for r in adds)
    if mode_on and len(adds) >= 2 and total >= 100:
        print("FATAL: accumulator overflow", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
