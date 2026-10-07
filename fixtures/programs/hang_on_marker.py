#!/usr/bin/env python3
"""Synthetic controlled program that hangs on a marker record.

Used to exercise the INDETERMINATE failure failure category: when a record
with kind == "hang" is present the program never terminates, so the
oracle times out and the verdict is indeterminate.
"""
import json
import sys
import time


def main() -> int:
    try:
        doc = json.load(sys.stdin)
    except Exception:
        return 2
    records = doc.get("records") or []
    if any(isinstance(r, dict) and r.get("kind") == "hang" for r in records):
        time.sleep(3600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
