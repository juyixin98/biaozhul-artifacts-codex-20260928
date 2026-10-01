#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from change_detection.experiments import run_scenarios  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible local change-point scenarios.")
    parser.add_argument("--output-dir", default="experiments/latest")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    frame = run_scenarios(output_dir=args.output_dir, run_id=args.run_id)
    print(frame.to_string(index=False))
    failures = int((frame["status"] == "FAIL").sum())
    print(f"\n{len(frame) - failures}/{len(frame)} scenarios passed; artifacts: {args.output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
