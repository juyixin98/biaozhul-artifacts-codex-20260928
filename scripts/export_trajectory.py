#!/usr/bin/env python3
"""Export a run trajectory to CSV with pandas."""

from __future__ import annotations

import sys

import httpx
import pandas as pd


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: export_trajectory.py BASE_URL RUN_ID OUTPUT_CSV", file=sys.stderr)
        return 2
    base_url, run_id, output_path = argv[1:]
    response = httpx.get(
        f"{base_url.rstrip('/')}/runs/{run_id}/trajectory", timeout=10
    )
    response.raise_for_status()
    frame = pd.DataFrame(response.json()["trajectory"])
    frame.to_csv(output_path, index=False)
    print(f"wrote {len(frame)} rounds to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
