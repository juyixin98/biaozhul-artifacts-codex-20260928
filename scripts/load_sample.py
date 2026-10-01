"""Load the local synthetic sample into a running service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--sample",
        default=str(Path(__file__).resolve().parents[1] / "data" / "sample_problem.json"),
    )
    args = parser.parse_args()
    payload = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    response = httpx.post(
        f"{args.base_url}/v1/problems",
        json=payload,
        headers={"X-Request-ID": "sample-loader"},
        timeout=10,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
