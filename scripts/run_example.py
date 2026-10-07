"""Create and execute an MBT run against a local reference or buggy service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("reference", "buggy"))
    parser.add_argument(
        "--backend",
        default="http://127.0.0.1:8000",
        help="MBT FastAPI backend URL",
    )
    parser.add_argument(
        "--reference-url",
        default="http://127.0.0.1:8101",
    )
    parser.add_argument(
        "--buggy-url",
        default="http://127.0.0.1:8102",
    )
    args = parser.parse_args()

    payload = json.loads((ROOT / "examples" / "reference_run.json").read_text())
    payload["model"] = json.loads(
        (ROOT / "fixtures" / "order_model.json").read_text()
    )
    payload["base_url"] = (
        args.reference_url if args.service == "reference" else args.buggy_url
    )

    with httpx.Client(
        base_url=args.backend,
        timeout=10,
        trust_env=False,
    ) as client:
        response = client.post("/runs", json=payload)
        response.raise_for_status()
        result = response.json()
        print(json.dumps(result, indent=2, sort_keys=True))

        if result["status"] == "failed":
            replay = client.get(result["replay_url"])
            replay.raise_for_status()
            output = ROOT / "data" / f"replay_run_{result['run_id']}.py"
            output.write_text(replay.text, encoding="utf-8")
            print(f"minimal replay written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
