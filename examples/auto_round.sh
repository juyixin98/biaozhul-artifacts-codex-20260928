#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
RUN_ID="$(curl -sS -X POST "$BASE_URL/runs" \
  -H 'content-type: application/json' \
  --data-binary @examples/asymmetric_problem.json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["run"]["run_id"])')"

echo "run_id=$RUN_ID"
python3 - "$BASE_URL" "$RUN_ID" <<'PY'
import json
import sys

import httpx

base_url, run_id = sys.argv[1:]
with httpx.Client(base_url=base_url) as client:
    while True:
        run = client.post(f"/runs/{run_id}/auto-round").json()["run"]
        latest = run["trajectory"][-1]
        print(
            latest["round"],
            "primal",
            f'{latest["primal_residual"]:.3e}',
            "dual",
            f'{latest["dual_residual"]:.3e}',
            "rho",
            f'{latest["rho"]:.3f}',
        )
        if run["status"] != "running":
            print(json.dumps({"status": run["status"], "z": run["z"], "reference": run["reference"]}, indent=2))
            break
PY

curl -sS "$BASE_URL/runs/$RUN_ID/events" | python3 -m json.tool > "examples/${RUN_ID}-events.json"
echo "wrote examples/${RUN_ID}-events.json"
