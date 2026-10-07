#!/usr/bin/env bash
# Service call examples. Assumes scripts/run_demo.sh is running, or start:
#   python3 -m uvicorn mdbt.services.reference:app --port 8000
#   python3 -m uvicorn mdbt.services.buggy:app     --port 8100
#   python3 -m uvicorn mdbt.api:app               --port 8200
set -euo pipefail

MODEL=$(cat fixtures/order_machine.json)

echo "== 1. validate the model =="
curl -sS -X POST http://127.0.0.1:8200/models/validate \
    -H 'Content-Type: application/json' \
    -d "{\"model\": $MODEL}" | python3 -m json.tool

echo "== 2. run the tester against the CORRECT reference service =="
curl -sS -X POST http://127.0.0.1:8200/runs \
    -H 'Content-Type: application/json' \
    -d "$(python3 - <<PY
import json
print(json.dumps({"sut": "reference", "model": json.loads('''$MODEL'''),
                  "seed": 20261007, "fuzz_steps": 40}))
PY
)" | python3 -m json.tool

echo "== 3. run the tester against the BUGGY service =="
RESP=$(curl -sS -X POST http://127.0.0.1:8200/runs \
    -H 'Content-Type: application/json' \
    -d "$(python3 - <<PY
import json
print(json.dumps({"sut": "buggy", "model": json.loads('''$MODEL'''),
                  "seed": 20261007, "fuzz_steps": 40}))
PY
)")
echo "$RESP" | python3 -m json.tool
RUN_ID=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')

echo "== 4. inspect per-step evidence for $RUN_ID =="
curl -sS "http://127.0.0.1:8200/runs/$RUN_ID/steps" | python3 -m json.tool | head -60

echo "== 5. list findings with failure categories =="
curl -sS "http://127.0.0.1:8200/runs/$RUN_ID/findings" | python3 -m json.tool

echo "== 6. fetch the minimal replay script for finding 0 =="
curl -sS "http://127.0.0.1:8200/runs/$RUN_ID/findings/0/replay"

echo "== 7. drive the SUTs directly (repeat-close scenario) =="
for PORT in 8100 8000; do
    echo "-- port $PORT --"
    RID="curl-demo-$PORT"
    for ACTION in create submit approve deliver close close; do
        curl -sS -X POST "http://127.0.0.1:$PORT/orders" \
            -H 'Content-Type: application/json' \
            -d "{\"resource_id\": \"$RID\", \"action\": \"$ACTION\"}"
        echo
    done
done
