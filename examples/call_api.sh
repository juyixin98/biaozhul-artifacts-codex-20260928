#!/usr/bin/env bash
set -euo pipefail

curl -sS http://127.0.0.1:8000/api/solve \
  -H 'content-type: application/json' \
  --data-binary @examples/knapsack.json
