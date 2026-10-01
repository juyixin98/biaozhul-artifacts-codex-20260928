#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

curl -s "${BASE_URL}/health"
echo

curl -s "${BASE_URL}/versions"
echo

curl -s -X POST "${BASE_URL}/datasets/generate" \
  -H "content-type: application/json" \
  -d '{
    "request_id": "example-generate-mar",
    "n": 200,
    "mechanism": "mar",
    "missing_probability": 0.2,
    "seed": 1729
  }'
echo

curl -s -X POST "${BASE_URL}/analyses" \
  -H "content-type: application/json" \
  -d '{
    "request_id": "example-mi",
    "method": "multiple_imputation",
    "m": 4,
    "seed": 1729,
    "records": [
      {"record_id":"nonuser-001","age_z":-1.0,"member":0,"used_support":0,"contact_frequency":0,"rating":0},
      {"record_id":"nonuser-002","age_z":0.0,"member":0,"used_support":0,"contact_frequency":0,"rating":0},
      {"record_id":"member-003","age_z":0.0,"member":1,"used_support":1,"contact_frequency":3,"rating":7},
      {"record_id":"member-004","age_z":1.0,"member":1,"used_support":1,"contact_frequency":4,"rating":null},
      {"record_id":"member-005","age_z":0.0,"member":1,"used_support":1,"contact_frequency":5,"rating":8},
      {"record_id":"guest-006","age_z":-1.0,"member":0,"used_support":1,"contact_frequency":2,"rating":null},
      {"record_id":"guest-007","age_z":0.0,"member":0,"used_support":1,"contact_frequency":3,"rating":6},
      {"record_id":"member-008","age_z":1.0,"member":1,"used_support":1,"contact_frequency":6,"rating":9},
      {"record_id":"member-009","age_z":0.0,"member":1,"used_support":1,"contact_frequency":4,"rating":null},
      {"record_id":"member-010","age_z":0.0,"member":1,"used_support":1,"contact_frequency":3,"rating":7}
    ]
  }'
echo

curl -s "${BASE_URL}/analyses/example-mi"
echo
