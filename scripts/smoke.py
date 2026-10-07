"""Smoke check against a running service instance (see README)."""

import sys

import httpx

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

doc_text = open("fixtures/sample_input.txt", encoding="utf-8").read()
lines = [ln for ln in doc_text.splitlines() if ln.strip()]
blocks = (
    [{"kind": "header", "text": lines[0]}]
    + [{"kind": "setting", "text": ln} for ln in lines[1:-1]]
    + [{"kind": "footer", "text": lines[-1]}]
)

r = httpx.post(f"{base}/jobs", json={"blocks": blocks, "max_steps": 1})
print("create:", r.status_code, r.json()["status"])
job_id = r.json()["job_id"]

r = httpx.post(f"{base}/jobs/{job_id}/resume")
body = r.json()
print("resume:", r.status_code, body["status"], "|", body["minimality_claim"])

r = httpx.get(f"{base}/jobs/{job_id}/report")
print("report:", r.json()["final_doc_masked"])
