#!/usr/bin/env python3
"""Minimal replay for a reduced MBT failure.

Run: python3 SCRIPT_NAME
It uses only Python standard-library HTTP code and exits non-zero if the
observed response differs from the recorded contract.
"""

import json
import sys
import urllib.error
import urllib.request

CONFIG = json.loads('{\n  "base_url": "http://127.0.0.1:8102",\n  "placeholder": "resource_id",\n  "simulate_fault": null,\n  "operations": [\n    {\n      "method": "POST",\n      "path": "/orders",\n      "body": {},\n      "expected_status": 201,\n      "expected_observation": "success",\n      "expected_error": null\n    },\n    {\n      "method": "POST",\n      "path": "/orders/{resource_id}/submit",\n      "body": {},\n      "expected_status": 200,\n      "expected_observation": "success",\n      "expected_error": null\n    },\n    {\n      "method": "POST",\n      "path": "/orders/{resource_id}/close",\n      "body": {},\n      "expected_status": 200,\n      "expected_observation": "success",\n      "expected_error": null\n    },\n    {\n      "method": "POST",\n      "path": "/orders/{resource_id}/close",\n      "body": {},\n      "expected_status": 409,\n      "expected_observation": "rejection",\n      "expected_error": "state_conflict"\n    }\n  ]\n}')

# Keep the replay local even when HTTP_PROXY/ALL_PROXY is configured.
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.ProxyHandler({}))
)


def request(method, url, body, simulate_fault):
    data = None
    headers = {"Accept": "application/json"}
    if simulate_fault:
        headers["X-Simulate-Fault"] = simulate_fault
    if method != "GET":
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            text = response.read().decode("utf-8")
            return response.status, json.loads(text) if text else None
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8")
        return exc.code, json.loads(text) if text else None


def main():
    resource_id = None
    for operation in CONFIG["operations"]:
        path = operation["path"]
        if resource_id is not None:
            path = path.replace(
                "{" + CONFIG["placeholder"] + "}", str(resource_id)
            )
        status, body = request(
            operation["method"],
            CONFIG["base_url"].rstrip("/") + path,
            operation["body"],
            CONFIG["simulate_fault"],
        )
        print(json.dumps({"status": status, "body": body}, sort_keys=True))
        if status != operation["expected_status"]:
            print(
                f"expected status {operation['expected_status']}, got {status}",
                file=sys.stderr,
            )
            return 1
        if operation["expected_error"]:
            if not isinstance(body, dict) or body.get("category") != operation["expected_error"]:
                print(
                    f"expected error category {operation['expected_error']}",
                    file=sys.stderr,
                )
                return 1
        if operation["expected_observation"] == "success" and isinstance(body, dict):
            resource_id = body.get("id", resource_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
