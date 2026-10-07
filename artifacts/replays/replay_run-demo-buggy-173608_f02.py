#!/usr/bin/env python3
"""Auto-generated minimal replay for run run-demo-buggy-173608.

Failure category : state_conflict
Trigger          : cancel (model in delivered)

Usage:
    python3 replay_run-demo-buggy-173608_f02.py --base-url http://127.0.0.1:8100   # buggy SUT
    python3 replay_run-demo-buggy-173608_f02.py --base-url http://127.0.0.1:8000 --expect-clean
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


SEQUENCE = [['guided-02', 'create'], ['guided-02', 'submit'], ['guided-02', 'approve'], ['guided-02', 'deliver'], ['guided-02', 'cancel']]
ID_PREFIX = 'reduce-run-demo-buggy-173608-'
EXPECT_TRIGGER = {'accepted': True, 'model_predicted_accepted': False}

# Local SUT: bypass any ambient HTTP/SOCKS proxy configuration.
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.ProxyHandler({}))
)


def post(base_url, resource_id, action):
    data = json.dumps({"resource_id": resource_id, "action": action}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/orders",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def uniqueify(seq):
    # Rewrite every concrete (non-ghost) resource id so repeated runs of this
    # script against the same long-lived SUT cannot collide with prior state.
    return [(rid if rid.startswith("ghost-") else ID_PREFIX + rid, a) for rid, a in seq]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument(
        "--expect-clean",
        action="store_true",
        help="run against the reference service; assert the trigger is rejected",
    )
    args = parser.parse_args()

    last = None
    for i, (rid, action) in enumerate(uniqueify(SEQUENCE), start=1):
        status, body = post(args.base_url, rid, action)
        marker = "TRIGGER" if i == len(SEQUENCE) else f"step {i}"
        print(f"{marker:7} {action:8} -> HTTP {status} {body}")
        last = (status, body)

    status, body = last
    trigger_accepted = bool(body.get("ok"))

    if args.expect_clean:
        assert not trigger_accepted, (
            "reference service unexpectedly accepted the conflicting trigger: "
            + json.dumps(body)
        )
        print("OK: reference service rejects the trigger; bug is SUT-specific.")
        return 0

    assert trigger_accepted == EXPECT_TRIGGER["accepted"], (
        f"expected accepted={EXPECT_TRIGGER['accepted']}, got {body!r}"
    )
    print("BUG REPRODUCED: category=state_conflict trigger=cancel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
