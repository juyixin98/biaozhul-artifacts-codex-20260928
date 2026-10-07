"""Generate a standalone, stdlib-only minimal replay script.

The generated script:

* needs only a running SUT URL (default ``http://127.0.0.1:8100``);
* replays the minimised command sequence with fresh resource ids;
* asserts the exact observed discrepancy, exits non-zero if it disappears;
* can be pointed at the reference service to confirm the bug is SUT-specific
  (``--expect-clean`` then asserts the model-conflicting call is rejected).
"""
from __future__ import annotations

import json

from .contracts import Command


TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated minimal replay for run {run_id}.

Failure category : {category}
Trigger          : {action} (model in {model_state})

Usage:
    python3 {script_name} --base-url http://127.0.0.1:8100   # buggy SUT
    python3 {script_name} --base-url http://127.0.0.1:8000 --expect-clean
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


SEQUENCE = {sequence}
ID_PREFIX = {id_prefix}
EXPECT_TRIGGER = {expect}

# Local SUT: bypass any ambient HTTP/SOCKS proxy configuration.
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
)


def post(base_url, resource_id, action):
    data = json.dumps({{"resource_id": resource_id, "action": action}}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/orders",
        data=data,
        headers={{"Content-Type": "application/json"}},
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
        marker = "TRIGGER" if i == len(SEQUENCE) else f"step {{i}}"
        print(f"{{marker:7}} {{action:8}} -> HTTP {{status}} {{body}}")
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
        f"expected accepted={{EXPECT_TRIGGER['accepted']}}, got {{body!r}}"
    )
    print("BUG REPRODUCED: category={category} trigger={action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def render_replay_script(
    *,
    run_id: str,
    commands: list[Command],
    category: str,
    trigger_step: dict,
    script_name: str,
    id_prefix: str,
) -> str:
    sequence = [[c.resource_id, c.action] for c in commands]
    expect = {
        "accepted": trigger_step["observation"]["accepted"],
        "model_predicted_accepted": trigger_step["expected"]["accepted"],
    }
    return TEMPLATE.format(
        run_id=run_id,
        script_name=script_name,
        category=category,
        action=trigger_step["command"]["action"],
        model_state=trigger_step["model_state_before"],
        sequence=repr(sequence),
        expect=repr(expect),
        id_prefix=repr(id_prefix),
    )
