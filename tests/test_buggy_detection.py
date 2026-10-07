"""The tester must detect all three planted stateful bugs and classify them."""
from __future__ import annotations

import asyncio

import httpx

from mdbt.campaign import run_campaign
from mdbt.errors import ErrorCategory
from mdbt.executor import InProcessTransport
from mdbt.services.buggy import create_app
from mdbt.storage import EvidenceStore


async def _post(app, rid, action):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sut"
    ) as client:
        resp = await client.post(
            "/orders", json={"resource_id": rid, "action": action}
        )
        return resp.status_code, resp.json()


def test_buggy_service_exhibits_three_planted_bugs():
    app = create_app()

    async def scenario():
        # 1. repeated close accepted
        rid = "dup"
        for a in ["create", "submit", "approve", "deliver", "close"]:
            await _post(app, rid, a)
        status, body = await _post(app, rid, "close")
        assert (status, body["ok"], body["state"]) == (200, True, "closed")

        # 2. out-of-order submit rolls paid -> submitted
        rid2 = "oos"
        for a in ["create", "submit", "approve"]:
            await _post(app, rid2, a)
        status, body = await _post(app, rid2, "submit")
        assert (status, body["ok"], body["state"]) == (200, True, "submitted")

        # 3. cancel after delivery accepted
        rid3 = "cad"
        for a in ["create", "submit", "approve", "deliver"]:
            await _post(app, rid3, a)
        status, body = await _post(app, rid3, "cancel")
        assert (status, body["ok"], body["state"]) == (200, True, "cancelled")

        # input handling is still correct (bugs are state-specific)
        status, _ = await _post(app, "ghost", "submit")
        assert status == 404
        status, body = await _post(app, "dup", "freeze")
        assert status == 400

    asyncio.run(scenario())


def test_campaign_finds_exactly_three_state_conflicts(spec, db_path, replay_dir):
    async def scenario():
        with EvidenceStore(db_path) as store:
            rid = store.new_run_id()
            result = await run_campaign(
                spec=spec,
                sut_name="buggy-order-service",
                transport_factory=lambda: InProcessTransport(create_app()),
                store=store,
                run_id=rid,
                seed=7,
                fuzz_steps=0,
                replay_dir=replay_dir,
            )
        assert result["status"] == "completed"
        assert result["summary"]["findings"] == 3
        assert result["summary"]["finding_categories"] == [
            ErrorCategory.STATE_CONFLICT.value
        ]

        by_action = {f["command"]["action"]: f for f in result["findings"]}
        assert set(by_action) == {"close", "submit", "cancel"}

        close = by_action["close"]
        assert close["title"].startswith("SUT wrongly accepted illegal 'close'")

        for action, want_length in [("close", 6), ("submit", 4), ("cancel", 5)]:
            f = by_action[action]
            assert f["category"] == ErrorCategory.STATE_CONFLICT.value
            assert f["reduced"] is True
            assert f["reduced_length"] == want_length
            assert f["replay_script"].startswith("#!/usr/bin/env python3")
            assert "BUG REPRODUCED" in f["replay_script"]
            assert f["reduction_attempted"] is True
            # The trigger was never deleted: last reduced command is the action.
            assert f["replay_script"]

    asyncio.run(scenario())


def test_deterministic_reproduction_with_same_seed(spec, tmp_path):
    async def once(db, replay):
        with EvidenceStore(db) as store:
            rid = store.new_run_id()
            return await run_campaign(
                spec=spec,
                sut_name="buggy-order-service",
                transport_factory=lambda: InProcessTransport(create_app()),
                store=store,
                run_id=rid,
                seed=123,
                fuzz_steps=15,
                replay_dir=replay,
            )

    async def scenario():
        r1 = await once(str(tmp_path / "a.db"), str(tmp_path / "ra"))
        r2 = await once(str(tmp_path / "b.db"), str(tmp_path / "rb"))
        assert r1["summary"] == r2["summary"]

    asyncio.run(scenario())
