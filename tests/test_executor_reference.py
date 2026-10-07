"""Reference service behaviour + no-false-positive campaign."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from mdbt.campaign import run_campaign
from mdbt.executor import InProcessTransport
from mdbt.oracle import FSMModel
from mdbt.scheduler import SchedulingDecision
from mdbt.services.reference import create_app
from mdbt.storage import EvidenceStore


async def _post(app, rid, action):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sut"
    ) as client:
        resp = await client.post(
            "/orders", json={"resource_id": rid, "action": action}
        )
        return resp.status_code, resp.json()


def test_reference_full_happy_path():
    app = create_app()

    async def scenario():
        rid = "ref-1"
        expected = [
            ("create", 201, "draft"),
            ("submit", 200, "submitted"),
            ("approve", 200, "paid"),
            ("deliver", 200, "delivered"),
            ("close", 200, "closed"),
        ]
        for action, want_status, want_state in expected:
            status, body = await _post(app, rid, action)
            assert status == want_status, body
            assert body["ok"] is True and body["state"] == want_state

    asyncio.run(scenario())


def test_reference_rejects_all_three_planted_bug_histories():
    app = create_app()

    async def scenario():
        # repeated close
        rid = "dup"
        for a in ["create", "submit", "approve", "deliver", "close"]:
            await _post(app, rid, a)
        status, body = await _post(app, rid, "close")
        assert status == 409 and body["ok"] is False
        assert body["state"] == "closed"

        # out-of-order submit after payment
        rid2 = "oos"
        for a in ["create", "submit", "approve"]:
            await _post(app, rid2, a)
        status, body = await _post(app, rid2, "submit")
        assert status == 409 and body["state"] == "paid"

        # cancel after delivery
        rid3 = "cad"
        for a in ["create", "submit", "approve", "deliver"]:
            await _post(app, rid3, a)
        status, body = await _post(app, rid3, "cancel")
        assert status == 409 and body["state"] == "delivered"

    asyncio.run(scenario())


def test_reference_error_categories_are_distinct():
    app = create_app()

    async def scenario():
        # missing resource -> 404 input_error
        status, _ = await _post(app, "never-created", "submit")
        assert status == 404
        # unknown action -> 400 input_error
        await _post(app, "u", "create")
        status, body = await _post(app, "u", "teleport")
        assert status == 400 and body["ok"] is False
        # legal-state conflict -> 409 state_conflict
        status, body = await _post(app, "u", "deliver")
        assert status == 409 and body["state"] == "draft"

    asyncio.run(scenario())


def test_executor_predicts_before_calling_and_updates_after_observation(spec):
    app = create_app()

    async def scenario():
        model = FSMModel.fresh(spec)
        from mdbt.contracts import Command
        from mdbt.executor import Executor

        ex = Executor.fresh(model, InProcessTransport(app))
        decision = SchedulingDecision(
            seq=1, rng_state_before=[], mode="t", candidates=["create"],
            chosen=Command("o1", "create"), rationale="",
            rng_state_after=[], resource_exists=False, model_state_hint="draft",
        )
        record = await ex.step(1, Command("o1", "create"), decision)
        assert record["model_state_before"] == "draft"
        assert record["expected"]["accepted"] is True
        assert record["observation"]["accepted"] is True
        assert record["model_state_after"] == "draft"
        assert record["verdict"]["kind"] == "consistent"

    asyncio.run(scenario())


def test_reference_campaign_has_zero_findings(spec, db_path, replay_dir):
    async def scenario():
        with EvidenceStore(db_path) as store:
            rid = store.new_run_id()
            result = await run_campaign(
                spec=spec,
                sut_name="reference",
                transport_factory=lambda: InProcessTransport(create_app()),
                store=store,
                run_id=rid,
                seed=42,
                fuzz_steps=30,
                replay_dir=replay_dir,
            )
        assert result["status"] == "completed"
        assert result["summary"]["findings"] == 0
        assert result["summary"]["finding_categories"] == []
        assert (
            result["summary"]["consistent_steps"]
            == result["summary"]["steps_executed"]
        )
        assert result["findings"] == []

    asyncio.run(scenario())
