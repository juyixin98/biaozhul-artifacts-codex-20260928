"""The four error categories must stay distinguishable end to end."""
from __future__ import annotations

import asyncio

import pytest

from mdbt.contracts import Command
from mdbt.errors import ErrorCategory, ResourceExhausted
from mdbt.executor import HttpTransport
from mdbt.replay import replay_commands
from mdbt.services.buggy import create_app


def test_unreachable_http_sut_is_resource_exhausted(spec):
    async def scenario():
        transport = HttpTransport("http://127.0.0.1:9", timeout=1.0)
        with pytest.raises(ResourceExhausted) as exc:
            await transport.send(Command("r", "create"))
        assert exc.value.category == ErrorCategory.RESOURCE_EXHAUSTED
        assert "connect" in exc.value.message

    asyncio.run(scenario())


def test_replay_step_budget_is_resource_exhausted(spec):
    async def scenario():
        from mdbt.executor import InProcessTransport

        commands = [
            Command("r", a)
            for a in ["create", "submit", "approve", "deliver", "cancel"]
        ]
        with pytest.raises(ResourceExhausted) as exc:
            await replay_commands(
                spec,
                commands,
                InProcessTransport(create_app()),
                max_steps=2,
                id_prefix="budget-",
            )
        assert exc.value.category == ErrorCategory.RESOURCE_EXHAUSTED
        assert exc.value.detail["budget"] == 2

    asyncio.run(scenario())


def test_input_error_and_state_conflict_are_not_collapsed(spec):
    async def scenario():
        import httpx

        from mdbt.executor import normalise_http
        from mdbt.oracle import FSMModel

        app = create_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://s"
        ) as c:
            # create one order
            await c.post("/orders", json={"resource_id": "r", "action": "create"})

            missing = normalise_http(
                await c.post(
                    "/orders", json={"resource_id": "nope", "action": "submit"}
                )
            )
            assert missing.category == ErrorCategory.INPUT_ERROR

            unknown = normalise_http(
                await c.post(
                    "/orders", json={"resource_id": "r", "action": "freeze"}
                )
            )
            assert unknown.category == ErrorCategory.INPUT_ERROR

            conflict = normalise_http(
                await c.post(
                    "/orders", json={"resource_id": "r", "action": "close"}
                )
            )
            assert conflict.category == ErrorCategory.STATE_CONFLICT

            # Oracle predicts the same split.
            model = FSMModel.fresh(spec)
            model.states["r"] = "draft"
            assert (
                model.expected("r", "close", resource_exists=True).category
                == ErrorCategory.STATE_CONFLICT
            )
            assert (
                model.expected("x", "close", resource_exists=False).category
                == ErrorCategory.INPUT_ERROR
            )

    asyncio.run(scenario())


def test_malformed_response_is_computation_failure():
    import httpx

    from mdbt.executor import normalise_http

    class FakeResponse:
        def __init__(self, status_code, payload, text=None):
            self.status_code = status_code
            self._payload = payload
            self.text = text if text is not None else str(payload)

        def json(self):
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload

    non_json = FakeResponse(200, ValueError("x"), text="not json at all")
    obs = normalise_http(non_json)
    assert obs.category == ErrorCategory.COMPUTATION_FAILURE
    assert "non-JSON" in obs.detail

    bad_envelope = FakeResponse(200, {"surprise": True})
    obs = normalise_http(bad_envelope)
    assert obs.category == ErrorCategory.COMPUTATION_FAILURE
    assert "envelope" in obs.detail
