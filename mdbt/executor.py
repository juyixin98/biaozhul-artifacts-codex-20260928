"""Execution kernel: oracle prediction and SUT observation are separated.

Per-command protocol (see :func:`Executor.step`)::

    expected = model.expected(command)          # pure, no SUT, no mutation
    raw      = transport.send(command)          # the *only* SUT interaction
    obs      = normalise(raw)
    verdict  = compare(expected, obs)
    model.apply_observation(obs)                # model moves after observation

Transports adapt either an in-process FastAPI app (tests/demo) or a real
HTTP URL (replay scripts) behind one method, so the kernel never needs to
know which one it is talking to.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from .contracts import Command, Observation, Verdict
from .errors import (
    ErrorCategory,
    HTTP_STATUS_CATEGORY,
    ResourceExhausted,
)
from .oracle import ExpectedOutcome, FSMModel
from .scheduler import SchedulingDecision


class Transport:
    async def send(self, command: Command) -> Observation:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class InProcessTransport(Transport):
    """Call a FastAPI SUT app over the ASGI stack (no network port)."""

    app: object
    timeout: float = 5.0

    async def send(self, command: Command) -> Observation:
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app),
                base_url="http://sut.local",
                timeout=self.timeout,
                trust_env=False,
            ) as client:
                response = await client.post(
                    "/orders",
                    json={
                        "resource_id": command.resource_id,
                        "action": command.action,
                    },
                )
        except httpx.TimeoutException as exc:
            raise ResourceExhausted(
                f"SUT request timed out after {self.timeout}s",
                detail={"resource_id": command.resource_id, "action": command.action},
            ) from exc
        except Exception as exc:  # malformed transport-level failure
            return Observation(
                accepted=False,
                status_code=0,
                state=None,
                resource_exists=False,
                category=ErrorCategory.COMPUTATION_FAILURE,
                detail=f"transport error: {type(exc).__name__}: {exc}",
            )
        return normalise_http(response)


@dataclass
class HttpTransport(Transport):
    """Call a running SUT over real HTTP."""

    base_url: str
    timeout: float = 5.0

    async def send(self, command: Command) -> Observation:
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout, trust_env=False
            ) as client:
                response = await client.post(
                    "/orders",
                    json={
                        "resource_id": command.resource_id,
                        "action": command.action,
                    },
                )
        except httpx.TimeoutException as exc:
            raise ResourceExhausted(
                f"SUT request timed out after {self.timeout}s",
                detail={"resource_id": command.resource_id, "action": command.action},
            ) from exc
        except httpx.ConnectError as exc:
            raise ResourceExhausted(
                f"cannot connect to SUT at {self.base_url}",
                detail={"base_url": self.base_url},
            ) from exc
        except Exception as exc:
            return Observation(
                accepted=False,
                status_code=0,
                state=None,
                resource_exists=False,
                category=ErrorCategory.COMPUTATION_FAILURE,
                detail=f"transport error: {type(exc).__name__}: {exc}",
            )
        return normalise_http(response)


def normalise_http(response: httpx.Response) -> Observation:
    """Convert a raw SUT HTTP response into the wire contract."""
    status = response.status_code
    category = HTTP_STATUS_CATEGORY.get(status, ErrorCategory.COMPUTATION_FAILURE)
    try:
        body = response.json()
    except Exception:
        return Observation(
            accepted=False,
            status_code=status,
            state=None,
            resource_exists=False,
            category=ErrorCategory.COMPUTATION_FAILURE,
            detail=f"SUT returned non-JSON body: {response.text[:120]!r}",
        )
    if not isinstance(body, dict) or "ok" not in body:
        return Observation(
            accepted=False,
            status_code=status,
            state=None,
            resource_exists=False,
            category=ErrorCategory.COMPUTATION_FAILURE,
            detail=f"SUT response missing 'ok' envelope: {body!r}"[:300],
        )
    accepted = bool(body.get("ok"))
    state = body.get("state")
    if state is not None and not isinstance(state, str):
        return Observation(
            accepted=False,
            status_code=status,
            state=None,
            resource_exists=False,
            category=ErrorCategory.COMPUTATION_FAILURE,
            detail=f"SUT reported non-string state: {state!r}",
        )
    exists = bool(body.get("exists", status != 404))
    detail = str(body.get("error", "")) if not accepted else str(body.get("message", ""))
    return Observation(
        accepted=accepted,
        status_code=status,
        state=state,
        resource_exists=exists,
        category=None if accepted else category,
        detail=detail,
    )


def compare(expected: ExpectedOutcome, obs: Observation) -> Verdict:
    """Decide whether an observation matches the pure prediction."""
    if expected.accepted != obs.accepted:
        want = "accepted" if expected.accepted else "rejected"
        got = "accepted" if obs.accepted else "rejected"
        # On an accept/reject split the semantic category is the *model's*
        # predicted conflict class (state_conflict / input_error).  The SUT's
        # own status (often 200 on a wrongful accept) cannot classify its bug.
        cat = (
            expected.category
            if expected.category is not None
            else (obs.category or ErrorCategory.COMPUTATION_FAILURE)
        )
        return Verdict(
            kind="discrepancy",
            category=cat,
            rationale=(
                f"model predicted {want} ({expected.reason}) but SUT {got}: "
                f"HTTP {obs.status_code} {obs.detail}"
            ),
        )

    if expected.accepted:
        if obs.state != expected.to_state:
            return Verdict(
                kind="discrepancy",
                category=ErrorCategory.STATE_CONFLICT,
                rationale=(
                    f"accepted but wrong target state: expected "
                    f"{expected.to_state!r}, SUT reported {obs.state!r}"
                ),
            )
        return Verdict(
            kind="consistent",
            category=None,
            rationale=f"accepted, reached {obs.state!r} as predicted",
        )

    # Both rejected: the rejection *category* must also agree.
    if obs.category != expected.category:
        return Verdict(
            kind="discrepancy",
            category=obs.category or ErrorCategory.COMPUTATION_FAILURE,
            rationale=(
                f"both rejected, but wrong error category: model expected "
                f"{expected.category.value if expected.category else None}, "
                f"SUT gave {obs.category.value if obs.category else None} "
                f"(HTTP {obs.status_code})"
            ),
        )
    return Verdict(
        kind="consistent",
        category=None,
        rationale=f"rejected as predicted ({obs.category.value if obs.category else 'n/a'})",
    )


@dataclass
class Executor:
    model: FSMModel
    transport: Transport
    known_resources: set[str]

    @classmethod
    def fresh(cls, model: FSMModel, transport: Transport) -> "Executor":
        return cls(model=model, transport=transport, known_resources=set())

    async def step(self, seq: int, command: Command, decision: SchedulingDecision) -> dict:
        """Execute one command and return a serialisable step record."""
        rid = command.resource_id
        state_before = self.model.state_of(rid)
        resource_exists = command.resource_id in self.known_resources

        # 1) pure prediction, model untouched
        expected = self.model.expected(
            rid, command.action, resource_exists=resource_exists
        )

        # 2) the single SUT interaction
        obs = await self.transport.send(command)

        # 3) verdict from prediction + observation only
        verdict = compare(expected, obs)

        # 4) bookkeeping learned from the observation
        if command.action == "create" and obs.accepted and obs.state is not None:
            self.known_resources.add(command.resource_id)
        if not obs.resource_exists and obs.status_code == 404:
            self.known_resources.discard(command.resource_id)

        # 5) advance the model strictly from the observation
        new_state, resynced = self.model.apply_observation(
            rid,
            command.action,
            observed_accepted=obs.accepted,
            observed_state=obs.state,
            resource_exists_after=command.resource_id in self.known_resources
            or (obs.resource_exists and obs.state is not None),
        )

        return {
            "seq": seq,
            "command": command.to_dict(),
            "decision": decision.to_dict(),
            "model_state_before": state_before,
            "expected": expected.to_dict(),
            "observation": obs.to_dict(),
            "model_state_after": new_state,
            "resynced": resynced,
            "verdict": verdict.to_dict(),
        }
