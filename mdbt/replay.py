"""Deterministic sequence replay against a fresh SUT instance.

Used both by the reducer (every candidate shortening must be re-validated
from a clean state - we never trust that a prefix "obviously" keeps the same
preconditions) and by the report layer to re-run recorded runs.
"""
from __future__ import annotations

from .contracts import Command
from .errors import ResourceExhausted
from .executor import Executor
from .oracle import FSMModel
from .scheduler import SchedulingDecision
from .spec import MachineSpec


def fixed_decision(
    seq: int, command: Command, *, resource_exists: bool, state_hint: str, mode: str
) -> SchedulingDecision:
    return SchedulingDecision(
        seq=seq,
        rng_state_before=[],
        mode=f"replay:{mode}",
        candidates=[command.action],
        chosen=command,
        rationale="replayed from recorded command sequence",
        rng_state_after=[],
        resource_exists=resource_exists,
        model_state_hint=state_hint,
    )


async def replay_commands(
    spec: MachineSpec,
    commands: list[Command],
    transport,
    *,
    max_steps: int | None = None,
    id_prefix: str = "",
    stop_at_first_discrepancy: bool = True,
    target_signature: tuple | None = None,
):
    """Run ``commands`` from a fresh model + fresh SUT.

    Returns ``(steps, finding_or_none)``.  Resource ids may be rewritten with
    ``id_prefix`` (e.g. ``replay-<run>-``) so replays never touch fuzz state.

    Raises :class:`ResourceExhausted` if the replay hits its step budget
    before completing (reported distinctly from discrepancies).
    """
    model = FSMModel.fresh(spec)
    executor = Executor.fresh(model, transport)

    # Track existence purely to rebuild decision metadata; the oracle itself
    # derives resource_exists from its own observed bookkeeping in executor.
    existing: set[str] = set()
    hints: dict[str, str] = {}

    steps = []
    budget = max_steps if max_steps is not None else len(commands)
    for i, original in enumerate(commands[:budget], start=1):
        command = original
        if id_prefix and not original.resource_id.startswith("ghost-"):
            command = Command(
                resource_id=id_prefix + original.resource_id,
                action=original.action,
            )
        exists = command.resource_id in existing
        hint = hints.get(command.resource_id, spec.initial_state)
        decision = fixed_decision(
            i,
            command,
            resource_exists=exists,
            state_hint=hint,
            mode="recorded",
        )
        record = await executor.step(i, command, decision)
        steps.append(record)

        obs = record["observation"]
        if obs["accepted"] and obs["state"] is not None:
            if command.action == "create":
                existing.add(command.resource_id)
            hints[command.resource_id] = obs["state"]

        if record["verdict"]["kind"] == "discrepancy":
            if target_signature is not None:
                from .reducer import discrepancy_signature

                if discrepancy_signature(record) == target_signature:
                    return steps, record
                # Different discrepancy: the model has already resynchronised
                # to the observed state, so keep replaying toward the target.
                continue
            if stop_at_first_discrepancy:
                return steps, record

    if len(commands) > budget:
        raise ResourceExhausted(
            "replay truncated by step budget before reaching the trigger",
            detail={"budget": budget, "commands": len(commands)},
        )
    return steps, None
