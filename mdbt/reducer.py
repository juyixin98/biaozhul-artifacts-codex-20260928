"""Sequence minimisation with precondition preservation/revalidation.

Naive delta-debugging can "reduce" into a sequence whose trigger now means
something different (e.g. dropping the ``deliver`` turns a repeated ``close``
into a plain legal one).  This reducer therefore:

1. greedily removes one command at a time, in a fresh SUT each attempt;
2. replays the shortened sequence from a clean initial state;
3. accepts a removal only if a discrepancy with the **same signature** still
    occurs (accepted/rejected flag, category, trigger action, and the model
    state the trigger was predicted from);
4. repeats to a fixed point and records every accept/reject decision.

The original trigger command is never deleted; dependencies it actually needs
are protected automatically because removing them changes the signature (the
replay revalidates preconditions instead of assuming them).
"""
from __future__ import annotations

from dataclasses import dataclass

from .contracts import Command
from .errors import ErrorCategory
from .replay import replay_commands
from .spec import MachineSpec


def discrepancy_signature(trigger_step: dict) -> tuple:
    """Identity of a discrepancy, used to judge whether reduction preserved it."""
    expected = trigger_step["expected"]
    obs = trigger_step["observation"]
    return (
        expected["accepted"],
        obs["accepted"],
        obs.get("category"),
        trigger_step["command"]["action"],
        trigger_step["model_state_before"],
        obs.get("state"),
    )


@dataclass
class ReductionResult:
    commands: list[Command]
    steps: list[dict]
    trigger_step: dict
    log: list[dict]


async def reduce_sequence(
    spec: MachineSpec,
    commands: list[Command],
    trigger_seq: int,
    target_signature: tuple,
    transport_factory,
    *,
    id_prefix: str = "reduce-",
) -> ReductionResult:
    """Minimise ``commands`` while preserving ``target_signature``.

    ``transport_factory`` must return a *fresh* transport per call. For local
    runs it wraps a newly constructed SUT app (clean in-memory state); for
    HTTP runs it returns a new client while ``id_prefix`` isolates state.
    """
    current = list(commands[:trigger_seq])
    log: list[dict] = []

    async def _run(seq: list[Command]):
        transport = transport_factory()
        steps, finding_step = await replay_commands(
            spec,
            seq,
            transport,
            id_prefix=id_prefix,
            target_signature=target_signature,
        )
        if finding_step is None:
            return steps, None
        return steps, finding_step

    changed = True
    pass_index = 0
    while changed:
        changed = False
        pass_index += 1
        i = 0
        while i < len(current):
            original_cmd = current[i]
            # Never delete the final trigger itself.
            if i == len(current) - 1:
                i += 1
                continue
            candidate = current[:i] + current[i + 1 :]
            steps, finding_step = await _run(candidate)
            decision: dict
            if finding_step is not None and discrepancy_signature(finding_step) == target_signature:
                decision = {
                    "attempt": f"pass{pass_index}",
                    "removed": original_cmd.to_dict(),
                    "accepted": True,
                    "new_length": len(candidate),
                    "reason": "same discrepancy signature reproduced from clean state",
                }
                current = candidate
                changed = True
                # Do not advance i: indices shifted after removal.
            else:
                decision = {
                    "attempt": f"pass{pass_index}",
                    "removed": original_cmd.to_dict(),
                    "accepted": False,
                    "length": len(candidate),
                    "reason": (
                        "precondition lost"
                        if finding_step is None
                        else "discrepancy signature changed"
                    ),
                }
                i += 1
            log.append(decision)

    steps, finding_step = await _run(current)
    if finding_step is None:
        raise RuntimeError(
            "reducer internal error: final replay lost the discrepancy"
        )
    return ReductionResult(
        commands=current,
        steps=steps,
        trigger_step=finding_step,
        log=log,
    )
