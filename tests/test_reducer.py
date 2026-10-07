"""Reducer must preserve preconditions (revalidate, never assume)."""
from __future__ import annotations

import asyncio

from mdbt.contracts import Command
from mdbt.executor import InProcessTransport
from mdbt.reducer import discrepancy_signature, reduce_sequence
from mdbt.replay import replay_commands
from mdbt.services.buggy import create_app


def _factory():
    return InProcessTransport(create_app())


def test_reducer_strips_noise_but_keeps_required_history(spec):
    async def scenario():
        # Noisy sequence: an unrelated resource interleaved, plus illegal
        # noise commands that do not affect the target resource.
        noisy = [
            Command("noise", "create"),
            Command("noise", "submit"),
            Command("main", "create"),
            Command("ghost-00001", "approve"),     # missing resource
            Command("main", "submit"),
            Command("noise", "approve"),
            Command("main", "approve"),
            Command("main", "freeze"),            # unknown action (illegal input)
            Command("main", "deliver"),
            Command("main", "cancel"),            # planted bug trigger
        ]
        steps, trigger = await replay_commands(
            spec, noisy, _factory(), id_prefix="t-"
        )
        # The noise resource itself hits a discrepancy; derive the target
        # signature from a clean run and require replay to continue past the
        # unrelated one.
        clean = [
            Command("main", a)
            for a in ["create", "submit", "approve", "deliver", "cancel"]
        ]
        _, clean_trigger = await replay_commands(
            spec, clean, _factory(), id_prefix="t-clean-"
        )
        assert clean_trigger is not None
        signature = discrepancy_signature(clean_trigger)
        steps, trigger = await replay_commands(
            spec, noisy, _factory(), id_prefix="t-",
            target_signature=signature,
        )
        assert trigger is not None
        assert trigger["command"]["action"] == "cancel"

        result = await reduce_sequence(
            spec,
            noisy,
            trigger_seq=len(steps),
            target_signature=signature,
            transport_factory=_factory,
            id_prefix="t-",
        )
        # Minimal history: create submit approve deliver cancel (5 commands),
        # noise resource and the illegal input are removed.
        assert [(c.resource_id, c.action) for c in result.commands] == [
            ("main", "create"),
            ("main", "submit"),
            ("main", "approve"),
            ("main", "deliver"),
            ("main", "cancel"),
        ]
        assert discrepancy_signature(result.trigger_step) == signature

        rejected = [d for d in result.log if not d["accepted"]]
        # Removing "deliver" must have been rejected (precondition revalidation).
        assert any(
            d["removed"]["action"] == "deliver" and "precondition" in d["reason"]
            or "signature" in d["reason"]
            for d in rejected
        )

    asyncio.run(scenario())


def test_reducer_cannot_delete_preconditions_for_repeated_close(spec):
    async def scenario():
        commands = [
            Command("r", a)
            for a in ["create", "submit", "approve", "deliver", "close", "close"]
        ]
        steps, trigger = await replay_commands(
            spec, commands, _factory(), id_prefix="t2-"
        )
        signature = discrepancy_signature(trigger)
        result = await reduce_sequence(
            spec, commands, trigger_seq=len(commands),
            target_signature=signature,
            transport_factory=_factory, id_prefix="t2-",
        )
        actions = [c.action for c in result.commands]
        assert actions == ["create", "submit", "approve", "deliver", "close", "close"]
        assert result.trigger_step["observation"]["accepted"] is True
        assert result.trigger_step["expected"]["accepted"] is False

    asyncio.run(scenario())


def test_reduced_sequence_does_not_fire_against_reference(spec):
    from mdbt.services.reference import create_app as ref_factory

    async def scenario():
        commands = [
            Command("r", a)
            for a in ["create", "submit", "approve", "submit"]
        ]
        # Against the reference service this sequence has no discrepancy at all.
        steps, trigger = await replay_commands(
            spec, commands,
            InProcessTransport(ref_factory()), id_prefix="t3-",
        )
        assert trigger is None
        assert steps[-1]["observation"]["status_code"] == 409
        assert steps[-1]["verdict"]["kind"] == "consistent"

    asyncio.run(scenario())
