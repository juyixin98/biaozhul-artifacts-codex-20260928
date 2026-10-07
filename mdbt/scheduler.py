"""Seeded scheduler.

Responsibilities:

* Track which resources exist and, locally, the scheduler-side view of each
  model state (used only to pick *legal* vs deliberately *illegal* actions;
  the oracle in the executor remains the authority on expectation).
* Record every scheduling decision: seed, RNG state before/after, the
  candidate set and the chosen command.  This log is what makes a run
  replayable.

Illegal actions are generated on purpose, in three flavours:

* ``unknown_action``     - action absent from the model (input_error)
* ``illegal_transition`` - known action with no edge from current state
* ``missing_resource``   - action against a never-created id
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .contracts import Command
from .errors import InputError
from .spec import MachineSpec


UNKNOWN_ACTION_POOL = ("freeze", "archive", "teleport", "reindex")


@dataclass
class SchedulingDecision:
    seq: int
    rng_state_before: list
    mode: str
    candidates: list[str]
    chosen: Command
    rationale: str
    rng_state_after: list
    resource_exists: bool
    model_state_hint: str

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "rng_state_before": self.rng_state_before,
            "mode": self.mode,
            "candidates": self.candidates,
            "chosen": self.chosen.to_dict(),
            "rationale": self.rationale,
            "rng_state_after": self.rng_state_after,
            "resource_exists": self.resource_exists,
            "model_state_hint": self.model_state_hint,
        }


@dataclass
class Scheduler:
    spec: MachineSpec
    seed: int
    illegal_fraction: float = 0.45
    max_resources: int = 16
    rng: random.Random = field(init=False)
    decisions: list[SchedulingDecision] = field(init=False, default_factory=list)
    existing_ids: list[str] = field(init=False, default_factory=list)
    states: dict[str, str] = field(init=False, default_factory=dict)
    _seq: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not 0.0 <= self.illegal_fraction <= 1.0:
            raise InputError(
                "illegal_fraction must be within [0, 1]",
                detail={"illegal_fraction": self.illegal_fraction},
            )
        self.rng = random.Random(self.seed)

    @property
    def rng_state(self) -> list:
        return list(self.rng.getstate()[1])

    def _new_id(self) -> str:
        idx = len(self.existing_ids)
        rid = f"order-{idx:03d}"
        self.existing_ids.append(rid)
        self.states[rid] = self.spec.initial_state
        return rid

    def _pick_legal_action(self, state: str) -> tuple[str, list[str]]:
        candidates = self.spec.legal_actions(state)
        if not candidates:
            raise InputError(
                "no legal actions from a non-terminal scheduler state",
                detail={"state": state},
            )
        return self.rng.choice(candidates), candidates

    def _pick_illegal_action(self, state: str) -> tuple[str, list[str], str]:
        edge_actions = {t.action for t in self.spec.transitions}
        illegal_known = sorted(edge_actions - set(self.spec.legal_actions(state)))
        unknown = [a for a in UNKNOWN_ACTION_POOL if a not in self.spec.actions]
        modes: list[str] = []
        if illegal_known:
            modes.append("illegal_transition")
        if unknown:
            modes.append("unknown_action")
        if self.existing_ids:
            modes.append("missing_resource")
        mode = self.rng.choice(modes)
        if mode == "illegal_transition":
            return self.rng.choice(illegal_known), illegal_known, mode
        if mode == "unknown_action":
            return self.rng.choice(unknown), unknown, mode
        ghost = f"ghost-{self.rng.randrange(100000):05d}"
        non_create = sorted(self.spec.actions - {"create"})
        return (
            Command(
                resource_id=ghost, action=self.rng.choice(non_create)
            ).action,
            [ghost],
            mode,
        )

    def next_command(self) -> tuple[Command, SchedulingDecision]:
        """Choose one command. Scheduler state advances only here."""
        self._seq += 1
        state_before = self.rng_state

        # Need a live resource: create one first.
        if not self.existing_ids:
            rid = self._new_id()
            create = sorted(a for a in self.spec.actions if a == "create") or [
                self.spec.legal_actions(self.spec.initial_state)[0]
            ]
            chosen = Command(resource_id=rid, action=create[0])
            decision = SchedulingDecision(
                seq=self._seq,
                rng_state_before=state_before,
                mode="bootstrap_create",
                candidates=[chosen.action],
                chosen=chosen,
                rationale="no resources tracked yet; create the first one",
                rng_state_after=self.rng_state,
                resource_exists=False,
                model_state_hint=self.spec.initial_state,
            )
            self.decisions.append(decision)
            return chosen, decision

        rid = self.rng.choice(self.existing_ids)
        current = self.states[rid]

        # Terminal resource: scheduler retires it and bootstraps a successor.
        if current in self.spec.terminal_states:
            self.states.pop(rid, None)
            self.existing_ids.remove(rid)
            if len(self.existing_ids) < self.max_resources and self.rng.random() < 0.8:
                new_id = self._new_id()
                create = "create" if "create" in self.spec.actions else (
                    self.spec.legal_actions(self.spec.initial_state)[0]
                )
                chosen = Command(resource_id=new_id, action=create)
                mode = "terminal_replace"
                rationale = f"resource {rid} terminal; create successor {new_id}"
                exists = False
                hint = self.spec.initial_state
            else:
                if not self.existing_ids:
                    new_id = self._new_id()
                    create = "create" if "create" in self.spec.actions else (
                        self.spec.legal_actions(self.spec.initial_state)[0]
                    )
                    chosen = Command(resource_id=new_id, action=create)
                    mode = "terminal_replace"
                    rationale = f"resource {rid} terminal; create successor {new_id}"
                    exists = False
                    hint = self.spec.initial_state
                else:
                    rid2 = self.rng.choice(self.existing_ids)
                    cur2 = self.states[rid2]
                    act, cands = self._pick_legal_action(cur2)
                    chosen = Command(resource_id=rid2, action=act)
                    mode = "legal"
                    rationale = f"legal action from {cur2!r} after retiring terminal"
                    exists = True
                    hint = cur2
            decision = SchedulingDecision(
                seq=self._seq,
                rng_state_before=state_before,
                mode=mode,
                candidates=[chosen.action],
                chosen=chosen,
                rationale=rationale,
                rng_state_after=self.rng_state,
                resource_exists=exists,
                model_state_hint=hint,
            )
            self.decisions.append(decision)
            return chosen, decision

        use_illegal = self.rng.random() < self.illegal_fraction
        if use_illegal:
            action, cands, mode = self._pick_illegal_action(current)
            if mode == "missing_resource":
                ghost = cands[0]
                chosen = Command(resource_id=ghost, action=action)
                rationale = f"deliberately target unknown resource {ghost}"
                exists = False
            else:
                chosen = Command(resource_id=rid, action=action)
                rationale = f"deliberately illegal {mode} from {current!r}"
                exists = True
        else:
            action, cands = self._pick_legal_action(current)
            chosen = Command(resource_id=rid, action=action)
            mode = "legal"
            rationale = f"legal action from {current!r}"
            exists = True

        decision = SchedulingDecision(
            seq=self._seq,
            rng_state_before=state_before,
            mode=mode,
            candidates=cands,
            chosen=chosen,
            rationale=rationale,
            rng_state_after=self.rng_state,
            resource_exists=exists,
            model_state_hint=current,
        )
        self.decisions.append(decision)
        return chosen, decision

    def observe_result(self, command: Command, accepted: bool, new_state: str | None) -> None:
        """Update the scheduler's local hint map from an observation."""
        if accepted and new_state is not None and command.resource_id in self.states:
            self.states[command.resource_id] = new_state
