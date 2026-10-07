"""Configurable finite-state-machine specification.

A model is a JSON-serialisable document::

    {
      "name": "order_lifecycle",
      "initial_state": "draft",
      "states": ["draft", "submitted", ...],
      "actions": ["create", "submit", ...],
      "transitions": [
        {"action": "submit", "from": "draft", "to": "submitted"}
      ],
      "terminal_states": ["closed", "cancelled"]
    }

The same action is allowed from several states; the *absence* of a transition
is the oracle definition of an illegal (state-conflict) operation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import InputError


@dataclass(frozen=True)
class Transition:
    action: str
    from_state: str
    to_state: str

    @classmethod
    def from_dict(cls, raw: object, index: int) -> "Transition":
        if not isinstance(raw, dict):
            raise InputError(
                f"transitions[{index}] must be an object", detail={"index": index}
            )
        missing = {"action", "from", "to"} - raw.keys()
        if missing:
            raise InputError(
                f"transitions[{index}] missing keys: {sorted(missing)}",
                detail={"index": index, "missing": sorted(missing)},
            )
        action, frm, to = raw["action"], raw["from"], raw["to"]
        for key, value in (("action", action), ("from", frm), ("to", to)):
            if not isinstance(value, str) or not value:
                raise InputError(
                    f"transitions[{index}].{key} must be a non-empty string",
                    detail={"index": index, "key": key},
                )
        return cls(action=action, from_state=frm, to_state=to)


@dataclass(frozen=True)
class MachineSpec:
    name: str
    initial_state: str
    states: frozenset[str]
    actions: frozenset[str]
    transitions: tuple[Transition, ...]
    terminal_states: frozenset[str] = field(default_factory=frozenset)

    def destinations(self) -> dict[tuple[str, str], str]:
        return {(t.from_state, t.action): t.to_state for t in self.transitions}

    def legal_actions(self, state: str) -> list[str]:
        return sorted(t.action for t in self.transitions if t.from_state == state)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "initial_state": self.initial_state,
            "states": sorted(self.states),
            "actions": sorted(self.actions),
            "terminal_states": sorted(self.terminal_states),
            "transitions": [
                {"action": t.action, "from": t.from_state, "to": t.to_state}
                for t in self.transitions
            ],
        }


def load_spec(raw: object) -> MachineSpec:
    """Parse and fully validate a model document.

    Every structural problem raises :class:`InputError` so callers never see a
    half-valid spec (this is where "boundary input silently wrong" is stopped).
    """
    if not isinstance(raw, dict):
        raise InputError("model document must be a JSON object", detail={"got": type(raw).__name__})

    def _string(key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise InputError(f"'{key}' must be a non-empty string", detail={"key": key})
        return value

    def _string_set(key: str, *, required: bool = True) -> frozenset[str]:
        if key not in raw:
            if required:
                raise InputError(f"'{key}' is required", detail={"key": key})
            return frozenset()
        value = raw[key]
        if not isinstance(value, list) or not value:
            raise InputError(
                f"'{key}' must be a non-empty array of strings", detail={"key": key}
            )
        if any(not isinstance(v, str) or not v for v in value):
            raise InputError(f"'{key}' contains a non-string or empty value", detail={"key": key})
        if len(set(value)) != len(value):
            raise InputError(f"'{key}' contains duplicates", detail={"key": key})
        return frozenset(value)

    name = _string("name")
    initial = _string("initial_state")
    states = _string_set("states")
    actions = _string_set("actions")
    terminals = _string_set("terminal_states", required=False)

    if initial not in states:
        raise InputError(
            "initial_state must be one of states",
            detail={"initial_state": initial, "states": sorted(states)},
        )
    if not terminals <= states:
        raise InputError(
            "terminal_states must be a subset of states",
            detail={"unknown": sorted(terminals - states)},
        )
    if initial in terminals:
        raise InputError(
            "initial_state cannot be terminal", detail={"initial_state": initial}
        )

    raw_transitions = raw.get("transitions")
    if not isinstance(raw_transitions, list) or not raw_transitions:
        raise InputError("'transitions' must be a non-empty array")
    transitions = tuple(
        Transition.from_dict(item, i) for i, item in enumerate(raw_transitions)
    )

    seen: set[tuple[str, str]] = set()
    for t in transitions:
        if t.action not in actions:
            raise InputError(
                f"transition uses unknown action {t.action!r}", detail={"action": t.action}
            )
        if t.from_state not in states:
            raise InputError(
                f"transition uses unknown from-state {t.from_state!r}",
                detail={"state": t.from_state},
            )
        if t.to_state not in states:
            raise InputError(
                f"transition targets unknown state {t.to_state!r}",
                detail={"state": t.to_state},
            )
        key = (t.from_state, t.action)
        if key in seen:
            raise InputError(
                "deterministic model cannot have two transitions for "
                f"(state={t.from_state!r}, action={t.action!r})",
                detail={"state": t.from_state, "action": t.action},
            )
        seen.add(key)

    target_states = {t.to_state for t in transitions}
    if "create" in actions:
        # Construction lands at the initial state without an explicit edge.
        target_states |= {initial}
    unreachable = states - {initial} - target_states
    if unreachable:
        raise InputError(
            "some states are unreachable from initial_state",
            detail={"unreachable": sorted(unreachable)},
        )

    return MachineSpec(
        name=name,
        initial_state=initial,
        states=states,
        actions=actions,
        transitions=transitions,
        terminal_states=terminals,
    )


def load_spec_file(path: str | Path) -> MachineSpec:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read model file: {exc}", detail={"path": str(path)}) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(
            f"model file is not valid JSON: {exc.msg}",
            detail={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    return load_spec(raw)
