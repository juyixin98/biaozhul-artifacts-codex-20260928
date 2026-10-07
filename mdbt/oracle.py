"""Pure model oracle (per-resource state tracking).

This module never imports an HTTP client and never calls the system under
test.  The execution protocol is deliberately split into three functions:

1. :meth:`FSMModel.expected` computes the predicted outcome **without
   mutating** any resource state.
2. The executor observes the real SUT response.
3. :meth:`FSMModel.apply_observation` advances *that resource's* state based
   solely on what was observed.

State is tracked per ``resource_id``: interleaving commands on different
resources must never let one resource's history corrupt another's prediction.
On a mismatch the affected resource is re-synchronised to the state the SUT
reported and the event is tagged ``resynced``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ErrorCategory
from .spec import MachineSpec


@dataclass(frozen=True)
class ExpectedOutcome:
    accepted: bool
    to_state: str | None
    category: ErrorCategory | None
    reason: str

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "to_state": self.to_state,
            "category": None if self.category is None else self.category.value,
            "reason": self.reason,
        }


@dataclass
class FSMModel:
    spec: MachineSpec
    states: dict[str, str] = field(default_factory=dict)
    resync_count: int = 0

    @classmethod
    def fresh(cls, spec: MachineSpec) -> "FSMModel":
        return cls(spec=spec, states={})

    def state_of(self, resource_id: str) -> str:
        return self.states.get(resource_id, self.spec.initial_state)

    def expected(
        self, resource_id: str, action: str, *, resource_exists: bool
    ) -> ExpectedOutcome:
        """Pure prediction. Never mutates model state."""
        if action == "create":
            if resource_exists:
                return ExpectedOutcome(
                    accepted=False,
                    to_state=self.spec.initial_state,
                    category=ErrorCategory.STATE_CONFLICT,
                    reason="create on an existing resource is a state conflict",
                )
            return ExpectedOutcome(
                accepted=True,
                to_state=self.spec.initial_state,
                category=None,
                reason=f"create resource at initial state {self.spec.initial_state!r}",
            )

        if not resource_exists:
            return ExpectedOutcome(
                accepted=False,
                to_state=None,
                category=ErrorCategory.INPUT_ERROR,
                reason="resource does not exist",
            )
        if action not in self.spec.actions:
            return ExpectedOutcome(
                accepted=False,
                to_state=None,
                category=ErrorCategory.INPUT_ERROR,
                reason=f"unknown action {action!r}",
            )

        current = self.state_of(resource_id)
        to_state = self.spec.destinations().get((current, action))
        if to_state is None:
            return ExpectedOutcome(
                accepted=False,
                to_state=None,
                category=ErrorCategory.STATE_CONFLICT,
                reason=f"action {action!r} is not legal in state {current!r}",
            )
        return ExpectedOutcome(
            accepted=True,
            to_state=to_state,
            category=None,
            reason=f"{current!r} --{action}--> {to_state!r}",
        )

    def apply_observation(
        self,
        resource_id: str,
        action: str,
        *,
        observed_accepted: bool,
        observed_state: str | None,
        resource_exists_after: bool,
    ) -> tuple[str, bool]:
        """Advance one resource's state from the observation only."""
        if not resource_exists_after:
            return self.state_of(resource_id), False
        if observed_state is None or observed_state not in self.spec.states:
            return self.state_of(resource_id), False

        current = self.state_of(resource_id)
        if observed_accepted:
            self.states[resource_id] = observed_state
            return observed_state, False

        # Rejected: only resync when the SUT reveals the model has drifted.
        if observed_state != current:
            self.states[resource_id] = observed_state
            self.resync_count += 1
            return observed_state, True
        return current, False
