"""Predictive state machine, separated from system-under-test calls."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import ObservationKind, StateMachineModel, TransitionSpec


class ModelStateError(RuntimeError):
    """A generated operation is not legal in the current model state."""


@dataclass(slots=True)
class ModelState:
    """Mutable model state.

    ``apply_observed`` is intentionally the only state-mutating operation.
    The HTTP adapter must call the SUT before calling it; a local assertion
    protects that ordering during development.
    """

    model: StateMachineModel
    state: str
    observed: bool = True

    @classmethod
    def initial(cls, model: StateMachineModel) -> "ModelState":
        return cls(model=model, state=model.initial_state, observed=True)

    def transition(self, name: str) -> TransitionSpec:
        for item in self.model.transitions:
            if item.name == name:
                transition = item
                break
        else:
            raise ModelStateError(f"unknown transition: {name}")
        if self.state not in transition.allowed_states:
            raise ModelStateError(
                f"{name} is not enabled in state {self.state}"
            )
        return transition

    def mark_call_dispatched(self) -> None:
        self.observed = False

    def apply_observed(self, transition_name: str) -> str:
        if self.observed:
            raise ModelStateError("cannot update model before a fresh SUT call")
        transition = self.transition(transition_name)
        assert transition.next_state is not None
        self.state = transition.next_state
        self.observed = True
        return self.state

    def reset_to_initial(self) -> None:
        self.state = self.model.initial_state
        self.observed = True

    def is_success(self, name: str) -> bool:
        return self.transition(name).observation is ObservationKind.SUCCESS
