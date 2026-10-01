"""In-memory service orchestration and response serialization."""

from __future__ import annotations

import threading
import uuid
from typing import Any

from app import config
from app.audit import check_layers, numeric_uncertainty
from app.core import (
    MutationEvidence,
    ProblemStore,
    add_candidates,
    candidate_relations,
    create_store,
    delete_candidate,
    initialize_problem,
)
from app.errors import NotFound, ValidationFailure
from app.models import (
    CandidateInput,
    CreateProblemRequest,
    EvidenceView,
    NumericPolicyView,
)
from app.parsing import build_policy, fraction_to_string

MAX_OBJECTIVES = 64


class ProblemRegistry:
    def __init__(self) -> None:
        self._stores: dict[str, ProblemStore] = {}
        self._lock = threading.RLock()

    def create(
        self, request: CreateProblemRequest, request_id: str
    ) -> tuple[ProblemStore, MutationEvidence]:
        with self._lock:
            problem_id = request.problem_id or f"problem-{uuid.uuid4().hex[:12]}"
            if problem_id in self._stores:
                raise ValidationFailure(
                    f"Problem id {problem_id!r} already exists.",
                    code="duplicate_problem_id",
                    stage="model-parsing",
                    status_code=409,
                    details={"problem_id": problem_id},
                )
            if request.objectives is None:
                raise ValidationFailure(
                    "The required 'objectives' list is missing.",
                    code="missing_objectives_specification",
                    stage="model-parsing",
                )
            if len(request.objectives) < 2:
                raise ValidationFailure(
                    "At least two fixed optimization objectives are required.",
                    code="too_few_objectives",
                    stage="model-parsing",
                )
            if len(request.objectives) > MAX_OBJECTIVES:
                raise ValidationFailure(
                    f"At most {MAX_OBJECTIVES} objectives are supported.",
                    code="too_many_objectives",
                    stage="model-parsing",
                    details={"received": len(request.objectives), "maximum": MAX_OBJECTIVES},
                )
            names = [
                objective.name or f"objective_{index}"
                for index, objective in enumerate(request.objectives)
            ]
            directions = [objective.direction for objective in request.objectives]
            policy = build_policy(
                request.numeric_settings.mode,
                request.numeric_settings.tolerance,
            )
            store = create_store(
                problem_id=problem_id,
                directions=directions,
                objective_names=names,
                policy=policy,
            )
            self._stores[problem_id] = store
            rows = [
                _candidate_row(candidate, index)
                for index, candidate in enumerate(request.candidates)
            ]
            evidence = initialize_problem(store, rows)
            if not request.candidates:
                evidence = MutationEvidence(
                    "problem.initialize",
                    "empty-problem",
                    ["Created an empty fixed-direction problem."],
                    0,
                    0,
                    0,
                    [],
                    [],
                )
            _assert_checked(store, evidence, request_id)
            return store, evidence

    def get(self, problem_id: str) -> ProblemStore:
        with self._lock:
            try:
                return self._stores[problem_id]
            except KeyError:
                raise NotFound(
                    f"Problem id {problem_id!r} does not exist.",
                    code="problem_not_found",
                    stage="state-read",
                    details={"problem_id": problem_id},
                ) from None

    def add(
        self,
        problem_id: str,
        candidates: list[CandidateInput],
        request_id: str,
    ) -> tuple[ProblemStore, MutationEvidence]:
        with self._lock:
            store = self.get(problem_id)
            if not candidates:
                raise ValidationFailure(
                    "At least one candidate is required.",
                    code="empty_candidate_batch",
                    stage="model-parsing",
                )
            rows = [_candidate_row(candidate, index) for index, candidate in enumerate(candidates)]
            evidence = add_candidates(store, rows)
            _assert_checked(store, evidence, request_id)
            return store, evidence

    def delete(
        self,
        problem_id: str,
        candidate_id: str,
        request_id: str,
    ) -> tuple[ProblemStore, MutationEvidence]:
        with self._lock:
            store = self.get(problem_id)
            evidence = delete_candidate(store, candidate_id)
            _assert_checked(store, evidence, request_id)
            return store, evidence

    def lock(self) -> threading.RLock:
        return self._lock


registry = ProblemRegistry()


def _candidate_row(
    candidate: CandidateInput, index: int
) -> tuple[str, list[Any] | None, dict[str, Any] | None]:
    candidate_id = candidate.id or f"candidate-{index + 1:04d}"
    return candidate_id, candidate.objectives, candidate.payload


def _assert_checked(
    store: ProblemStore, evidence: MutationEvidence, request_id: str
) -> None:
    failures = [check for check in check_layers(store) if not check.passed]
    if failures:
        details = "; ".join(f"{check.name}: {check.details}" for check in failures)
        raise RuntimeError(f"Post-operation verification failed for request {request_id}: {details}")


def policy_view(store: ProblemStore) -> NumericPolicyView:
    return NumericPolicyView(
        mode=store.policy.mode,
        tolerance=(
            None
            if store.policy.tolerance is None
            else fraction_to_string(store.policy.tolerance)
        ),
        exact_by_default=store.policy.is_exact,
    )


def state_view(store: ProblemStore) -> dict[str, Any]:
    layers = [
        {"rank": index, "candidate_ids": candidate_ids}
        for index, candidate_ids in enumerate(store.layers(), start=1)
    ]
    candidates = []
    for candidate in store.ordered_points():
        _, dominates_ids, dominated_by_ids, _ = candidate_relations(
            store, candidate.id
        )
        candidates.append(
            {
                "id": candidate.id,
                "objectives": list(candidate.declared),
                "payload": candidate.payload,
                "rank": candidate.rank,
                "duplicate_group": candidate.group_id,
                "dominates": dominates_ids,
                "dominated_by": dominated_by_ids,
            }
        )
    return {
        "problem_id": store.problem_id,
        "objective_count": store.objective_count,
        "objective_names": store.objective_names,
        "directions": list(store.directions),
        "numeric_policy": policy_view(store).model_dump(),
        "candidate_count": len(store.points),
        "layers": layers,
        "candidates": candidates,
    }


def evidence_view(store: ProblemStore, evidence: MutationEvidence) -> EvidenceView:
    checks = check_layers(store)
    duplicate_groups = [
        group.member_ids
        for group in store.ordered_groups()
        if len(group.member_ids) > 1
    ]
    return EvidenceView(
        algorithm_version=config.ALGORITHM_VERSION,
        operation=evidence.operation,
        processing_stage=evidence.stage,
        steps=list(evidence.steps),
        pair_comparisons=evidence.pair_comparisons,
        edges_added=evidence.edges_added,
        edges_removed=evidence.edges_removed,
        affected_candidate_ids=list(evidence.affected_candidate_ids),
        duplicate_groups=duplicate_groups,
        checks=[check.as_dict() for check in checks],
        uncertainty=numeric_uncertainty(store),
    )


def response(
    *,
    request_id: str,
    stage: str,
    data: dict[str, Any],
    store: ProblemStore | None = None,
    evidence: MutationEvidence | None = None,
) -> dict[str, Any]:
    uncertainty = numeric_uncertainty(store) if store is not None else []
    return {
        "request_id": request_id,
        "version": config.SERVICE_VERSION,
        "stage": stage,
        "data": data,
        "evidence": (
            evidence_view(store, evidence).model_dump()
            if store is not None and evidence is not None
            else None
        ),
        "errors": [],
        "uncertainty": uncertainty,
    }
