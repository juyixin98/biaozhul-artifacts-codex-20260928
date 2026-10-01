"""Thread-safe in-memory dataset service."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from app.errors import conflict_error, not_found_error
from app.evidence import build_evidence, numerical_diagnostics
from app.kernel import (
    CandidateInput,
    DeletionReport,
    DuplicateCandidateId,
    IncrementalSorter,
    InsertionReport,
    Objective,
    UnknownCandidate,
)


@dataclass
class Dataset:
    dataset_id: str
    sorter: IncrementalSorter
    tolerance: float


class DatasetStore:
    def __init__(self) -> None:
        self._datasets: dict[str, Dataset] = {}
        self._lock = threading.RLock()

    def create(
        self,
        objectives: tuple[Objective, ...],
        candidates: tuple[CandidateInput, ...],
        tolerance: float,
        dataset_id: str | None = None,
    ) -> Dataset:
        if dataset_id is None:
            dataset_id = uuid.uuid4().hex
        with self._lock:
            if dataset_id in self._datasets:
                raise conflict_error(
                    "DUPLICATE_DATASET_ID",
                    f"dataset already exists: {dataset_id}",
                    {"dataset_id": dataset_id},
                )
            sorter = IncrementalSorter(objectives)
            sorter.add_candidates(candidates)
            dataset = Dataset(dataset_id=dataset_id, sorter=sorter, tolerance=tolerance)
            self._datasets[dataset_id] = dataset
            return dataset

    def get(self, dataset_id: str) -> Dataset:
        with self._lock:
            try:
                return self._datasets[dataset_id]
            except KeyError:
                raise not_found_error(
                    "DATASET_NOT_FOUND",
                    f"dataset not found: {dataset_id}",
                    {"dataset_id": dataset_id},
                ) from None

    def delete(self, dataset_id: str) -> None:
        with self._lock:
            if dataset_id not in self._datasets:
                raise not_found_error(
                    "DATASET_NOT_FOUND",
                    f"dataset not found: {dataset_id}",
                    {"dataset_id": dataset_id},
                )
            del self._datasets[dataset_id]

    def insert(
        self,
        dataset_id: str,
        candidates: tuple[CandidateInput, ...],
    ) -> tuple[Dataset, InsertionReport]:
        with self._lock:
            dataset = self.get(dataset_id)
            try:
                report = dataset.sorter.add_candidates(candidates)
            except DuplicateCandidateId as exc:
                raise conflict_error(
                    "DUPLICATE_CANDIDATE_ID",
                    str(exc),
                    {"candidate_id": exc.candidate_id},
                ) from None
            return dataset, report

    def remove_candidate(
        self,
        dataset_id: str,
        candidate_id: str,
    ) -> tuple[Dataset, DeletionReport]:
        with self._lock:
            dataset = self.get(dataset_id)
            try:
                report = dataset.sorter.remove_candidate(candidate_id)
            except UnknownCandidate:
                raise not_found_error(
                    "CANDIDATE_NOT_FOUND",
                    f"unknown candidate id: {candidate_id}",
                    {"dataset_id": dataset_id, "candidate_id": candidate_id},
                ) from None
            return dataset, report


def build_dataset_payload(dataset: Dataset, processing_steps: list[str]) -> dict[str, object]:
    sorter = dataset.sorter
    evidence = build_evidence(sorter)
    diagnostics = numerical_diagnostics(sorter, dataset.tolerance)

    return {
        "dataset_id": dataset.dataset_id,
        "api_version": "v1",
        "algorithm_version": "exact-domination-graph-v1",
        "kernel_version": sorter.version,
        "objectives": [
            {"name": objective.name, "direction": objective.direction}
            for objective in sorter.objectives
        ],
        "tolerance": dataset.tolerance,
        "candidates": [
            {
                "id": candidate.id,
                "values": {
                    name: candidate.vector[index]
                    for index, name in enumerate(sorter.objective_names)
                },
                "rank": sorter.ranks[candidate.id],
            }
            for candidate in sorter.snapshot_candidates()
        ],
        "layers": sorter.layers(),
        "evidence": {
            "comparisons_performed": evidence.comparisons_performed,
            "pairwise_dominance_edges": evidence.pairwise_dominance_edges,
            "duplicate_groups": [list(group) for group in evidence.duplicate_groups],
            "independent_reference_layers": [
                list(layer) for layer in evidence.independent_reference_layers
            ],
            "independent_reference_matched": evidence.independent_reference_matched,
            "layer_non_dominated": evidence.layer_non_dominated,
            "transitivity_preserved": evidence.transitivity_preserved,
            "ordering_predicate": evidence.ordering_predicate,
            "tolerance_applied_to_ordering": evidence.tolerance_applied_to_ordering,
        },
        "diagnostics": {
            "tolerance": diagnostics.tolerance,
            "tolerance_role": diagnostics.tolerance_role,
            "constant_objectives": list(diagnostics.constant_objectives),
            "near_equal_pairs": list(diagnostics.near_equal_pairs),
            "duplicate_vector_count": diagnostics.duplicate_vector_count,
            "uncertainties": list(diagnostics.uncertainties),
        },
        "processing": processing_steps,
    }


def stateless_dataset(
    objectives: tuple[Objective, ...],
    candidates: tuple[CandidateInput, ...],
    tolerance: float,
) -> Dataset:
    sorter = IncrementalSorter(objectives)
    sorter.add_candidates(candidates)
    return Dataset(
        dataset_id="stateless-sort",
        sorter=sorter,
        tolerance=tolerance,
    )
