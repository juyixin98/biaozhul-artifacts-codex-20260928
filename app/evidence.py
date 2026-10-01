"""Independent evidence and numerical diagnostics.

The reference ranking in this module deliberately does not call the
incremental kernel's rank result.  It rebuilds an all-pairs domination matrix
and repeatedly removes the current non-dominated front.  The kernel and this
reference can therefore disagree if an incremental update is incorrect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from app.kernel import CandidateInput, Direction, IncrementalSorter


@dataclass(frozen=True)
class Evidence:
    comparisons_performed: int
    pairwise_dominance_edges: int
    duplicate_groups: tuple[tuple[str, ...], ...]
    independent_reference_layers: tuple[tuple[str, ...], ...]
    independent_reference_matched: bool
    layer_non_dominated: bool
    transitivity_preserved: bool
    ordering_predicate: str
    tolerance_applied_to_ordering: bool


@dataclass(frozen=True)
class DiagnosticReport:
    tolerance: float
    tolerance_role: str
    constant_objectives: tuple[str, ...]
    near_equal_pairs: tuple[dict[str, object], ...]
    duplicate_vector_count: int
    uncertainties: tuple[dict[str, object], ...]


def _exact_dominates(
    left: Sequence[float],
    right: Sequence[float],
    directions: Sequence[Direction],
) -> bool:
    strictly_better = False
    for left_value, right_value, direction in zip(left, right, directions):
        if direction == "min":
            if left_value > right_value:
                return False
            if left_value < right_value:
                strictly_better = True
        else:
            if left_value < right_value:
                return False
            if left_value > right_value:
                strictly_better = True
    return strictly_better


def reference_layers(
    candidates: Sequence[CandidateInput],
    directions: Sequence[Direction],
) -> tuple[tuple[str, ...], ...]:
    remaining = set(candidate.id for candidate in candidates)
    by_id = {candidate.id: candidate for candidate in candidates}
    layers: list[tuple[str, ...]] = []

    while remaining:
        front: list[str] = []
        for candidate_id in sorted(remaining):
            is_dominated = any(
                other_id != candidate_id
                and _exact_dominates(
                    by_id[other_id].vector,
                    by_id[candidate_id].vector,
                    directions,
                )
                for other_id in remaining
            )
            if not is_dominated:
                front.append(candidate_id)
        layers.append(tuple(front))
        remaining.difference_update(front)

    return tuple(layers)


def _layers_to_rank(layers: Sequence[Sequence[str]]) -> dict[str, int]:
    return {
        candidate_id: rank
        for rank, layer in enumerate(layers)
        for candidate_id in layer
    }


def _check_layer_non_domination(
    sorter: IncrementalSorter,
    layers: Sequence[Sequence[str]],
) -> bool:
    for layer in layers:
        for index, left_id in enumerate(layer):
            for right_id in layer[index + 1 :]:
                left = sorter.candidates[left_id]
                right = sorter.candidates[right_id]
                if _exact_dominates(left.vector, right.vector, sorter.directions):
                    return False
                if _exact_dominates(right.vector, left.vector, sorter.directions):
                    return False
    return True


def _check_graph_transitivity(sorter: IncrementalSorter) -> bool:
    ids = sorted(sorter.candidates)
    if not ids:
        return True
    index = {candidate_id: i for i, candidate_id in enumerate(ids)}
    rows: list[int] = []
    cols: list[int] = []
    for parent_id, children in sorter.children.items():
        for child_id in children:
            rows.append(index[parent_id])
            cols.append(index[child_id])

    data = np.ones(len(rows), dtype=np.int8)
    size = len(ids)
    graph = csr_matrix((data, (rows, cols)), shape=(size, size))
    reach = shortest_path(csgraph=graph, directed=True, unweighted=True)

    for source, source_id in enumerate(ids):
        for target, target_id in enumerate(ids):
            if source_id == target_id:
                if np.isfinite(reach[source, source]) and reach[source, source] > 0:
                    return False
                continue
            if not np.isfinite(reach[source, target]):
                continue
            if sorter.ranks[source_id] >= sorter.ranks[target_id]:
                return False
    return True


def build_evidence(sorter: IncrementalSorter) -> Evidence:
    candidates = sorter.snapshot_candidates()
    reference = reference_layers(candidates, sorter.directions)
    reference_ranks = _layers_to_rank(reference)
    independent_match = sorter.ranks == reference_ranks
    layers = sorter.layers()

    return Evidence(
        comparisons_performed=len(candidates) * max(0, len(candidates) - 1) // 2,
            pairwise_dominance_edges=sorter.edge_count(),
        duplicate_groups=tuple(tuple(group) for group in sorter.duplicate_groups()),
        independent_reference_layers=reference,
        independent_reference_matched=independent_match,
        layer_non_dominated=_check_layer_non_domination(sorter, layers),
        transitivity_preserved=_check_graph_transitivity(sorter),
        ordering_predicate="exact no-worse in every fixed direction and strictly better in one",
        tolerance_applied_to_ordering=False,
    )


def numerical_diagnostics(
    sorter: IncrementalSorter,
    tolerance: float,
) -> DiagnosticReport:
    candidates = sorter.snapshot_candidates()
    names = sorter.objective_names
    matrix = np.array(
        [candidate.vector for candidate in candidates],
        dtype=np.float64,
        order="C",
    )

    if candidates:
        constant_objectives = tuple(
            name
            for name, column in zip(names, matrix.T)
            if np.allclose(column, column[0], rtol=0.0, atol=0.0)
        )
    else:
        constant_objectives = ()

    near_pairs: list[dict[str, object]] = []
    uncertainties: list[dict[str, object]] = []

    if candidates and tolerance > 0.0:
        # Pairwise Euclidean values are normalized per-dimension for distance:
        # in max dimensions the direction is reversed, while distance itself is
        # unchanged.  This remains a report only and never changes ordering.
        normalized = np.array(
            [
                [
                    -value if direction == "max" else value
                    for value, direction in zip(candidate.vector, sorter.directions)
                ]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        for left_index in range(len(candidates)):
            for right_index in range(left_index + 1, len(candidates)):
                difference = np.abs(
                    normalized[left_index] - normalized[right_index]
                )
                distance = float(np.max(difference))
                if distance <= tolerance:
                    pair = {
                        "left_id": candidates[left_index].id,
                        "right_id": candidates[right_index].id,
                        "chebyshev_distance": distance,
                    }
                    near_pairs.append(pair)
                    if distance > 0.0:
                        uncertainties.append(
                            {
                                "code": "NEAR_EQUAL_WITHIN_TOLERANCE",
                                "severity": "uncertain",
                                "message": (
                                    "The pair is close to the tolerance; exact ordering "
                                    "was retained to preserve transitivity."
                                ),
                                "pair": pair,
                            }
                        )

    unique_vectors = {candidate.vector for candidate in candidates}
    duplicate_vector_count = len(candidates) - len(unique_vectors)

    return DiagnosticReport(
        tolerance=tolerance,
        tolerance_role=(
            "diagnostic-only; the exact declared numeric predicate is used for all ranks"
        ),
        constant_objectives=constant_objectives,
        near_equal_pairs=tuple(near_pairs),
        duplicate_vector_count=duplicate_vector_count,
        uncertainties=tuple(uncertainties),
    )
