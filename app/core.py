"""Finite-set non-dominated sorting kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable

import numpy as np
from scipy.sparse import csc_matrix, find

from app.errors import Conflict, ValidationFailure
from app.config import MAX_OBJECTIVES
from app.parsing import Direction, NumericPolicy, direction_sign, parse_fraction


@dataclass(slots=True)
class CandidateRecord:
    id: str
    declared: tuple[Any, ...]
    exact_declared: tuple[Fraction, ...]
    canonical: tuple[Fraction, ...]
    payload: dict[str, Any] | None
    group_id: str
    rank: int


@dataclass(slots=True)
class ObjectiveGroup:
    id: str
    canonical: tuple[Fraction, ...]
    member_ids: list[str]
    rank: int = 1


@dataclass(slots=True)
class MutationEvidence:
    operation: str
    stage: str
    steps: list[str]
    pair_comparisons: int
    edges_added: int
    edges_removed: int
    affected_candidate_ids: list[str]
    inserted_candidate_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProblemStore:
    problem_id: str
    objective_count: int
    objective_names: list[str]
    directions: list[Direction]
    signs: tuple[int, ...]
    policy: NumericPolicy
    points: dict[str, CandidateRecord] = field(default_factory=dict)
    groups: dict[str, ObjectiveGroup] = field(default_factory=dict)
    group_order: list[str] = field(default_factory=list)
    group_by_vector: dict[tuple[Fraction, ...], str] = field(default_factory=dict)
    group_down: dict[str, set[str]] = field(default_factory=dict)
    group_up: dict[str, set[str]] = field(default_factory=dict)
    _group_sequence: int = 0

    def ordered_groups(self) -> list[ObjectiveGroup]:
        return [self.groups[group_id] for group_id in self.group_order]

    def ordered_points(self) -> list[CandidateRecord]:
        return [self.points[candidate_id] for candidate_id in self.points]

    def _new_group_id(self) -> str:
        self._group_sequence += 1
        return f"group-{self._group_sequence:04d}"

    def layers(self) -> list[list[str]]:
        if not self.group_order:
            return []
        maximum_rank = max(group.rank for group in self.ordered_groups())
        result: list[list[str]] = [[] for _ in range(maximum_rank)]
        for candidate in self.ordered_points():
            result[candidate.rank - 1].append(candidate.id)
        return result


def dominates(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> bool:
    """Strict dominance after all objectives are transformed to maximize."""
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def _strict_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    all_not_worse = np.all(left[:, None, :] >= right[None, :, :], axis=2)
    any_better = np.any(left[:, None, :] > right[None, :, :], axis=2)
    return np.asarray(all_not_worse & any_better, dtype=bool)


def _rerank_with_scipy(store: ProblemStore) -> None:
    """Compute ranks via Kahn's longest path over the strict dominance DAG."""
    if not store.group_order:
        return
    # Build a SciPy sparse edge matrix as an independent structural representation.
    index = {group_id: i for i, group_id in enumerate(store.group_order)}
    rows: list[int] = []
    columns: list[int] = []
    for up_id, down_ids in store.group_down.items():
        for down_id in down_ids:
            rows.append(index[up_id])
            columns.append(index[down_id])
    edge_matrix = csc_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)),
        shape=(len(store.group_order), len(store.group_order)),
    )
    indegree = np.asarray(edge_matrix.sum(axis=0)).reshape(-1)
    if edge_matrix.nnz != len(rows):
        raise Conflict(
            "Duplicate edges detected in the sparse dominance graph.",
            code="duplicate_dominance_edge",
            stage="initial-rerank",
        )
    queue = [
        store.group_order[i]
        for i, degree in enumerate(indegree)
        if int(degree) == 0
    ]
    ranks = {group_id: 1 for group_id in store.group_order}
    processed = 0
    edge_rows, edge_columns, _ = find(edge_matrix)
    outgoing: dict[str, list[str]] = {group_id: [] for group_id in store.group_order}
    for row, column in zip(edge_rows, edge_columns, strict=True):
        outgoing[store.group_order[int(row)]].append(store.group_order[int(column)])
    while queue:
        up_id = queue.pop(0)
        processed += 1
        for down_id in outgoing[up_id]:
            ranks[down_id] = max(ranks[down_id], ranks[up_id] + 1)
            indegree[index[down_id]] -= 1
            if int(indegree[index[down_id]]) == 0:
                queue.append(down_id)
    if processed != len(store.group_order):
        raise Conflict(
            "Dominance graph contains a cycle, which is impossible for strict "
            "numeric dominance and indicates kernel corruption.",
            code="dominance_cycle",
            stage="initial-rerank",
        )
    for group_id, rank in ranks.items():
        group = store.groups[group_id]
        group.rank = int(rank)
        for candidate_id in group.member_ids:
            store.points[candidate_id].rank = int(rank)


def _rerank_affected(store: ProblemStore, affected: set[str]) -> None:
    """Recompute longest-path ranks for affected groups in their local DAG."""
    if not affected:
        return
    rank = {
        group_id: 1
        + max(
            (
                store.groups[up_id].rank
                for up_id in store.group_up[group_id]
                if up_id not in affected
            ),
            default=0,
        )
        for group_id in affected
    }
    indegree = {
        group_id: len([up for up in store.group_up[group_id] if up in affected])
        for group_id in affected
    }
    queue = [group_id for group_id, degree in indegree.items() if degree == 0]
    processed: list[str] = []
    while queue:
        up_id = queue.pop(0)
        processed.append(up_id)
        for down_id in sorted(store.group_down[up_id]):
            if down_id not in affected:
                continue
            rank[down_id] = max(rank[down_id], rank[up_id] + 1)
            indegree[down_id] -= 1
            if indegree[down_id] == 0:
                queue.append(down_id)
    if len(processed) != len(affected):
        raise Conflict(
            "Dominance graph contains a cycle, which is impossible for strict "
            "numeric dominance and indicates kernel corruption.",
            code="dominance_cycle",
            stage="incremental-rerank",
        )
    for group_id, new_rank in rank.items():
        group = store.groups[group_id]
        group.rank = new_rank
        for candidate_id in group.member_ids:
            store.points[candidate_id].rank = new_rank


def _descendants(store: ProblemStore, roots: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        group_id = stack.pop()
        if group_id in seen:
            continue
        seen.add(group_id)
        stack.extend(store.group_down[group_id])
    return seen


def create_store(
    *,
    problem_id: str,
    directions: list[Direction],
    objective_names: list[str] | None = None,
    policy: NumericPolicy,
) -> ProblemStore:
    if len(directions) < 2:
        raise ValidationFailure(
            "At least two objectives are required for multi-objective sorting.",
            code="too_few_objectives",
            stage="model-parsing",
            details={"objective_count": len(directions)},
        )
    if len(directions) > MAX_OBJECTIVES:
        raise ValidationFailure(
            f"At most {MAX_OBJECTIVES} objectives are supported.",
            code="too_many_objectives",
            stage="model-parsing",
            details={"received": len(directions), "maximum": MAX_OBJECTIVES},
        )
    names = objective_names or [f"objective_{i}" for i in range(len(directions))]
    if len(names) != len(directions):
        raise ValidationFailure(
            "Objective name count must match direction count.",
            code="objective_name_count_mismatch",
        )
    return ProblemStore(
        problem_id=problem_id,
        objective_count=len(directions),
        objective_names=names,
        directions=list(directions),
        signs=tuple(direction_sign(direction) for direction in directions),
        policy=policy,
    )


def _parse_candidate(
    store: ProblemStore,
    candidate_id: str,
    objectives: Any,
    payload: dict[str, Any] | None,
) -> CandidateRecord:
    if objectives is None:
        raise ValidationFailure(
            f"Candidate {candidate_id!r} is missing the required 'objectives' field.",
            code="missing_objectives",
            stage="model-parsing",
            details={"candidate_id": candidate_id},
        )
    if not isinstance(objectives, list):
        raise ValidationFailure(
            f"Candidate {candidate_id!r} objectives must be a list.",
            code="invalid_objectives_shape",
            stage="model-parsing",
            details={"candidate_id": candidate_id},
        )
    if len(objectives) != store.objective_count:
        raise ValidationFailure(
            f"Candidate {candidate_id!r} has {len(objectives)} objectives; "
            f"problem {store.problem_id!r} requires {store.objective_count}.",
            code="objective_count_mismatch",
            stage="model-parsing",
            details={
                "candidate_id": candidate_id,
                "received": len(objectives),
                "required": store.objective_count,
            },
        )
    missing_indexes = [i for i, value in enumerate(objectives) if value is None]
    if missing_indexes:
        raise ValidationFailure(
            f"Candidate {candidate_id!r} has missing objective values.",
            code="missing_objective_value",
            stage="model-parsing",
            details={
                "candidate_id": candidate_id,
                "missing_indexes": missing_indexes,
            },
        )

    exact_values = [
        parse_fraction(
            raw_value,
            location=f"candidate[{candidate_id}].objectives[{index}]",
        )
        for index, raw_value in enumerate(objectives)
    ]
    canonical = tuple(
        store.policy.normalize_transformed(sign * value)
        for sign, value in zip(store.signs, exact_values, strict=True)
    )
    return CandidateRecord(
        id=candidate_id,
        declared=tuple(objectives),
        exact_declared=tuple(exact_values),
        canonical=canonical,
        payload=payload,
        group_id="",
        rank=0,
    )


def _add_edge(store: ProblemStore, up_id: str, down_id: str) -> bool:
    if down_id in store.group_down[up_id]:
        return False
    store.group_down[up_id].add(down_id)
    store.group_up[down_id].add(up_id)
    return True


def _build_initial_groups(store: ProblemStore, records: list[CandidateRecord]) -> int:
    for record in records:
        vector = record.canonical
        group_id = store.group_by_vector.get(vector)
        if group_id is None:
            group_id = store._new_group_id()
            store.group_by_vector[vector] = group_id
            store.groups[group_id] = ObjectiveGroup(
                id=group_id, canonical=vector, member_ids=[record.id]
            )
            store.group_order.append(group_id)
            store.group_down[group_id] = set()
            store.group_up[group_id] = set()
        else:
            store.groups[group_id].member_ids.append(record.id)
        record.group_id = group_id
        store.points[record.id] = record

    groups = store.ordered_groups()
    if not groups:
        return 0
    vectors = np.asarray([group.canonical for group in groups], dtype=object)
    matrix = _strict_matrix(vectors, vectors)
    edges_added = 0
    for i, up_group in enumerate(groups):
        for j, down_group in enumerate(groups):
            if matrix[i, j] and _add_edge(store, up_group.id, down_group.id):
                edges_added += 1
    _rerank_with_scipy(store)
    return edges_added


def initialize_problem(
    store: ProblemStore,
    rows: Iterable[tuple[str, list[Any] | None, dict[str, Any] | None]],
) -> MutationEvidence:
    records: list[CandidateRecord] = []
    seen_ids: set[str] = set()
    for index, (raw_id, objectives, payload) in enumerate(rows):
        candidate_id = raw_id.strip() if isinstance(raw_id, str) else raw_id
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValidationFailure(
                "Candidate id is required and must be a non-empty string.",
                code="invalid_candidate_id",
                stage="model-parsing",
                details={"row": index},
            )
        if candidate_id in seen_ids:
            raise Conflict(
                f"Candidate id {candidate_id!r} appears more than once.",
                code="duplicate_candidate_id",
                stage="model-parsing",
                details={"candidate_id": candidate_id},
            )
        seen_ids.add(candidate_id)
        records.append(_parse_candidate(store, candidate_id, objectives, payload))

    group_count = len({record.canonical for record in records})
    pair_comparisons = group_count * max(0, group_count - 1)
    edges_added = _build_initial_groups(store, records)
    return MutationEvidence(
        operation="problem.initialize",
        stage="initial-full-comparison",
        steps=[
            "Parsed all candidate objectives as exact declared values.",
            "Grouped identical canonical vectors without merging object identities.",
            f"Compared {group_count} canonical groups with vectorized NumPy predicates.",
            "Computed initial ranks as longest paths in the strict dominance DAG.",
        ],
        pair_comparisons=pair_comparisons,
        edges_added=edges_added,
        edges_removed=0,
        affected_candidate_ids=[record.id for record in records],
        inserted_candidate_ids=[record.id for record in records],
    )


def add_candidates(
    store: ProblemStore,
    rows: Iterable[tuple[str, list[Any] | None, dict[str, Any] | None]],
) -> MutationEvidence:
    records: list[CandidateRecord] = []
    seen_ids: set[str] = set()
    for raw_id, objectives, payload in rows:
        candidate_id = raw_id.strip() if isinstance(raw_id, str) else raw_id
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValidationFailure(
                "Candidate id is required and must be a non-empty string.",
                code="invalid_candidate_id",
                stage="model-parsing",
            )
        if candidate_id in store.points or candidate_id in seen_ids:
            raise Conflict(
                f"Candidate id {candidate_id!r} already exists.",
                code="duplicate_candidate_id",
                stage="model-parsing",
                details={"candidate_id": candidate_id},
            )
        seen_ids.add(candidate_id)
        records.append(_parse_candidate(store, candidate_id, objectives, payload))

    if not records:
        raise ValidationFailure(
            "At least one candidate is required.",
            code="empty_candidate_batch",
            stage="model-parsing",
        )

    existing_groups = store.ordered_groups()
    unique_new_vectors: list[tuple[Fraction, ...]] = []
    for record in records:
        if (
            record.canonical not in unique_new_vectors
            and record.canonical not in store.group_by_vector
        ):
            unique_new_vectors.append(record.canonical)

    pair_comparisons = 0
    if existing_groups and unique_new_vectors:
        pair_comparisons += len(unique_new_vectors) * len(existing_groups)
    if len(unique_new_vectors) > 1:
        pair_comparisons += len(unique_new_vectors) * (len(unique_new_vectors) - 1)

    roots: set[str] = set()
    for record in records:
        group_id = store.group_by_vector.get(record.canonical)
        if group_id is not None:
            roots.add(group_id)
            record.group_id = group_id
            store.groups[group_id].member_ids.append(record.id)
            store.points[record.id] = record

    created: list[ObjectiveGroup] = []
    for vector in unique_new_vectors:
        group_id = store._new_group_id()
        group = ObjectiveGroup(
            id=group_id, canonical=vector, member_ids=[]
        )
        created.append(group)
        store.groups[group_id] = group
        store.group_order.append(group_id)
        store.group_by_vector[vector] = group_id
        store.group_down[group_id] = set()
        store.group_up[group_id] = set()
        roots.add(group_id)

    edges_added = 0
    if existing_groups and unique_new_vectors:
        new_array = np.asarray(unique_new_vectors, dtype=object)
        old_array = np.asarray(
            [group.canonical for group in existing_groups], dtype=object
        )
        new_dominates = _strict_matrix(new_array, old_array)
        old_dominates = _strict_matrix(old_array, new_array)
        for i, group in enumerate(created):
            for j, old_group in enumerate(existing_groups):
                if new_dominates[i, j] and _add_edge(store, group.id, old_group.id):
                    edges_added += 1
                if old_dominates[j, i] and _add_edge(store, old_group.id, group.id):
                    edges_added += 1

    if len(created) > 1:
        vectors = np.asarray(unique_new_vectors, dtype=object)
        matrix = _strict_matrix(vectors, vectors)
        for i, up_group in enumerate(created):
            for j, down_group in enumerate(created):
                if matrix[i, j] and _add_edge(store, up_group.id, down_group.id):
                    edges_added += 1

    for record in records:
        if not record.group_id:
            group_id = store.group_by_vector[record.canonical]
            record.group_id = group_id
            store.groups[group_id].member_ids.append(record.id)
        store.points[record.id] = record

    affected = _descendants(store, roots)
    _rerank_affected(store, affected)
    affected_ids = [
        candidate_id
        for group_id in affected
        for candidate_id in store.groups[group_id].member_ids
    ]
    return MutationEvidence(
        operation="candidate.insert",
        stage="incremental-local-rerank",
        steps=[
            "Validated identities and exact objective vectors before mutating state.",
            "Attached duplicate identities to their existing canonical group.",
            f"Compared {len(unique_new_vectors)} new groups against the current graph.",
            "Restricted rank repair to inserted groups and their dominated descendants.",
        ],
        pair_comparisons=pair_comparisons,
        edges_added=edges_added,
        edges_removed=0,
        affected_candidate_ids=affected_ids,
        inserted_candidate_ids=[record.id for record in records],
    )


def delete_candidate(store: ProblemStore, candidate_id: str) -> MutationEvidence:
    if candidate_id not in store.points:
        raise ValidationFailure(
            f"Candidate id {candidate_id!r} does not exist.",
            code="candidate_not_found",
            stage="candidate-delete",
            status_code=404,
            details={"candidate_id": candidate_id},
        )

    record = store.points[candidate_id]
    group = store.groups[record.group_id]
    removed_group = len(group.member_ids) == 1
    if removed_group:
        affected = _descendants(store, [group.id]) - {group.id}
        edges_removed = len(store.group_down[group.id]) + len(
            store.group_up[group.id]
        )
        affected_ids = [
            member_id
            for affected_group_id in affected
            for member_id in store.groups[affected_group_id].member_ids
        ]
        for down_id in list(store.group_down[group.id]):
            store.group_up[down_id].discard(group.id)
        for up_id in list(store.group_up[group.id]):
            store.group_down[up_id].discard(group.id)
        del store.groups[group.id]
        del store.group_down[group.id]
        del store.group_up[group.id]
        del store.group_by_vector[group.canonical]
        store.group_order.remove(group.id)
        _rerank_affected(store, affected)
    else:
        edges_removed = 0
        affected = {group.id}
        affected_ids = list(group.member_ids)

    group.member_ids.remove(candidate_id)
    del store.points[candidate_id]
    return MutationEvidence(
        operation="candidate.delete",
        stage="incremental-downstream-restore"
        if removed_group
        else "duplicate-identity-remove",
        steps=[
            "Resolved the requested object identity to its canonical group.",
            "Removed one identity while retaining all unrelated duplicate identities."
            if not removed_group
            else "Removed the final group and all of its incident dominance edges.",
            "Restored ranks for every group previously dominated by the removed group."
            if removed_group
            else "Shared dominance relations and rank remain attached to duplicate peers.",
        ],
        pair_comparisons=0,
        edges_added=0,
        edges_removed=edges_removed,
        affected_candidate_ids=affected_ids,
    )


def candidate_relations(
    store: ProblemStore, candidate_id: str
) -> tuple[CandidateRecord, list[str], list[str], list[str]]:
    if candidate_id not in store.points:
        raise ValidationFailure(
            f"Candidate id {candidate_id!r} does not exist.",
            code="candidate_not_found",
            stage="candidate-relation-read",
            status_code=404,
            details={"candidate_id": candidate_id},
        )
    record = store.points[candidate_id]
    group = store.groups[record.group_id]
    down_groups = store.group_down[group.id]
    up_groups = store.group_up[group.id]
    dominates_ids = sorted(
        member_id
        for down_id in down_groups
        for member_id in store.groups[down_id].member_ids
    )
    dominated_by_ids = sorted(
        member_id
        for up_id in up_groups
        for member_id in store.groups[up_id].member_ids
    )
    duplicate_ids = sorted(member_id for member_id in group.member_ids)
    return record, dominates_ids, dominated_by_ids, duplicate_ids
