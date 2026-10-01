"""Exact incremental non-dominated sorting kernel.

A graph edge ``p -> c`` means that candidate ``p`` dominates ``c`` under
the fixed per-objective directions.  Edges are retained for every pairwise
domination relation, including transitive relations.  The layer number is
therefore the longest path length from any source in this DAG.  Equal
vectors receive no edge between themselves; each equal object retains its
own identity and its own dominance edges to other vectors.

No comparison tolerance is accepted here.  Tolerances can be useful in a
diagnostic report, but applying an approximate predicate directly can create
cycles and break dominance transitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

Direction = Literal["min", "max"]
Vector = tuple[float | int, ...]


@dataclass(frozen=True)
class Objective:
    name: str
    direction: Direction


@dataclass(frozen=True)
class CandidateInput:
    id: str
    vector: Vector


@dataclass(frozen=True)
class RankChange:
    id: str
    from_rank: int
    to_rank: int


@dataclass(frozen=True)
class InsertionReport:
    inserted_ids: tuple[str, ...]
    promotions: tuple[RankChange, ...]


@dataclass(frozen=True)
class DeletionReport:
    deleted_id: str
    affected_ids: tuple[str, ...]
    rank_changes: tuple[RankChange, ...]


class KernelError(ValueError):
    """Base class for deterministic kernel validation failures."""


class DuplicateCandidateId(KernelError):
    def __init__(self, candidate_id: str) -> None:
        self.candidate_id = candidate_id
        super().__init__(f"candidate id already exists: {candidate_id}")


class UnknownCandidate(KernelError):
    def __init__(self, candidate_id: str) -> None:
        self.candidate_id = candidate_id
        super().__init__(f"unknown candidate id: {candidate_id}")


def dominates(
    left: Sequence[float | int],
    right: Sequence[float | int],
    directions: Sequence[Direction],
) -> bool:
    """Return whether ``left`` exactly dominates ``right``.

    ``left`` must be no worse in every objective and strictly better in at
    least one.  A maximization objective treats larger values as better.
    """

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


class IncrementalSorter:
    def __init__(self, objectives: Sequence[Objective]) -> None:
        if not objectives:
            raise KernelError("at least one objective is required")
        self.objectives = tuple(objectives)
        self.directions = tuple(objective.direction for objective in objectives)
        self.candidates: dict[str, CandidateInput] = {}
        self.ranks: dict[str, int] = {}
        self.parents: dict[str, set[str]] = {}
        self.children: dict[str, set[str]] = {}
        self.version: int = 0

    @property
    def objective_names(self) -> tuple[str, ...]:
        return tuple(objective.name for objective in self.objectives)

    def add_candidates(self, items: Iterable[CandidateInput]) -> InsertionReport:
        items = tuple(items)
        before = dict(self.ranks)
        seen_in_batch: set[str] = set()
        inserted: list[str] = []

        # Validate the complete batch before mutating state so a rejected batch
        # remains atomic.
        for item in items:
            if item.id in seen_in_batch:
                raise DuplicateCandidateId(item.id)
            if item.id in self.candidates:
                raise DuplicateCandidateId(item.id)
            if len(item.vector) != len(self.objectives):
                raise KernelError(
                    f"candidate {item.id} has {len(item.vector)} objectives, "
                    f"expected {len(self.objectives)}"
                )
            seen_in_batch.add(item.id)

        for item in items:
            candidate_id = item.id
            self.candidates[candidate_id] = item
            self.parents[candidate_id] = set()
            self.children[candidate_id] = set()

            for other_id, other in self.candidates.items():
                if other_id == candidate_id:
                    continue
                if dominates(item.vector, other.vector, self.directions):
                    self._add_edge(candidate_id, other_id)
                elif dominates(other.vector, item.vector, self.directions):
                    self._add_edge(other_id, candidate_id)

            parent_rank_max = max(
                (self.ranks[parent_id] for parent_id in self.parents[candidate_id]),
                default=-1,
            )
            self.ranks[candidate_id] = parent_rank_max + 1
            self._promote_descendants(candidate_id)
            inserted.append(candidate_id)

        self.version += 1
        promotions = tuple(
            RankChange(candidate_id, before[candidate_id], self.ranks[candidate_id])
            for candidate_id in before
            if self.ranks[candidate_id] != before[candidate_id]
        )
        return InsertionReport(tuple(inserted), promotions)

    def remove_candidate(self, candidate_id: str) -> DeletionReport:
        if candidate_id not in self.candidates:
            raise UnknownCandidate(candidate_id)

        affected = self._descendants(candidate_id)
        old_ranks = {node_id: self.ranks[node_id] for node_id in affected}

        for parent_id in self.parents[candidate_id]:
            self.children[parent_id].discard(candidate_id)
        for child_id in self.children[candidate_id]:
            self.parents[child_id].discard(candidate_id)

        del self.candidates[candidate_id]
        del self.ranks[candidate_id]
        del self.parents[candidate_id]
        del self.children[candidate_id]

        self._recover_descendants(affected)
        self.version += 1

        changes = tuple(
            RankChange(node_id, old_ranks[node_id], self.ranks[node_id])
            for node_id in sorted(affected)
            if old_ranks[node_id] != self.ranks[node_id]
        )
        return DeletionReport(candidate_id, tuple(sorted(affected)), changes)

    def layers(self) -> list[list[str]]:
        if not self.ranks:
            return []
        layer_count = max(self.ranks.values()) + 1
        result: list[list[str]] = [[] for _ in range(layer_count)]
        for candidate_id, rank in self.ranks.items():
            result[rank].append(candidate_id)
        return [sorted(layer) for layer in result]

    def duplicate_groups(self) -> list[list[str]]:
        groups: dict[Vector, list[str]] = {}
        for candidate_id, candidate in self.candidates.items():
            groups.setdefault(candidate.vector, []).append(candidate_id)
        return [sorted(ids) for ids in groups.values() if len(ids) > 1]

    def edge_count(self) -> int:
        return sum(len(children) for children in self.children.values())

    def snapshot_candidates(self) -> tuple[CandidateInput, ...]:
        return tuple(self.candidates[candidate_id] for candidate_id in sorted(self.candidates))

    def _add_edge(self, parent_id: str, child_id: str) -> None:
        self.children[parent_id].add(child_id)
        self.parents[child_id].add(parent_id)

    def _promote_descendants(self, start_id: str) -> None:
        stack: list[str] = [start_id]
        while stack:
            parent_id = stack.pop()
            required_rank = self.ranks[parent_id] + 1
            for child_id in self.children[parent_id]:
                if self.ranks[child_id] < required_rank:
                    self.ranks[child_id] = required_rank
                    stack.append(child_id)

    def _descendants(self, start_id: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.children[start_id])
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            stack.extend(self.children[node_id])
        return seen

    def _recover_descendants(self, affected: set[str]) -> None:
        if not affected:
            return

        internal_predecessors = {
            node_id: {
                parent_id
                for parent_id in self.parents[node_id]
                if parent_id in affected
            }
            for node_id in affected
        }
        ready = [
            node_id
            for node_id in affected
            if not internal_predecessors[node_id]
        ]
        recomputed: set[str] = set()

        while ready:
            node_id = ready.pop()
            current_parent_rank_max = max(
                (
                    self.ranks[parent_id]
                    for parent_id in self.parents[node_id]
                ),
                default=-1,
            )
            self.ranks[node_id] = current_parent_rank_max + 1
            recomputed.add(node_id)

            for child_id in self.children[node_id]:
                if child_id not in affected:
                    continue
                internal_predecessors[child_id].discard(node_id)
                if not internal_predecessors[child_id]:
                    ready.append(child_id)

        if recomputed != affected:
            raise RuntimeError("dominance graph contains a cycle; this indicates a kernel bug")
