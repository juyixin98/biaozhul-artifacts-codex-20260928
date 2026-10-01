"""Independent post-operation evidence and numerical diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from app.core import CandidateRecord, ProblemStore, dominates
from app.parsing import fraction_to_string


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    details: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "details": self.details}


def check_layers(store: ProblemStore) -> list[CheckResult]:
    checks: list[CheckResult] = []
    groups = store.ordered_groups()
    ranks = sorted({group.rank for group in groups})
    expected_ranks = list(range(1, len(ranks) + 1))
    checks.append(
        CheckResult(
            "contiguous-one-based-ranks",
            ranks == expected_ranks,
            f"Observed ranks {ranks}; expected {expected_ranks}.",
        )
    )

    layer_ok = True
    layer_details = "No dominance edge exists inside any layer."
    for rank in ranks:
        members = [g for g in groups if g.rank == rank]
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                if (
                    right.id in store.group_down[left.id]
                    or left.id in store.group_down[right.id]
                ):
                    layer_ok = False
                    layer_details = (
                        f"Layer {rank} contains dominance between groups "
                        f"{left.id} and {right.id}."
                    )
                    break

    checks.append(CheckResult("layer-wise-non-dominance", layer_ok, layer_details))

    invariant_ok = True
    invariant_details = "Every rank equals one plus its strongest dominator rank."
    for group in groups:
        upper_ranks = [
            store.groups[up_id].rank for up_id in store.group_up[group.id]
        ]
        expected = 1 + max(upper_ranks, default=0)
        if group.rank != expected:
            invariant_ok = False
            invariant_details = (
                f"{group.id} has rank {group.rank} but rank {expected} "
                "follows from its incoming dominance edges."
            )
            break
    checks.append(CheckResult("rank-longest-path-invariant", invariant_ok, invariant_details))

    duplicate_ok = True
    duplicate_details = "All duplicate identities share one group, rank, and relation set."
    for group in groups:
        if len(group.member_ids) != len(set(group.member_ids)):
            duplicate_ok = False
            duplicate_details = f"{group.id} contains a repeated member identity."
            break
        for candidate_id in group.member_ids:
            if store.points[candidate_id].group_id != group.id:
                duplicate_ok = False
                duplicate_details = f"{candidate_id} is not linked to {group.id}."
            if store.points[candidate_id].rank != group.rank:
                duplicate_ok = False
                duplicate_details = f"{candidate_id} does not inherit {group.id} rank."
    checks.append(CheckResult("duplicate-identity-sharing", duplicate_ok, duplicate_details))

    identity_count = sum(len(group.member_ids) for group in groups)
    count_ok = identity_count == len(store.points)
    checks.append(
        CheckResult(
            "candidate-identity-count",
            count_ok,
            f"{identity_count} grouped identities; registry contains {len(store.points)}.",
        )
    )
    return checks


def numeric_uncertainty(store: ProblemStore) -> list[str]:
    if store.policy.is_exact:
        return []
    tolerance = fraction_to_string(store.policy.tolerance)  # type: ignore[arg-type]
    notes = [
        "Quantized mode is an exact strict partial order on snapped values, but "
        f"declared values within half of tolerance {tolerance} may be indistinguishable."
    ]
    for candidate in store.ordered_points():
        signed = [
            sign * value
            for sign, value in zip(store.signs, candidate.exact_declared, strict=True)
        ]
        snapped = list(candidate.canonical)
        changed = [
            f"objective[{index}] {fraction_to_string(before)} -> "
            f"{fraction_to_string(after)}"
            for index, (before, after) in enumerate(zip(signed, snapped, strict=True))
            if before != after
        ]
        if changed:
            notes.append(f"Candidate {candidate.id}: " + "; ".join(changed) + ".")
    return notes


def pairwise_relation(
    left: CandidateRecord, right: CandidateRecord
) -> str:
    """Small explicit diagnostic independent of the adjacency graph."""
    if left.canonical == right.canonical:
        return "equal"
    if dominates(left.canonical, right.canonical):
        return "dominates"
    if dominates(right.canonical, left.canonical):
        return "dominated_by"
    return "incomparable"
