from __future__ import annotations

import random

import pytest

from app.kernel import (
    CandidateInput,
    DuplicateCandidateId,
    IncrementalSorter,
    KernelError,
    Objective,
    UnknownCandidate,
    dominates,
)


MIN_MIN = (Objective("cost", "min"), Objective("risk", "min"))
MIXED = (Objective("cost", "min"), Objective("reliability", "max"))


def c(candidate_id: str, values: tuple[float, ...]) -> CandidateInput:
    return CandidateInput(id=candidate_id, vector=values)


def local_dominates(left, right, directions):
    better_any = False
    for left_value, right_value, direction in zip(left, right, directions):
        if direction == "min":
            if left_value > right_value:
                return False
            if left_value < right_value:
                better_any = True
        else:
            if left_value < right_value:
                return False
            if left_value > right_value:
                better_any = True
    return better_any


def peeling_layers(candidates, directions):
    remaining = set(candidates)
    result = []
    while remaining:
        front = sorted(
            candidate_id
            for candidate_id in remaining
            if not any(
                other_id != candidate_id
                and local_dominates(
                    candidates[other_id], candidates[candidate_id], directions
                )
                for other_id in remaining
            )
        )
        result.append(front)
        remaining.difference_update(front)
    return result


def assert_matches_reference(sorter):
    vectors = {
        candidate_id: list(candidate.vector)
        for candidate_id, candidate in sorter.candidates.items()
    }
    reference = peeling_layers(vectors, sorter.directions)
    assert sorter.layers() == reference

    for layer in sorter.layers():
        for index, left_id in enumerate(layer):
            for right_id in layer[index + 1 :]:
                assert not dominates(
                    sorter.candidates[left_id].vector,
                    sorter.candidates[right_id].vector,
                    sorter.directions,
                )
                assert not dominates(
                    sorter.candidates[right_id].vector,
                    sorter.candidates[left_id].vector,
                    sorter.directions,
                )


def test_two_dimensional_hand_calculation_and_duplicate_identities():
    sorter = IncrementalSorter(MIXED)
    items = [
        c("A", (1, 3)),
        c("B", (3, 1)),
        c("C", (2, 2)),
        c("D", (1, 3)),
        c("E", (2, 3)),
        c("F", (4, 0)),
    ]
    sorter.add_candidates(items)

    assert sorter.layers() == [["A", "D"], ["E"], ["C"], ["B"], ["F"]]
    assert sorter.ranks["A"] == sorter.ranks["D"] == 0
    assert sorter.parents["A"] == set()
    assert sorter.parents["D"] == set()
    assert sorter.children["A"] == sorter.children["D"] == {"B", "C", "E", "F"}
    assert_matches_reference(sorter)


def test_deleting_one_duplicate_preserves_shared_domination_then_recovers_layers():
    sorter = IncrementalSorter(MIXED)
    sorter.add_candidates(
        [
            c("A", (1, 3)),
            c("D", (1, 3)),
            c("E", (2, 3)),
            c("C", (2, 2)),
            c("B", (3, 1)),
            c("F", (4, 0)),
        ]
    )

    first_delete = sorter.remove_candidate("A")
    assert first_delete.rank_changes == ()
    assert sorter.layers() == [["D"], ["E"], ["C"], ["B"], ["F"]]
    assert sorter.children["D"] == {"B", "C", "E", "F"}

    second_delete = sorter.remove_candidate("D")
    assert [change.id for change in second_delete.rank_changes] == ["B", "C", "E", "F"]
    assert sorter.layers() == [["E"], ["C"], ["B"], ["F"]]
    assert_matches_reference(sorter)


def test_incremental_insert_promotes_existing_layers():
    sorter = IncrementalSorter(MIXED)
    sorter.add_candidates(
        [
            c("A", (1, 3)),
            c("D", (1, 3)),
            c("E", (2, 3)),
            c("C", (2, 2)),
            c("B", (3, 1)),
            c("F", (4, 0)),
        ]
    )

    report = sorter.add_candidates([c("P", (0, 3))])
    assert sorter.layers() == [["P"], ["A", "D"], ["E"], ["C"], ["B"], ["F"]]
    assert {change.id: (change.from_rank, change.to_rank) for change in report.promotions} == {
        "A": (0, 1),
        "D": (0, 1),
        "E": (1, 2),
        "C": (2, 3),
        "B": (3, 4),
        "F": (4, 5),
    }


def test_mixed_directions_are_fixed_per_dimension():
    sorter = IncrementalSorter(MIXED)
    sorter.add_candidates([c("A", (1, 10)), c("B", (2, 9)), c("C", (2, 10))])

    assert sorter.layers() == [["A"], ["C"], ["B"]]
    assert dominates((1, 10), (2, 9), ("min", "max"))
    assert not dominates((2, 9), (1, 10), ("min", "max"))


def test_high_dimensional_degenerate_objectives_and_tie_front():
    objectives = (
        Objective("x1", "min"),
        Objective("x2", "min"),
        Objective("x3", "min"),
        Objective("constant", "min"),
    )
    sorter = IncrementalSorter(objectives)
    sorter.add_candidates(
        [
            c("p1", (0, 1, 1, 5)),
            c("p2", (1, 0, 1, 5)),
            c("p3", (1, 1, 0, 5)),
            c("dominated", (1, 1, 1, 5)),
            c("p1-duplicate", (0, 1, 1, 5)),
        ]
    )

    assert sorter.layers() == [["p1", "p1-duplicate", "p2", "p3"], ["dominated"]]
    assert_matches_reference(sorter)


@pytest.mark.parametrize("seed", [1, 17, 42, 99])
def test_random_insertion_and_deletion_matches_all_pairs_peeling(seed):
    rng = random.Random(seed)
    directions = tuple(rng.choice(["min", "max"]) for _ in range(4))
    objectives = tuple(Objective(f"o{i}", direction) for i, direction in enumerate(directions))
    sorter = IncrementalSorter(objectives)

    values = [0, 1, 2, 2, 3]
    all_items = []
    for index in range(32):
        vector = tuple(rng.choice(values) for _ in directions)
        all_items.append(c(f"n{index:02d}", vector))

    rng.shuffle(all_items)
    split = 18
    sorter.add_candidates(all_items[:split])
    assert_matches_reference(sorter)
    sorter.add_candidates(all_items[split:])
    assert_matches_reference(sorter)

    removal_order = [candidate.id for candidate in all_items]
    rng.shuffle(removal_order)
    for candidate_id in removal_order:
        sorter.remove_candidate(candidate_id)
        assert_matches_reference(sorter)

    assert sorter.layers() == []


def test_rejected_batch_is_atomic_and_unknown_delete_is_specific():
    sorter = IncrementalSorter(MIN_MIN)
    sorter.add_candidates([c("existing", (1, 1))])

    with pytest.raises(DuplicateCandidateId, match="existing"):
        sorter.add_candidates([c("new", (0, 0)), c("existing", (2, 2))])

    assert set(sorter.candidates) == {"existing"}
    assert sorter.layers() == [["existing"]]

    with pytest.raises(UnknownCandidate, match="missing"):
        sorter.remove_candidate("missing")


def test_kernel_rejects_wrong_dimension_and_empty_objectives():
    sorter = IncrementalSorter(MIN_MIN)
    with pytest.raises(KernelError, match="expected 2"):
        sorter.add_candidates([c("bad", (1, 2, 3))])

    with pytest.raises(KernelError, match="at least one objective"):
        IncrementalSorter(())
