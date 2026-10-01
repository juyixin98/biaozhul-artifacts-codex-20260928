from __future__ import annotations

from fractions import Fraction

import pytest

from app.core import add_candidates, candidate_relations, delete_candidate
from app.errors import Conflict, ValidationFailure
from app.parsing import build_policy
from app.core import create_store, dominates


def transformed(point, directions):
    result = []
    for value, direction in zip(point, directions, strict=True):
        number = Fraction(str(value)) if not isinstance(value, Fraction) else value
        result.append(number if direction == "max" else -number)
    return tuple(result)


def oracle_layers(points, directions):
    """Independent exhaustive reference: rank = 1 + count of distinct dominators."""
    vectors = {
        identifier: transformed(values, directions)
        for identifier, values in points.items()
    }
    dominator_count = {}
    for identifier, vector in vectors.items():
        dominators = set()
        for other_id, other in vectors.items():
            if other_id == identifier:
                continue
            if dominates(other, vector):
                dominators.add(other)
        dominator_count[identifier] = len(dominators)
    maximum = max(dominator_count.values(), default=-1)
    return [
        sorted(
            identifier
            for identifier, count in dominator_count.items()
            if count == level
        )
        for level in range(maximum + 1)
    ]


def assert_matches_oracle(store, expected_points, directions=None):
    directions = directions or list(store.directions)
    actual = [sorted(layer) for layer in store.layers()]
    expected = [
        layer for layer in oracle_layers(expected_points, directions) if layer
    ]
    assert actual == expected
    for layer_number, layer in enumerate(actual, start=1):
        for left_id in layer:
            for right_id in layer:
                if left_id == right_id:
                    continue
                left = store.points[left_id].canonical
                right = store.points[right_id].canonical
                assert not dominates(left, right)
                assert not dominates(right, left)


def test_two_dimensional_hand_calculated_mixed_directions(empty_store):
    points = {
        "a": (1, 1),
        "b": (2, 2),
        "c": (1, 1),
        "d": (3, 1),
        "e": (2, 0),
    }
    add_candidates(empty_store, [(key, list(value), None) for key, value in points.items()])

    assert empty_store.layers() == [["a", "b", "c"], ["d", "e"]]
    assert_matches_oracle(empty_store, points, ["min", "max"])

    # a and c are distinct objects, but their duplicate group carries one relation set.
    _, dominated_a, dominators_a, duplicates_a = candidate_relations(empty_store, "a")
    _, dominated_c, dominators_c, duplicates_c = candidate_relations(empty_store, "c")
    assert duplicates_a == ["a", "c"]
    assert duplicates_c == ["a", "c"]
    assert dominated_a == ["d", "e"]
    assert dominated_c == ["d", "e"]
    assert dominators_a == []
    assert dominators_c == []
    assert empty_store.points["a"].group_id == empty_store.points["c"].group_id


def test_all_max_and_high_dimensional_degenerate_case():
    store = create_store(
        problem_id="high-dim",
        directions=["max"] * 4,
        policy=build_policy("exact"),
    )
    points = {
        "p1": (1, 0, 0, 0),
        "p2": (0, 1, 0, 0),
        "p3": (0, 0, 1, 0),
        "p4": (0, 0, 0, 1),
        "p5": (0, 0, 0, 0),
        "p6": (1, 0, 0, 0),
    }
    add_candidates(store, [(key, list(value), None) for key, value in points.items()])

    assert store.layers() == [["p1", "p2", "p3", "p4", "p6"], ["p5"]]
    assert_matches_oracle(store, points, ["max"] * 4)


def test_incremental_insertion_promotes_only_affected_layers(empty_store):
    initial = {
        "a": (0, 0),
        "b": (2, -2),
        "c": (4, -4),
    }
    add_candidates(empty_store, [(key, list(value), None) for key, value in initial.items()])
    assert empty_store.layers() == [["a"], ["b"], ["c"]]

    add_candidates(empty_store, [("x", [1, -1], None)])
    assert empty_store.layers() == [["a"], ["x"], ["b"], ["c"]]

    # Inserting a duplicate preserves the other object's identity and rank.
    add_candidates(empty_store, [("b-copy", [2, -2], None)])
    assert empty_store.points["b"].rank == 3
    assert empty_store.points["b-copy"].rank == 3
    assert empty_store.layers() == [["a"], ["x"], ["b", "b-copy"], ["c"]]


def test_deletion_restores_subsequent_layers(empty_store):
    points = {
        "top1": (0, 0),
        "top2": (1, 1),
        "middle": (2, 0),
        "bottom": (3, -1),
    }
    add_candidates(empty_store, [(key, list(value), None) for key, value in points.items()])
    assert empty_store.layers() == [["top1", "top2"], ["middle"], ["bottom"]]

    # top2 was middle's only remaining upper neighbor after deleting top1.
    delete_candidate(empty_store, "top1")
    assert empty_store.layers() == [["top2"], ["middle"], ["bottom"]]
    evidence = delete_candidate(empty_store, "top2")
    assert empty_store.layers() == [["middle"], ["bottom"]]
    assert evidence.stage == "incremental-downstream-restore"
    assert evidence.edges_removed == 2


def test_deleting_one_duplicate_identity_keeps_shared_relations(empty_store):
    add_candidates(
        empty_store,
        [
            ("a", [1, 1], None),
            ("a-copy", [1, 1], None),
            ("b", [2, 0], None),
        ],
    )
    evidence = delete_candidate(empty_store, "a-copy")
    assert evidence.stage == "duplicate-identity-remove"
    assert empty_store.layers() == [["a"], ["b"]]
    _, dominated, dominators, duplicates = candidate_relations(empty_store, "a")
    assert duplicates == ["a"]
    assert dominated == ["b"]
    assert dominators == []


def test_missing_objectives_are_rejected_with_specific_failure(empty_store):
    with pytest.raises(ValidationFailure) as missing_shape:
        add_candidates(empty_store, [("bad", None, None)])
    assert missing_shape.value.code == "missing_objectives"

    with pytest.raises(ValidationFailure) as missing_value:
        add_candidates(empty_store, [("bad", [1, None], None)])
    assert missing_value.value.code == "missing_objective_value"
    assert missing_value.value.details["missing_indexes"] == [1]

    with pytest.raises(ValidationFailure) as wrong_count:
        add_candidates(empty_store, [("bad", [1], None)])
    assert wrong_count.value.code == "objective_count_mismatch"


def test_exact_default_preserves_tiny_decimal_difference():
    store = create_store(
        problem_id="exact-decimals",
        directions=["min", "min"],
        policy=build_policy("exact"),
    )
    add_candidates(
        store,
        [
            ("a", ["0.10", "0.20"], None),
            ("b", ["0.1", "0.20000000000000000001"], None),
        ],
    )
    assert store.points["a"].canonical != store.points["b"].canonical
    assert dominates(store.points["a"].canonical, store.points["b"].canonical)


def test_quantized_tolerance_remains_transitive_and_reports_uncertainty():
    policy = build_policy("quantized", "0.1")
    store = create_store(
        problem_id="grid",
        directions=["min", "min"],
        policy=policy,
    )
    add_candidates(
        store,
        [
            ("a", ["0", "0"], None),
            ("b", ["0.05", "0.05"], None),
            ("c", ["-0.05", "-0.05"], None),
            # D and B are incomparable after snap: x favors D, y favors B.
            ("d", ["0.06", "0.0"], None),
        ],
    )
    assert store.points["a"].canonical == store.points["b"].canonical
    assert store.points["b"].canonical == store.points["c"].canonical
    assert store.points["d"].canonical != store.points["b"].canonical
    assert store.layers() == [["a", "b", "c"], ["d"]]


def test_duplicate_candidate_id_rejected_before_state_mutation(empty_store):
    add_candidates(empty_store, [("a", [0, 0], None)])
    with pytest.raises(Conflict) as error:
        add_candidates(
            empty_store,
            [("b", [9, 9], None), ("a", [1, 1], None)],
        )
    assert error.value.code == "duplicate_candidate_id"
    assert set(empty_store.points) == {"a"}
