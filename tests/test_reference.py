"""Property-style checks against a deliberately separate exhaustive oracle."""

from __future__ import annotations

from fractions import Fraction

from app.core import add_candidates, create_store, delete_candidate, dominates
from app.parsing import build_policy


def brute_force_layers(points, directions):
    transformed = {}
    for identifier, values in points.items():
        transformed[identifier] = tuple(
            Fraction(value) if direction == "max" else -Fraction(value)
            for value, direction in zip(values, directions, strict=True)
        )

    direct_dominators = {
        identifier: set() for identifier in transformed
    }
    for left_id, left in transformed.items():
        for right_id, right in transformed.items():
            if left_id != right_id and dominates(left, right):
                direct_dominators[right_id].add(left_id)

    # Standard non-dominated fronts, implemented by repeated exhaustive removal.
    remaining = set(transformed)
    fronts = []
    while remaining:
        front = []
        for identifier in sorted(remaining):
            if not (direct_dominators[identifier] & remaining):
                front.append(identifier)
        fronts.append(sorted(front))
        remaining.difference_update(front)
    return fronts


def rows_from(points):
    return [(identifier, list(values), None) for identifier, values in points.items()]


def test_mixed_direction_degenerate_high_dimensional_points_match_brute_force():
    directions = ["min", "max", "min", "max"]
    points = {
        "a": (1, 0, 9, 0),
        "b": (1, 0, 9, 0),
        "c": (0, 1, 9, 0),
        "d": (1, 0, 8, 1),
        "e": (0, 0, 0, 0),
        "f": (2, 2, 10, 2),
    }
    store = create_store(
        problem_id="reference-high",
        directions=directions,
        policy=build_policy("exact"),
    )
    add_candidates(store, rows_from(points))

    expected = brute_force_layers(points, directions)
    actual = [sorted(layer) for layer in store.layers()]
    assert actual == expected

    # Every reported front must be internally non-dominated under all directions.
    vectors = {identifier: store.points[identifier].canonical for identifier in points}
    for front in actual:
        for left_id in front:
            for right_id in front:
                if left_id != right_id:
                    assert not dominates(vectors[left_id], vectors[right_id])


def test_repeated_insertion_and_deletion_always_matches_brute_force():
    directions = ["min", "min", "max"]
    sequence = [
        ("p1", (1, 1, 1)),
        ("p2", (0, 2, 0)),
        ("p3", (2, 0, 0)),
        ("p4", (1, 1, 1)),
        ("p5", (2, 2, 2)),
        ("p6", (0, 0, 0)),
    ]
    store = create_store(
        problem_id="reference-incremental",
        directions=directions,
        policy=build_policy("exact"),
    )
    current: dict[str, tuple[int, int, int]] = {}
    for identifier, values in sequence:
        current[identifier] = values
        add_candidates(store, [(identifier, list(values), None)])
        assert [sorted(layer) for layer in store.layers()] == brute_force_layers(
            current, directions
        )

    for identifier in ["p4", "p1", "p5"]:
        del current[identifier]
        delete_candidate(store, identifier)
        assert [sorted(layer) for layer in store.layers()] == brute_force_layers(
            current, directions
        )


def test_quantized_snapped_order_is_a_transitive_partial_order():
    policy = build_policy("quantized", "0.25")
    values = ["0", "0.12", "0.24", "0.25", "0.37", "0.49"]
    points = {
        f"{left_id}-{right_id}": (left, right)
        for left_id, left in enumerate(values)
        for right_id, right in enumerate(values)
        if left_id + right_id % 2 == 0
    }
    directions = ["min", "min"]
    store = create_store(
        problem_id="reference-grid",
        directions=directions,
        policy=policy,
    )
    add_candidates(store, rows_from(points))

    # Snap values manually in decimal coordinates and compare strictly exactly.
    from decimal import Decimal, ROUND_HALF_EVEN

    def snap(value):
        number = Decimal(value)
        return (number / Decimal("0.25")).quantize(
            Decimal("1"), rounding=ROUND_HALF_EVEN
        ) * Decimal("0.25")

    snapped = {
        identifier: tuple(-snap(value) for value in values)
        for identifier, values in points.items()
    }
    for layer in store.layers():
        for i, left_id in enumerate(layer):
            for right_id in layer[i + 1 :]:
                assert not dominates(snapped[left_id], snapped[right_id])
                assert not dominates(snapped[right_id], snapped[left_id])
