"""Unit tests for the AST mutation engine."""
import ast

import pytest

from app.models import MutationOperator
from app.mutator import (
    MutationPoint,
    _unreachable_lines,
    apply_mutation,
    find_mutation_points,
)

ALL_OPS = set(MutationOperator)


def test_comparison_points():
    pts = find_mutation_points("m.py", "def f(a, b):\n    return a < b\n",
                               ALL_OPS)
    mutated = sorted(p.mutated_segment for p in pts)
    assert mutated == ["a <= b", "a > b"]


def test_arithmetic_points():
    pts = find_mutation_points("m.py", "def f(a, b):\n    return a + b\n",
                               ALL_OPS)
    assert len(pts) == 1
    assert pts[0].mutated_segment == "a - b"
    assert pts[0].operator is MutationOperator.ARITHMETIC


def test_condition_negation_point():
    src = "def f(x):\n    if x > 1:\n        return 1\n    return 0\n"
    pts = find_mutation_points("m.py", src,
                               {MutationOperator.CONDITION_NEGATION})
    assert len(pts) == 1
    assert pts[0].mutated_segment == "not x > 1"


def test_apply_mutation_roundtrip():
    src = "def f(a, b):\n    return a + b\n"
    pts = find_mutation_points("m.py", src, ALL_OPS)
    mutated = apply_mutation(src, pts[0])
    assert "a - b" in mutated
    assert "a + b" not in mutated


def test_apply_mutation_condition():
    src = "def f(x):\n    while x > 0:\n        x = x - 1\n    return x\n"
    pts = [p for p in find_mutation_points("m.py", src, ALL_OPS)
           if p.operator is MutationOperator.CONDITION_NEGATION]
    assert len(pts) == 1
    mutated = apply_mutation(src, pts[0])
    assert "while not x > 0" in mutated


def test_unreachable_detection():
    src = (
        "def f(x):\n"
        "    return x\n"
        "    y = x + 1\n"
        "    return y\n"
    )
    tree = ast.parse(src)
    assert _unreachable_lines(tree) == {3, 4}


def test_equivalent_points_flagged():
    src = (
        "def f(x):\n"
        "    return x\n"
        "    y = x + 1\n"
    )
    pts = find_mutation_points("m.py", src, ALL_OPS)
    assert len(pts) == 1
    assert pts[0].equivalent is True


def test_apply_mutation_propagates_syntax_error():
    src = "def f(a, b):\n    return a + b\n"
    real = find_mutation_points("m.py", src, ALL_OPS)[0]
    point = MutationPoint(
        file_path="m.py", node_index=real.node_index,
        operator=MutationOperator.ARITHMETIC,
        line=2, col=11, original_segment="a + b",
        mutated_segment="a +", equivalent=False,
    )
    with pytest.raises(SyntaxError):
        apply_mutation(src, point)


def test_apply_mutation_index_mismatch():
    point = MutationPoint(
        file_path="m.py", node_index=999,
        operator=MutationOperator.ARITHMETIC,
        line=1, col=0, original_segment="a + b",
        mutated_segment="a - b", equivalent=False,
    )
    with pytest.raises(ValueError):
        apply_mutation("def f(a, b):\n    return a + b\n", point)
