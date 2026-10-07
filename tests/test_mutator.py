from __future__ import annotations

import pytest

from mutation_backend.models import MutationOperator
from mutation_backend.mutator import generate_mutations


def test_comparison_mutation_has_source_mapping(fixture_project):
    mutations = generate_mutations(
        fixture_project,
        ["algorithm_pkg/search.py"],
        [],
        [MutationOperator.COMPARISON],
    )

    loop_mutations = [m for m in mutations if m.location.line == 6 and m.before == "<"]
    assert {(m.after, m.location.column) for m in loop_mutations} == {
        ("<=", 10),
        (">", 10),
        (">=", 10),
        ("==", 10),
        ("!=", 10),
    }
    assert len({m.mutant_id for m in loop_mutations}) == 5
    assert all(m.location.node_type == "Compare" for m in loop_mutations)


def test_arithmetic_and_condition_inversion_are_ast_based(fixture_project):
    mutations = generate_mutations(
        fixture_project,
        ["algorithm_pkg/pricing.py"],
        [],
        [MutationOperator.ARITHMETIC, MutationOperator.CONDITIONAL_INVERT],
    )

    subtotal_multiply = [
        m
        for m in mutations
        if m.operator_family == MutationOperator.ARITHMETIC and m.location.line == 8
    ]
    assert len(subtotal_multiply) == 5
    assert sorted(m.after for m in subtotal_multiply) == ["%", "+", "-", "/", "//"]

    inverted = [m for m in mutations if m.operator_family == MutationOperator.CONDITIONAL_INVERT]
    assert len(inverted) == 1
    assert inverted[0].before == "subtotal > 100"
    assert inverted[0].after == "subtotal <= 100"


def test_condition_erase_is_explicitly_syntax_invalid_after_unparse(fixture_project):
    mutations = generate_mutations(
        fixture_project,
        ["algorithm_pkg/shapes.py"],
        [],
        [MutationOperator.CONDITION_ERASE],
    )

    assert len(mutations) == 2
    for mutation in mutations:
        compile(mutation.original_source, "<original>", "exec")
        with pytest.raises(SyntaxError, match="invalid syntax"):
            compile(mutation.after, "<fragment>", "exec")
        with pytest.raises(SyntaxError, match="invalid syntax"):
            compile(mutation.mutated_source, "<mutated>", "exec")
