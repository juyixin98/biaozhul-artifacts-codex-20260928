from __future__ import annotations

from mutation_backend.models import MutationOperator
from mutation_backend.mutator import generate_mutations
from mutation_backend.equivalence import equivalence_basis


def test_constant_folded_identity_is_undetermined_equivalence(fixture_project):
    mutations = generate_mutations(
        fixture_project,
        ["algorithm_pkg/pricing.py"],
        [],
        [MutationOperator.ARITHMETIC],
    )

    equivalent = [m for m in mutations if m.after == "//" and m.location.line == 4]
    assert len(equivalent) == 1
    basis = equivalence_basis(equivalent[0])
    assert basis is not None
    assert "identical value" in basis
    assert basis.endswith("3")


def test_runtime_dependent_arithmetic_is_not_called_equivalent(fixture_project):
    mutations = generate_mutations(
        fixture_project,
        ["algorithm_pkg/pricing.py"],
        [],
        [MutationOperator.ARITHMETIC],
    )

    runtime_mutation = [m for m in mutations if m.location.line == 15]
    assert runtime_mutation
    assert all(equivalence_basis(m) is None for m in runtime_mutation)
