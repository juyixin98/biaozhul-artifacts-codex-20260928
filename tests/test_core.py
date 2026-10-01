from __future__ import annotations

import itertools
import math
import threading

import numpy as np
import pytest

from permutation_service.core import PermutationRequest, run_permutation_test
from permutation_service.errors import (
    CancellationError,
    ExactLimitExceededError,
    InvalidInputError,
)


def reference_mean_abs_difference(values, group1_indices, group1_size):
    group1 = [values[i] for i in group1_indices]
    group2 = [
        values[i]
        for i in range(len(values))
        if i not in set(group1_indices)
    ]
    return abs(sum(group1) / group1_size - sum(group2) / (len(values) - group1_size))


def reference_exact(values, group1_size):
    statistics = [
        reference_mean_abs_difference(values, indices, group1_size)
        for indices in itertools.combinations(range(len(values)), group1_size)
    ]
    observed = reference_mean_abs_difference(
        values, range(group1_size), group1_size
    )
    return observed, statistics, sum(stat >= observed for stat in statistics)


def test_small_hand_computed_full_permutation_distribution():
    result = run_permutation_test(
        PermutationRequest(
            values=[1.0, 2.0, 3.0],
            group1_size=1,
            method="exact",
            exact_limit_cap=10,
            return_null_distribution=True,
        )
    )

    assert result.method_used == "exact"
    assert result.statistic == pytest.approx(1.5)
    assert result.total_possible_assignments == 3
    assert result.evaluated_permutations == 3
    assert result.extreme_permutations == 2
    assert result.p_value == pytest.approx(2 / 3)
    assert tuple(sorted(result.null_distribution)) == (0.0, 1.5, 1.5)
    assert result.significant is False


def test_exact_distribution_matches_independent_reference():
    values = [0.5, -1.25, 2.0, 3.5, -0.25, 1.75]
    observed, reference_distribution, reference_extreme = reference_exact(values, 2)

    result = run_permutation_test(
        PermutationRequest(
            values=values,
            group1_size=2,
            method="exact",
            exact_limit_cap=20,
            return_null_distribution=True,
        )
    )

    assert result.total_possible_assignments == math.comb(6, 2)
    assert result.statistic == pytest.approx(observed)
    assert tuple(sorted(result.null_distribution)) == tuple(
        sorted(reference_distribution)
    )
    assert result.extreme_permutations == reference_extreme
    assert result.p_value == pytest.approx(
        reference_extreme / math.comb(6, 2)
    )


def test_label_swap_symmetry():
    original = run_permutation_test(
        PermutationRequest(
            values=[1.0, 2.0, 3.0, 4.0],
            group1_size=2,
            method="exact",
            exact_limit_cap=10,
        )
    )
    swapped = run_permutation_test(
        PermutationRequest(
            values=[3.0, 4.0, 1.0, 2.0],
            group1_size=2,
            method="exact",
            exact_limit_cap=10,
        )
    )

    assert swapped.statistic == pytest.approx(original.statistic)
    assert swapped.p_value == pytest.approx(original.p_value)
    assert swapped.total_possible_assignments == original.total_possible_assignments


def test_equal_values_do_not_create_artificial_significance():
    result = run_permutation_test(
        PermutationRequest(
            values=[7.0, 7.0, 7.0, 7.0],
            group1_size=2,
            method="exact",
            exact_limit_cap=10,
        )
    )

    assert result.statistic == 0.0
    assert result.evaluated_permutations == 6
    assert result.extreme_permutations == 6
    assert result.p_value == 1.0
    assert result.significant is False


def test_repeated_values_still_count_as_distinct_sample_identities():
    result = run_permutation_test(
        PermutationRequest(
            values=[1.0, 1.0, 2.0, 2.0],
            group1_size=2,
            method="exact",
            exact_limit_cap=10,
            return_null_distribution=True,
        )
    )

    assert result.evaluated_permutations == 6
    assert result.extreme_permutations == 2
    assert result.p_value == pytest.approx(1 / 3)
    # Values repeat, but the four identity assignments with one value from
    # each measured level remain separate zero-statistic assignments.
    assert sorted(result.null_distribution) == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_extreme_unbalanced_groups_exact_p_value():
    values = [100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    result = run_permutation_test(
        PermutationRequest(
            values=values,
            group1_size=1,
            method="exact",
            exact_limit_cap=10,
        )
    )

    assert result.total_possible_assignments == 7
    assert result.extreme_permutations == 1
    assert result.p_value == pytest.approx(1 / 7)
    assert result.statistic == pytest.approx(99.0)


def test_auto_uses_monte_carlo_when_combinations_exceed_threshold():
    result = run_permutation_test(
        PermutationRequest(
            values=list(range(10)),
            group1_size=5,
            method="auto",
            exact_max_combinations=1,
            monte_carlo_replications=99,
            random_seed=123,
        )
    )

    assert result.method_used == "monte_carlo"
    assert result.total_possible_assignments == math.comb(10, 5)
    assert result.evaluated_permutations == 99
    assert result.random_seed == 123
    assert result.finite_sample_correction == "plus-one:(extreme+1)/(B+1)"


def test_random_seed_replays_same_distribution():
    request_arguments = dict(
        values=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        group1_size=5,
        method="monte_carlo",
        monte_carlo_replications=101,
        random_seed=20260930,
        return_null_distribution=True,
    )
    first = run_permutation_test(PermutationRequest(**request_arguments))
    second = run_permutation_test(PermutationRequest(**request_arguments))

    assert second.null_distribution == first.null_distribution
    assert second.p_value == first.p_value
    assert second.extreme_permutations == first.extreme_permutations
    assert second.random_seed == 20260930


def test_monte_carlo_plus_one_for_unique_extreme_observation():
    result = run_permutation_test(
        PermutationRequest(
            values=[10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            group1_size=1,
            method="monte_carlo",
            monte_carlo_replications=99,
            random_seed=42,
            return_null_distribution=True,
        )
    )

    assert result.method_used == "monte_carlo"
    assert result.total_possible_assignments == 6
    assert result.evaluated_permutations == 99
    assert result.extreme_permutations == 1
    assert result.p_value == pytest.approx(1 / 100)
    assert result.standard_error == pytest.approx(
        math.sqrt((1 / 100) * (99 / 100) / 100)
    )
    assert result.confidence_interval_95[0] >= 0.0
    assert result.confidence_interval_95[1] <= 1.0


def test_monte_carlo_preserves_group_sizes():
    observed_group_sizes: list[int] = []
    original_default_rng = np.random.default_rng

    class RecordingGenerator(np.random.Generator):
        def permutation(self, x, axis=0):
            shuffled = super().permutation(x, axis=axis)
            observed_group_sizes.extend(int(row.sum()) for row in shuffled)
            return shuffled

    def recording_default_rng(seed=None):
        return RecordingGenerator(np.random.PCG64(seed))

    np.random.default_rng = recording_default_rng
    try:
        result = run_permutation_test(
            PermutationRequest(
                values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            group1_size=2,
            method="monte_carlo",
            monte_carlo_replications=50,
            random_seed=7,
                return_null_distribution=True,
            )
        )
    finally:
        np.random.default_rng = original_default_rng

    assert result.evaluated_permutations == 50
    assert len(observed_group_sizes) >= 50
    assert set(observed_group_sizes) == {2}
    assert len(result.null_distribution) == 50
    assert all(math.isfinite(value) for value in result.null_distribution)


def test_cancelation_is_not_reported_as_not_significant():
    canceled = threading.Event()

    def progress(completed, total, phase, batch_size):
        if completed > 0:
            canceled.set()

    with pytest.raises(CancellationError) as exc_info:
        run_permutation_test(
            PermutationRequest(
                values=list(range(28)),
                group1_size=14,
                method="exact",
                exact_limit_cap=50_000_000,
            ),
            progress=progress,
            is_canceled=canceled.is_set,
        )

    assert exc_info.value.error_code == "CANCELED"



def test_invalid_inputs_have_specific_categories():
    cases = [
        (dict(values=[1.0], group1_size=1, method="exact"), "at least two"),
        (
            dict(values=[1.0, 2.0], group1_size=2, method="exact"),
            "group1_size",
        ),
        (
            dict(values=[float("nan"), 2.0], group1_size=1, method="exact"),
            "finite",
        ),
        (
            dict(values=[float("inf"), 2.0], group1_size=1, method="exact"),
            "finite",
        ),
        (
            dict(
                values=[1.0, 2.0, 3.0],
                group1_size=1,
                alpha=1.0,
                method="exact",
            ),
            "alpha",
        ),
    ]
    for arguments, fragment in cases:
        with pytest.raises(InvalidInputError) as exc_info:
            run_permutation_test(PermutationRequest(**arguments))
        assert fragment in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_INPUT"


def test_explicit_exact_enforcement_rejects_combination_cap():
    with pytest.raises(ExactLimitExceededError) as exc_info:
        run_permutation_test(
            PermutationRequest(
                values=list(range(12)),
                group1_size=6,
                method="exact",
                exact_limit_cap=10,
            )
        )

    assert exc_info.value.error_code == "EXACT_LIMIT_EXCEEDED"
    assert exc_info.value.details["total_possible_assignments"] == math.comb(12, 6)


def test_numpy_array_inputs_are_supported_by_reference_sanity_check():
    observed, distribution, extreme = reference_exact([1.0, 2.0, 3.0], 1)
    assert observed == 1.5
    assert sorted(distribution) == [0.0, 1.5, 1.5]
    assert extreme == 2
    assert np.isfinite(observed)
