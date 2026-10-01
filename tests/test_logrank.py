import pytest

from survival_core import Observation, log_rank

from ._fixtures import all_censored_group, comparison_group_b, simple_group


def test_two_group_logrank_matches_independent_hand_arithmetic():
    observations = simple_group() + comparison_group_b()
    result = log_rank(observations, ["A", "B"])

    assert result.status == "estimated"
    assert result.degrees_freedom == 1
    assert result.observed == {"A": 2, "B": 2}
    assert result.expected == pytest.approx(
        {"A": 2.630952380952381, "B": 1.369047619047619}
    )
    assert result.variance_covariance_matrix == [[pytest.approx(0.8638038548752834)]]
    assert result.chi_square == pytest.approx(0.4608695652173912)
    assert result.p_value == pytest.approx(0.49721786092780695)


def test_delayed_entry_logrank_uses_strict_entry_rule_and_tied_events():
    observations = [
        Observation("A1", "A", 4.0, True, entry=1.0),
        Observation("A2", "A", 4.0, False, entry=2.0),
        Observation("A3", "A", 3.0, True, entry=0.0),
        Observation("B1", "B", 4.0, True, entry=0.0),
    ]
    result = log_rank(observations, ["A", "B"])
    assert result.status == "estimated"
    assert result.observed == {"A": 2, "B": 1}
    assert result.expected == pytest.approx({"A": 2.083333333333333, "B": 0.9166666666666666})
    assert result.variance_covariance_matrix == [[pytest.approx(59 / 144)]]
    assert result.chi_square == pytest.approx(0.016949152542372882)
    assert result.p_value == pytest.approx(0.8964169676673137)


def test_no_events_in_any_group_is_not_run_without_p_value():
    observations = all_censored_group() + [
        Observation("E1", "E", 2.0, False),
        Observation("E2", "E", 5.0, False),
    ]
    result = log_rank(observations, ["C", "E"])
    assert result.status == "not_run"
    assert result.reason == "no_events_in_any_group"
    assert result.chi_square is None
    assert result.p_value is None
    assert result.expected == {"C": 0.0, "E": 0.0}


def test_single_group_comparison_is_not_run():
    result = log_rank(simple_group(), ["A"])
    assert result.status == "not_run"
    assert result.reason == "log_rank_requires_at_least_two_non_empty_groups"
    assert result.expected == {"A": 2.0}
    assert result.p_value is None


def test_declared_empty_group_is_failed_and_keeps_empty_label():
    observations = simple_group() + comparison_group_b()
    result = log_rank(observations, ["A", "B", "EMPTY"])
    assert result.status == "failed"
    assert result.reason == "empty_group_excluded_from_log_rank"
    assert result.group_order == ["A", "B", "EMPTY"]
    assert result.observed == {"A": 2, "B": 2, "EMPTY": 0}
    assert result.expected == {"A": None, "B": None, "EMPTY": None}
    assert result.chi_square is None
    assert result.p_value is None


def test_invalid_alpha_is_value_error():
    with pytest.raises(ValueError, match="significance_level"):
        log_rank(simple_group(), ["A"], significance_level=0)
