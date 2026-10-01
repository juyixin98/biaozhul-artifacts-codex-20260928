import pytest

from core.contracts import AnalysisRequest
from core.errors import ProbabilityRangeError, SampleContractError, WeightError


def test_probability_above_one_is_rejected_not_clipped(log_dir):
    with pytest.raises(ProbabilityRangeError) as excinfo:
        AnalysisRequest(y=[0], p=[1.000001])

    assert excinfo.value.category == "probability_range_error"
    assert "1.000001" in str(excinfo.value)


def test_probability_below_zero_is_rejected_with_failure_category():
    with pytest.raises(ProbabilityRangeError) as excinfo:
        AnalysisRequest(y=[0], p=[-0.01])

    assert excinfo.value.category == "probability_range_error"


def test_mismatched_lengths_rejected():
    with pytest.raises(SampleContractError, match="same number"):
        AnalysisRequest(y=[0, 1], p=[0.2], w=[1, 1])


def test_negative_weight_rejected():
    with pytest.raises(WeightError) as excinfo:
        AnalysisRequest(y=[0, 1], p=[0.2, 0.4], w=[1, -0.5])

    assert excinfo.value.category == "weight_error"


def test_all_zero_weights_rejected():
    with pytest.raises(WeightError, match="total sample weight"):
        AnalysisRequest(y=[0, 1], p=[0.2, 0.4], w=[0.0, 0.0])


def test_fixed_edges_must_span_closed_probability_domain():
    with pytest.raises(SampleContractError, match="start at 0.0"):
        AnalysisRequest(y=[0], p=[0.5], edges=[0.1, 1.0])


def test_fixed_edges_must_be_strictly_increasing():
    with pytest.raises(SampleContractError, match="strictly increasing"):
        AnalysisRequest(y=[0, 1], p=[0.2, 0.2], edges=[0.0, 0.2, 0.2, 1.0])
