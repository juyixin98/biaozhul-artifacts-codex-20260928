"""Unit tests for failure classification: target / other / pass /
indeterminate must be distinguished, with reasons."""
from __future__ import annotations

from ddmin_service.classifier import ExecutionOutcome, classify
from ddmin_service.models import ClassifierConfig, Verdict

CFG = ClassifierConfig(target_exit_codes=[3], pass_exit_codes=[0], timeout_seconds=1.0)


def outcome(code, stderr="", timed_out=False):
    return ExecutionOutcome(exit_code=code, stderr=stderr, timed_out=timed_out,
                            duration_ms=1.0)


def test_target_exit_code_is_target():
    result = classify(outcome(3), CFG)
    assert result.verdict == Verdict.TARGET
    assert "target" in result.reason


def test_pass_exit_code_is_pass():
    assert classify(outcome(0), CFG).verdict == Verdict.PASS


def test_other_exit_code_is_other():
    result = classify(outcome(2), CFG)
    assert result.verdict == Verdict.OTHER
    assert "neither" in result.reason


def test_timeout_is_indeterminate():
    result = classify(outcome(None, timed_out=True), CFG)
    assert result.verdict == Verdict.INDETERMINATE
    assert "cannot determine" in result.reason


def test_stderr_pattern_must_match_for_target():
    cfg = ClassifierConfig(target_exit_codes=[3], target_stderr_pattern="accumulator")
    assert classify(outcome(3, "FATAL: accumulator overflow"), cfg).verdict == Verdict.TARGET
    wrong = classify(outcome(3, "some other crash"), cfg)
    assert wrong.verdict == Verdict.OTHER
    assert "did not match" in wrong.reason
