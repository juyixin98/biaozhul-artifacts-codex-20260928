"""Failure classification: maps raw execution outcomes to verdicts.

Every classification carries a human-readable reason so diagnostics can
explain why a candidate was accepted, rejected or could not be decided.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ClassifierConfig, Verdict


@dataclass(frozen=True)
class ExecutionOutcome:
    exit_code: int | None  # None when the run did not terminate
    stderr: str
    timed_out: bool
    duration_ms: float


@dataclass(frozen=True)
class Classification:
    verdict: Verdict
    reason: str


def classify(outcome: ExecutionOutcome, config: ClassifierConfig) -> Classification:
    if outcome.timed_out or outcome.exit_code is None:
        return Classification(
            Verdict.INDETERMINATE,
            f"program did not finish within {config.timeout_seconds}s; cannot determine",
        )
    code = outcome.exit_code
    if code in config.target_exit_codes:
        if config.target_stderr_pattern:
            if re.search(config.target_stderr_pattern, outcome.stderr):
                return Classification(
                    Verdict.TARGET,
                    f"exit {code} is a target code and stderr matched "
                    f"{config.target_stderr_pattern!r}",
                )
            return Classification(
                Verdict.OTHER,
                f"exit {code} is a target code but stderr did not match "
                f"{config.target_stderr_pattern!r}; treating as other error",
            )
        return Classification(Verdict.TARGET, f"exit {code} is a configured target error code")
    if code in config.pass_exit_codes:
        return Classification(Verdict.PASS, f"exit {code} is a success code")
    return Classification(Verdict.OTHER, f"exit {code} is neither a target nor a success code")
