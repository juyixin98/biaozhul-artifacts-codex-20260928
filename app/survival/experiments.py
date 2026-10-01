"""Reproducible experiments comparing the local core with hand/reference output."""

from __future__ import annotations

from math import isclose
from typing import Any

from pydantic import BaseModel, Field

from .fixtures import FIXTURE_GROUPS, get_fixture
from .km import kaplan_meier
from .logrank import logrank_test
from .references import (
    kaplan_meier_reference,
    logrank_two_group_reference,
    reference_library,
)
from ..versions import DISCLAIMER, KM_ESTIMATOR_VERSION, LOGRANK_VERSION, SYNTHETIC_FIXTURE_VERSION


KM_SCENARIOS = [
    "hand_km",
    "left_truncation",
    "tied_events",
    "heavy_censoring",
    "all_censored",
]
LOGRANK_SCENARIOS = ["logrank_hand", "logrank_no_events"]


class ComparisonCheck(BaseModel):
    scenario: str
    analysis: str
    passed: bool
    max_abs_difference: float | None = None
    failures: list[str] = Field(default_factory=list)


class ReproductionReport(BaseModel):
    fixture_version: str
    estimator_versions: list[str]
    reference_library: dict[str, str] | None = None
    checks: list[ComparisonCheck]
    disclaimer: str = DISCLAIMER


def _find_row(rows: list[dict[str, Any]], time: float) -> dict[str, Any] | None:
    return next((row for row in rows if isclose(row["time"], time, abs_tol=1e-10)), None)


def _compare_km_scenario(name: str) -> ComparisonCheck:
    records = get_fixture(name)
    group = records[0].group
    core = kaplan_meier(records, groups=[group], include_baseline=True)
    reference = kaplan_meier_reference(records, group)
    failures: list[str] = []
    max_difference = 0.0

    reference_rows = reference["rows"]
    core_rows = core.groups[0].timeline
    if len(core_rows) != len(reference_rows):
        failures.append(
            f"row count {len(core_rows)} does not match reference {len(reference_rows)}"
        )

    for core_row in core_rows:
        ref_row = _find_row(reference_rows, core_row.time)
        if ref_row is None:
            failures.append(f"reference missing time {core_row.time:g}")
            continue
        for key in ("entered", "at_risk", "events", "censored"):
            if getattr(core_row, key) != ref_row[key]:
                failures.append(
                    f"t={core_row.time:g} {key} core={getattr(core_row, key)} "
                    f"reference={ref_row[key]}"
                )
        if core_row.survival is not None:
            max_difference = max(max_difference, abs(core_row.survival - ref_row["survival"]))
            if not isclose(core_row.survival, ref_row["survival"], abs_tol=1e-10):
                failures.append(
                    f"t={core_row.time:g} survival core={core_row.survival:.12f} "
                    f"reference={ref_row['survival']:.12f}"
                )
        for key, ref_key in (("ci_lower", "ci_lower"), ("ci_upper", "ci_upper")):
            core_value = getattr(core_row, key)
            if core_value is not None:
                max_difference = max(max_difference, abs(core_value - ref_row[ref_key]))
                if not isclose(core_value, ref_row[ref_key], abs_tol=1e-9):
                    failures.append(
                        f"t={core_row.time:g} {key} core={core_value:.12f} "
                        f"reference={ref_row[ref_key]:.12f}"
                    )

    return ComparisonCheck(
        scenario=name,
        analysis="kaplan-meier",
        passed=not failures,
        max_abs_difference=max_difference,
        failures=failures,
    )


def _compare_logrank_scenario(name: str) -> ComparisonCheck:
    records = get_fixture(name)
    core = logrank_test(records)
    failures: list[str] = []
    max_difference: float | None = None

    if name == "logrank_no_events":
        if core.testable or core.p_value is not None:
            failures.append("all-censored data must not produce a p-value")
        return ComparisonCheck(
            scenario=name,
            analysis="log-rank",
            passed=not failures,
            max_abs_difference=max_difference,
            failures=failures,
        )

    reference = logrank_two_group_reference(records, core.groups)
    max_difference = max(
        abs(core.chi_square - reference["chi_square"]),
        abs((core.p_value or 0.0) - reference["p_value"]),
    )
    if not isclose(core.chi_square, reference["chi_square"], rel_tol=1e-10, abs_tol=1e-10):
        failures.append(
            f"chi-square core={core.chi_square:.12f} reference={reference['chi_square']:.12f}"
        )
    if core.p_value is None or not isclose(
        core.p_value, reference["p_value"], rel_tol=1e-10, abs_tol=1e-10
    ):
        failures.append(f"p-value core={core.p_value} reference={reference['p_value']:.12f}")

    return ComparisonCheck(
        scenario=name,
        analysis="log-rank",
        passed=not failures,
        max_abs_difference=max_difference,
        failures=failures,
    )


def run_reproduction() -> ReproductionReport:
    checks = [_compare_km_scenario(name) for name in KM_SCENARIOS]
    checks.extend(_compare_logrank_scenario(name) for name in LOGRANK_SCENARIOS)
    library = reference_library()
    return ReproductionReport(
        fixture_version=SYNTHETIC_FIXTURE_VERSION,
        estimator_versions=[KM_ESTIMATOR_VERSION, LOGRANK_VERSION],
        reference_library={"name": library.name, "version": library.version},
        checks=checks,
    )


def declared_groups_for_fixture(name: str) -> list[str] | None:
    return FIXTURE_GROUPS.get(name)
