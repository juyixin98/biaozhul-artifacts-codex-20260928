from __future__ import annotations

from app.survival.experiments import run_reproduction
from app.survival.references import reference_library


def test_reference_library_is_available():
    library = reference_library()
    assert library.name == "lifelines"
    assert library.version == "0.30.3"


def test_all_reproduction_checks_pass_against_reference():
    report = run_reproduction()
    assert report.reference_library == {"name": "lifelines", "version": "0.30.3"}
    assert len(report.checks) == 7
    failures = [
        (check.scenario, check.analysis, check.failures)
        for check in report.checks
        if not check.passed
    ]
    assert failures == []


def test_reproduction_report_declares_statistical_scope():
    report = run_reproduction()
    assert report.fixture_version == "fixtures-v1"
    assert report.estimator_versions
    assert "not intended" in report.disclaimer or "not" in report.disclaimer.lower()
