from __future__ import annotations

from pathlib import Path

import pytest

from mutation_backend.engine import run_mutation_analysis
from mutation_backend.isolation import source_snapshot
from mutation_backend.models import MutantFate, MutationOperator, MutationRequest


OPERATORS = [
    MutationOperator.COMPARISON,
    MutationOperator.ARITHMETIC,
    MutationOperator.CONDITIONAL_INVERT,
    MutationOperator.CONDITION_ERASE,
]


def _request(fixture_project: Path, test_command: list[str]) -> MutationRequest:
    return MutationRequest(
        request_id="acceptance-fixture-run",
        project_path=fixture_project,
        test_command=test_command,
        source_paths=["algorithm_pkg/**/*.py"],
        operators=OPERATORS,
        baseline_timeout_seconds=10,
        mutant_timeout_seconds=0.5,
    )


def _by_id(report):
    return {mutant.mutant_id: mutant for mutant in report.mutants}


def test_full_fixture_has_known_mutant_fates(tmp_path, fixture_project, test_command):
    before = source_snapshot(fixture_project)
    report = run_mutation_analysis(_request(fixture_project, test_command), work_root=tmp_path)
    after = source_snapshot(fixture_project)

    assert report.status == "completed"
    assert report.engine_version == "1.0.0"
    assert report.timeouts == {"baseline_seconds": 10.0, "mutant_seconds": 0.5}
    assert report.baseline["category"] == "passed"
    assert report.source_unchanged is True
    assert [item.model_dump() for item in before] == report.source_before["files"]
    assert [item.model_dump() for item in after] == report.source_after["files"]

    assert report.summary == {
        "pending": 0,
        "syntax_invalid": 5,
        "startup_failed": 4,
        "killed": 35,
        "survived": 17,
        "timeout": 3,
        "equivalent_undetermined": 1,
        "runner_error": 0,
    }

    mutants = _by_id(report)
    invalid = mutants["mut-dbfc247492e6"]
    assert invalid.fate == MutantFate.SYNTAX_INVALID
    assert invalid.failure_category == "syntax_error"
    assert invalid.location.path == "algorithm_pkg/pricing.py"
    assert invalid.location.line == 9

    startup = mutants["mut-96cb8f856c30"]
    assert startup.fate == MutantFate.STARTUP_FAILED
    assert startup.failure_category == "startup_failure"
    assert startup.exit_code == 2
    assert "collection" in startup.reason

    timeout_mutant = mutants["mut-4ddb5e0e4575"]
    assert timeout_mutant.fate == MutantFate.TIMEOUT
    assert timeout_mutant.failure_category == "timeout"
    assert timeout_mutant.exit_code is None
    assert timeout_mutant.duration_seconds >= 0.5

    equivalent = mutants["mut-ab4611496041"]
    assert equivalent.fate == MutantFate.EQUIVALENT_UNDETERMINED
    assert equivalent.determination == "undetermined"
    assert equivalent.equivalent_basis is not None
    assert equivalent.equivalent_basis.endswith("3")

    killed = mutants["mut-20ad0d9b5c95"]
    assert killed.fate == MutantFate.KILLED
    assert killed.failure_category == "test_failure"
    assert killed.exit_code == 1

    assert not any(Path(mutant.workspace).exists() for mutant in report.mutants if mutant.workspace)
    assert not (tmp_path / report.request_id / "baseline").exists()


def test_repeated_runs_are_deterministic_and_isolated(tmp_path, fixture_project, test_command):
    first = run_mutation_analysis(_request(fixture_project, test_command), work_root=tmp_path / "one")
    second = run_mutation_analysis(_request(fixture_project, test_command), work_root=tmp_path / "two")

    assert first.summary == second.summary
    assert [(m.mutant_id, m.fate, m.location.path, m.location.line, m.before, m.after) for m in first.mutants] == [
        (m.mutant_id, m.fate, m.location.path, m.location.line, m.before, m.after) for m in second.mutants
    ]
    assert first.source_before == first.source_after


def test_baseline_failure_is_separate_from_mutant_timeout(tmp_path, fixture_project, test_command):
    request = _request(fixture_project, test_command)
    request = request.model_copy(
        update={"test_command": ["python3", "-c", "raise SystemExit(2)"], "baseline_timeout_seconds": 1}
    )
    report = run_mutation_analysis(request, work_root=tmp_path)

    assert report.status == "baseline_failed"
    assert report.mutants == []
    assert report.baseline["category"] == "startup_failure"
    baseline_event = next(event for event in report.events if event.step == "baseline")
    assert baseline_event.status == "failed"


def test_baseline_timeout_is_separate_from_mutant_timeout(tmp_path, fixture_project, test_command):
    request = _request(fixture_project, test_command)
    request = request.model_copy(
        update={
            "test_command": ["python3", "-c", "import time; time.sleep(1)"],
            "baseline_timeout_seconds": 0.2,
            "mutant_timeout_seconds": 0.9,
        }
    )
    report = run_mutation_analysis(request, work_root=tmp_path)

    assert report.status == "baseline_timeout"
    assert report.mutants == []
    assert report.baseline["timed_out"] is True
    assert report.baseline["category"] == "timeout"
    assert report.timeouts == {"baseline_seconds": 0.2, "mutant_seconds": 0.9}
    assert any(event.step == "baseline" and event.status == "timeout" for event in report.events)


def test_workspace_cannot_live_inside_mutated_project(fixture_project, test_command):
    request = _request(fixture_project, test_command)
    with pytest.raises(ValueError, match="outside project_path"):
        run_mutation_analysis(request, work_root=fixture_project / "nested-work")
