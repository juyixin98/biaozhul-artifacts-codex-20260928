"""Integration tests: known mutant fates on the bundled fixture package.

The reference table below is hand-derived from fixtures/sample_pkg/algos.py
and its test suite; it is NOT produced by the mutation engine itself.
"""
from pathlib import Path

import pytest

from app.config import Settings
from app.models import MutationOperator, RunRequest, RunStatus
from app.runner import MutationRunner, source_digest
from app.store import EvidenceStore

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

# Hand-written reference fates for fixtures/sample_pkg, keyed by
# (original_segment, mutated_segment).
EXPECTED_FATES = {
    ("x < lo", "not x < lo"): "killed",
    ("x < lo", "x <= lo"): "survived",   # x == lo returns lo == x either way
    ("x < lo", "x > lo"): "killed",
    ("x > 0", "x >= 0"): "killed",       # boundary 0 is tested
    ("x > 0", "x < 0"): "killed",
    ("total + n", "total - n"): "killed",
    ("x * x", "x / x"): "killed",
    ("age >= 18", "not age >= 18"): "killed",
    ("age >= 18", "age > 18"): "survived",  # 18 itself is never tested
    ("age >= 18", "age <= 18"): "killed",
    ("count + 1", "count - 1"): "killed",
    ("i < len(nums)", "not i < len(nums)"): "killed",
    ("i < len(nums)", "i <= len(nums)"): "killed",  # IndexError off-by-one
    ("i < len(nums)", "i > len(nums)"): "killed",
    ("nums[i] < 0", "not nums[i] < 0"): "killed",
    ("nums[i] < 0", "nums[i] <= 0"): "killed",  # zero element is tested
    ("nums[i] < 0", "nums[i] > 0"): "killed",
    ("i + 1", "i - 1"): "killed",
    # Unreachable statements in dead_code_demo: statically equivalent.
    ("x + 1", "x - 1"): "undetermined",
    ("y > 10", "not y > 10"): "undetermined",
    ("y > 10", "y >= 10"): "undetermined",
    ("y > 10", "y < 10"): "undetermined",
}


def _make_runner(tmp_path, **overrides):
    settings = Settings(db_path=str(tmp_path / "evidence.db"), **overrides)
    store = EvidenceStore(settings.db_path)
    return MutationRunner(settings, store), store


def _request(src, timeout=15.0):
    return RunRequest(
        source_dir=str(src), test_dir="tests",
        timeout_seconds=timeout, requester="pytest-suite",
        operators=list(MutationOperator),
    )


@pytest.fixture(scope="module")
def sample_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sample_run")
    runner, store = _make_runner(tmp)
    src = FIXTURES / "sample_pkg"
    digest_before = source_digest(src)
    summary = runner.run(_request(src))
    mutants = store.list_mutants(summary.run_id)
    return summary, mutants, digest_before, source_digest(src)


def test_run_completed_with_expected_counts(sample_run):
    summary, mutants, _, _ = sample_run
    assert summary.status is RunStatus.COMPLETED
    assert summary.total == 22
    assert len(mutants) == 22
    assert summary.killed == 16
    assert summary.survived == 2
    assert summary.undetermined == 4
    assert summary.timeout == 0
    assert summary.invalid_syntax == 0
    assert summary.startup_failure == 0


def test_every_mutant_fate_matches_reference(sample_run):
    _, mutants, _, _ = sample_run
    seen = {}
    for m in mutants:
        key = (m.original_segment, m.mutated_segment)
        assert key in EXPECTED_FATES, f"unexpected mutant {key}"
        seen[key] = m.outcome.value
    assert seen == EXPECTED_FATES
    assert len(seen) == len(EXPECTED_FATES)


def test_source_mapping_fields_present(sample_run):
    _, mutants, _, _ = sample_run
    for m in mutants:
        assert m.file_path == "algos.py"
        assert m.line > 0
        assert m.col >= 0
        assert m.operator in {
            "comparison", "arithmetic", "condition_negation"
        }
        assert m.detail  # every fate carries an explanation


def test_original_tree_unchanged(sample_run):
    _, _, digest_before, digest_after = sample_run
    assert digest_before == digest_after


def test_repeatable_results(tmp_path):
    src = FIXTURES / "sample_pkg"
    fates_per_run = []
    for i in range(2):
        rep_dir = tmp_path / f"rep{i}"
        rep_dir.mkdir()
        runner, store = _make_runner(rep_dir)
        summary = runner.run(_request(src))
        fates_per_run.append(
            {(m.original_segment, m.mutated_segment): m.outcome.value
             for m in store.list_mutants(summary.run_id)}
        )
    assert fates_per_run[0] == fates_per_run[1] == EXPECTED_FATES


def test_timeout_distinct_from_baseline(tmp_path):
    runner, store = _make_runner(tmp_path)
    summary = runner.run(_request(FIXTURES / "timeout_pkg", timeout=2.0))
    assert summary.status is RunStatus.COMPLETED
    assert summary.timeout == 1          # n = n + 1 spins forever
    assert summary.killed == 3
    mutants = store.list_mutants(summary.run_id)
    timed_out = [m for m in mutants if m.outcome.value == "timeout"]
    assert timed_out[0].mutated_segment == "n + 1"
    assert "timeout" in timed_out[0].detail


def test_startup_failure_classification(tmp_path):
    runner, store = _make_runner(tmp_path)
    summary = runner.run(_request(FIXTURES / "startup_pkg"))
    assert summary.status is RunStatus.COMPLETED
    assert summary.startup_failure == 1
    mutants = store.list_mutants(summary.run_id)
    assert mutants[0].mutated_segment == "10 / 2"
    assert mutants[0].outcome.value == "startup_failure"


def test_baseline_failed_status(tmp_path):
    runner, _ = _make_runner(tmp_path)
    summary = runner.run(_request(FIXTURES / "baseline_fail_pkg"))
    assert summary.status is RunStatus.BASELINE_FAILED
    assert summary.total == 0


def test_baseline_timeout_status(tmp_path):
    runner, _ = _make_runner(tmp_path, baseline_timeout_seconds=1.0)
    summary = runner.run(_request(FIXTURES / "slow_baseline_pkg"))
    assert summary.status is RunStatus.BASELINE_TIMEOUT
    assert summary.total == 0


def test_invalid_original_source_status(tmp_path):
    runner, _ = _make_runner(tmp_path)
    summary = runner.run(_request(FIXTURES / "invalid_pkg"))
    assert summary.status is RunStatus.INVALID_SOURCE


def test_invalid_syntax_mutant_classification(tmp_path):
    """A mutant whose mutated segment is not valid Python is classified
    invalid_syntax without executing the test suite."""
    from app.mutator import MutationPoint

    runner, _ = _make_runner(tmp_path)
    pkg = tmp_path / "pkg"
    (pkg / "tests").mkdir(parents=True)
    (pkg / "mod.py").write_text("def f(a, b):\n    return a + b\n")
    (pkg / "tests" / "test_mod.py").write_text(
        "def test_f():\n    assert True\n"
    )
    from app.mutator import find_mutation_points
    real = find_mutation_points(
        "mod.py", "def f(a, b):\n    return a + b\n",
        set(MutationOperator),
    )[0]
    point = MutationPoint(
        file_path="mod.py", node_index=real.node_index,
        operator=MutationOperator.ARITHMETIC,
        line=2, col=11, original_segment="a + b",
        mutated_segment="a +", equivalent=False,
    )
    import logging
    record = runner._run_mutant(
        _request(pkg), "runx", 0, pkg, pkg, "tests", point,
        logging.getLogger("test"),
    )
    assert record.outcome.value == "invalid_syntax"
    assert "does not compile" in record.detail
