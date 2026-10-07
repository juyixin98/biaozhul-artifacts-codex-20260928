"""Reduction tests with a hand-written reference answer.

The fixture input triggers the bug only when mode=strict, level>=3 and
debug=on are ALL present, so the hand-derived 1-minimal result under
block removal is exactly: header, mode=strict, level=5, debug=on, footer.
"""

from app.kernel import run_candidate
from app.models import Verdict

EXPECTED_FINAL_TEXT = (
    "BEGIN config-v1\nmode=strict\nlevel=5\ndebug=on\nEND config-v1\n"
)


def test_reduction_matches_handwritten_reference(reducer, sample_doc):
    job_id = reducer.start(sample_doc)
    result = reducer.run(job_id)
    assert result.status == "done"
    assert result.final_doc.to_text() == EXPECTED_FINAL_TEXT
    assert result.final_size == 5
    assert "1-minimal under the block-removal operator" in (
        result.minimality_claim)
    assert "global" in result.minimality_claim  # explicitly disclaimed


def test_reduced_input_still_reproduces_target_error(reducer, settings,
                                                     sample_doc):
    job_id = reducer.start(sample_doc)
    result = reducer.run(job_id)
    outcome = run_candidate(settings, result.final_doc)
    assert outcome.verdict == Verdict.TARGET


def test_result_is_1_minimal_independent_check(reducer, settings, sample_doc):
    """Independently verify 1-minimality: removing ANY remaining removable
    block must no longer deterministically fail with the target error.
    This check uses the kernel directly, not the reducer."""
    job_id = reducer.start(sample_doc)
    result = reducer.run(job_id)
    removable = result.final_doc.removable_indexes()
    assert removable, "reference result should still have removable blocks"
    for idx in removable:
        outcome = run_candidate(settings, result.final_doc.without(idx))
        assert outcome.verdict != Verdict.TARGET, (
            f"removing block {idx} still reproduces target error; "
            "result is not 1-minimal"
        )


def test_every_candidate_kept_syntax_valid(reducer, store, sample_doc):
    job_id = reducer.start(sample_doc)
    reducer.run(job_id)
    steps = store.get_steps(job_id)
    assert steps, "expected recorded evidence steps"
    job = store.get_job(job_id)
    final_doc = store.doc_from_canonical(job["current_doc"])
    assert final_doc.validate_syntax() == []


def test_interruption_and_resume(reducer, store, sample_doc):
    job_id = reducer.start(sample_doc)
    partial = reducer.run(job_id, max_steps=1)
    assert partial.status == "interrupted"
    job = store.get_job(job_id)
    assert job["status"] == "interrupted"

    resumed = reducer.run(job_id)  # resume from checkpoint
    assert resumed.status == "done"
    assert resumed.final_doc.to_text() == EXPECTED_FINAL_TEXT

    # A fresh uninterrupted run must converge to the same result.
    job2 = reducer.start(sample_doc)
    full = reducer.run(job2)
    assert full.final_doc.canonical() == resumed.final_doc.canonical()


def test_refuses_non_failing_input(reducer):
    import pytest

    from tests.test_models import make_doc

    doc = make_doc("mode=strict")  # does not trigger the bug
    job_id = reducer.start(doc)
    with pytest.raises(RuntimeError, match="does not reproduce the target"):
        reducer.run(job_id)
