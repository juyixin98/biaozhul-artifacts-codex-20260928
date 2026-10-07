"""Kernel classification tests. Expected verdicts are hand-derived from
the fixture program's documented exit protocol, not from the reducer."""

from app.kernel import run_candidate
from app.models import Verdict

from tests.test_models import make_doc


def test_target_error_when_all_triggers_present(settings):
    doc = make_doc("mode=strict", "level=3", "debug=on")
    outcome = run_candidate(settings, doc)
    assert outcome.verdict == Verdict.TARGET
    assert outcome.exit_code == 3


def test_pass_when_a_trigger_is_missing(settings):
    doc = make_doc("mode=strict", "level=3")  # no debug=on
    outcome = run_candidate(settings, doc)
    assert outcome.verdict == Verdict.PASS
    assert outcome.exit_code == 0


def test_other_error_on_bad_level(settings):
    doc = make_doc("mode=strict", "level=abc", "debug=on")
    outcome = run_candidate(settings, doc)
    assert outcome.verdict == Verdict.OTHER
    assert outcome.exit_code == 2


def test_indeterminate_on_hang(settings):
    settings.run_timeout_sec = 0.3
    doc = make_doc("simulate_hang=1")
    outcome = run_candidate(settings, doc)
    assert outcome.verdict == Verdict.INDETERMINATE
    assert outcome.exit_code is None
