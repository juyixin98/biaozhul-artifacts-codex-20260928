"""Cache binding tests: the cache key must bind program digest,
environment fingerprint, and input."""

from app.config import Settings
from app.kernel import cache_key, program_digest
from app.models import Verdict

from tests.test_models import make_doc


def test_cache_hit_on_repeat_evaluation(reducer, store):
    doc = make_doc("mode=strict", "level=3", "debug=on")
    first = reducer.evaluate(doc)
    assert first.verdict == Verdict.TARGET
    assert store.cache_size() == 1
    second = reducer.evaluate(doc)
    assert second.verdict == Verdict.TARGET
    assert second.reason.startswith("cache hit")
    assert store.cache_size() == 1  # no new entry, no re-run


def test_cache_key_binds_input(settings):
    digest = program_digest(settings.target_program)
    k1 = cache_key(settings, digest, make_doc("a=1"))
    k2 = cache_key(settings, digest, make_doc("a=2"))
    assert k1 != k2


def test_cache_key_binds_program_digest(settings):
    doc = make_doc("a=1")
    k1 = cache_key(settings, "digest-A", doc)
    k2 = cache_key(settings, "digest-B", doc)
    assert k1 != k2


def test_cache_key_binds_environment(settings, tmp_path):
    doc = make_doc("a=1")
    digest = program_digest(settings.target_program)
    other_env = Settings(
        db_path=str(tmp_path / "other.db"),
        target_program=settings.target_program,
        env_fingerprint_extra=("simulated-env-change",),
    )
    assert cache_key(settings, digest, doc) != cache_key(other_env, digest, doc)
