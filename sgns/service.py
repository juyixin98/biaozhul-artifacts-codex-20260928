"""Application service orchestrating validation, training, artifacts, queries.

State machine for a run id:

    PENDING (created by start_run, training happens synchronously)
      -> COMPLETED (artifact written, digest recorded)
      -> FAILED    (computation failure recorded; id remains non-restartable)

A second ``start_run`` with an existing id is STATE_CONFLICT. Query methods
only load COMPLETED runs and validate the NPZ digest on every load, so a
tampered artifact is COMPUTATION_FAILED rather than a stale answer.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from . import evaluation
from .artifact import ModelArtifact, load_artifact, save_artifact
from .config import TrainConfig
from .data import tokenize_sentences, validate_corpus_text
from .errors import StateConflictError
from .kernel import train_skipgram
from .sampling import NegativeTable
from .storage import Store
from .vocab import build_vocabulary, encode_sentence

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_id(kind: str, value: str) -> None:
    from .errors import InputError

    if not isinstance(value, str) or not _ID_RE.match(value):
        raise InputError(
            f"{kind} must be 1-64 letters, digits, '-' or '_'",
            value=value,
        )


class EmbeddingService:
    def __init__(self, db_path: str | Path, artifact_dir: str | Path) -> None:
        self.store = Store(db_path)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def create_corpus(
        self,
        corpus_id: str,
        text: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        _validate_id("corpus_id", corpus_id)
        normalized = validate_corpus_text(text, max_tokens)
        sentences = tokenize_sentences(normalized, max_tokens)
        try:
            self.store.create_corpus(corpus_id, normalized, max_tokens)
        except sqlite3.IntegrityError as exc:
            raise StateConflictError(
                "corpus id already exists",
                corpus_id=corpus_id,
            ) from exc
        token_count = sum(len(s) for s in sentences)
        return {
            "corpus_id": corpus_id,
            "sentences": len(sentences),
            "tokens": token_count,
        }

    def start_run(
        self,
        run_id: str,
        corpus_id: str,
        config_overrides: dict[str, Any] | None = None,
        record_events: int = 0,
    ) -> dict[str, Any]:
        _validate_id("run_id", run_id)
        _validate_id("corpus_id", corpus_id)
        config = TrainConfig.from_dict(dict(config_overrides or {}))
        row = self.store.get_corpus(corpus_id)
        self.store.create_run(run_id, corpus_id, config.to_dict())
        try:
            sentences = tokenize_sentences(row["text"], row["max_tokens"])
            vocab = build_vocabulary(
                sentences,
                min_count=config.min_count,
                subsample_t=config.subsample_t,
            )
            encoded = [encode_sentence(s, vocab) for s in sentences]
            table = NegativeTable.build(vocab, config.negative_exponent)
            result = train_skipgram(
                encoded,
                vocab,
                table,
                config,
                record_events=record_events,
            )
            artifact = ModelArtifact(
                vocab=vocab,
                config=config,
                input_vectors=result.input_vectors,
                output_vectors=result.output_vectors,
            )
            saved = save_artifact(self.artifact_dir, run_id, artifact)
            stats = result.stats.to_dict()
            stats["vocab_size"] = vocab.size
            stats["dropped_types"] = len(vocab.raw_counts) - vocab.size
            self.store.mark_completed(
                run_id,
                artifact_path=str(saved["path"]),
                artifact_sha256=str(saved["sha256"]),
                stats=stats,
            )
            response = {
                "run_id": run_id,
                "corpus_id": corpus_id,
                "status": "COMPLETED",
                "config": config.to_dict(),
                "stats": stats,
                "artifact": saved,
            }
            if result.trace:
                response["trace"] = [event.to_dict() for event in result.trace]
            return response
        except Exception as exc:
            from .errors import SGNSError

            if isinstance(exc, SGNSError):
                self.store.mark_failed(run_id, exc.to_dict())
            raise

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.store.get_run(run_id)
        return self._run_row_to_dict(row)

    def _completed_artifact(self, run_id: str) -> ModelArtifact:
        row = self.store.get_run(run_id)
        if row["status"] != "COMPLETED":
            raise StateConflictError(
                "run has no completed artifact to query",
                run_id=run_id,
                status=row["status"],
            )
        return load_artifact(self.artifact_dir, run_id)

    def vector(self, run_id: str, word: str) -> dict[str, Any]:
        artifact = self._completed_artifact(run_id)
        result = evaluation.get_vector(artifact, word)
        result["run_id"] = run_id
        return result

    def similarity(self, run_id: str, word_a: str, word_b: str) -> dict[str, Any]:
        artifact = self._completed_artifact(run_id)
        result = evaluation.cosine_similarity(artifact, word_a, word_b)
        result["run_id"] = run_id
        return result

    def neighbours(self, run_id: str, word: str, top_k: int = 5) -> dict[str, Any]:
        artifact = self._completed_artifact(run_id)
        result = evaluation.nearest_neighbours(artifact, word, top_k)
        result["run_id"] = run_id
        return result

    def evaluate(self, run_id: str, pairs: list[dict[str, Any]]) -> dict[str, Any]:
        artifact = self._completed_artifact(run_id)
        result = evaluation.evaluate_pairs(artifact, pairs)
        result["run_id"] = run_id
        return result

    @staticmethod
    def _run_row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "corpus_id": row["corpus_id"],
            "status": row["status"],
            "config": json.loads(row["config_json"]),
            "artifact_path": row["artifact_path"],
            "artifact_sha256": row["artifact_sha256"],
            "stats": json.loads(row["stats_json"]) if row["stats_json"] else None,
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_runs(self, corpus_id: str | None = None) -> dict[str, Any]:
        return {
            "runs": [self._run_row_to_dict(row) for row in
                     self.store.list_runs(corpus_id)]
        }
