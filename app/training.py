"""Skip-gram negative-sampling training orchestration."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import TrainConfig
from .data import Vocab, build_vocab, corpus_fingerprint, validate_corpus
from .errors import ResourceExhaustedError
from .model import ModelArtifact, initial_embeddings, negative_sampling_step
from .rng import (
    STREAM_INIT,
    STREAM_NEGATIVE,
    STREAM_ORDER,
    STREAM_SUBSAMPLE,
    child_rng,
)
from .sampling import build_epoch_pairs, draw_negative_indexes

EventListener = Callable[[dict], None]


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    artifact: ModelArtifact
    vocab: Vocab
    events: tuple[dict, ...]
    final_loss: float


def make_run_id(corpus: list[list[str]], config: TrainConfig) -> str:
    digest = hashlib.sha256(
        (
            corpus_fingerprint(corpus)
            + ":"
            + json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"run-{digest}"


def _rng_state(rng: np.random.Generator) -> dict:
    bit_generator = rng.bit_generator
    state = bit_generator.state
    return {
        "bit_generator": state["bit_generator"],
        "state": int(state["state"]["state"]),
        "inc": int(state["state"]["inc"]),
    }


def train_skipgram(
    corpus: list[list[str]],
    config: TrainConfig,
    *,
    run_id: str | None = None,
    event_listener: EventListener | None = None,
) -> TrainingResult:
    validate_corpus(corpus)
    config.validate()
    fingerprint = corpus_fingerprint(corpus)
    run_id = run_id or make_run_id(corpus, config)
    if not isinstance(run_id, str) or not run_id:
        raise ResourceExhaustedError("invalid generated run identifier")

    events: list[dict] = []

    def record(event: dict) -> None:
        events.append(event)
        if event_listener is not None:
            event_listener(event)

    init_rng = child_rng(config.seed, STREAM_INIT)
    subsample_rng = child_rng(config.seed, STREAM_SUBSAMPLE)
    order_rng = child_rng(config.seed, STREAM_ORDER)
    negative_rng = child_rng(config.seed, STREAM_NEGATIVE)
    vocab = build_vocab(corpus, min_count=config.min_count)

    record(
        {
            "type": "run_started",
            "run_id": run_id,
            "seed": config.seed,
            "config": config.to_dict(),
            "corpus_fingerprint": fingerprint,
            "sentence_count": len(corpus),
            "token_count": int(vocab.counts.sum()),
            "rng_streams": {
                name: _rng_state(rng)
                for name, rng in (
                    (STREAM_INIT, init_rng),
                    (STREAM_SUBSAMPLE, subsample_rng),
                    (STREAM_ORDER, order_rng),
                    (STREAM_NEGATIVE, negative_rng),
                )
            },
        }
    )
    record(
        {
            "type": "vocab_frozen",
            "run_id": run_id,
            "tokens": list(vocab.tokens),
            "counts": [int(x) for x in vocab.counts],
            "frequencies": [float(x) for x in vocab.frequencies],
            "negative_distribution": [float(x) for x in vocab.negative_distribution],
        }
    )

    input_matrix, output_matrix = initial_embeddings(
        vocab.size, config.dim, init_rng, init_range=config.init_range
    )
    record(
        {
            "type": "vectors_initialized",
            "run_id": run_id,
            "input_rows": vocab.size,
            "dim": config.dim,
            "init_range": config.init_range,
            "init_rng_after_initialization": _rng_state(init_rng),
            "output_initial_norm": 0.0,
        }
    )

    last_mean_loss = 0.0
    for epoch in range(1, config.epochs + 1):
        pairs = build_epoch_pairs(
            corpus,
            vocab,
            subsample_rng,
            window=config.window,
            subsample_t=config.subsample_t,
        )
        if not pairs:
            raise ResourceExhaustedError(
                "epoch contains no trainable pairs after sub-sampling",
                {"run_id": run_id, "epoch": epoch},
            )
        ordered = [pairs[i] for i in order_rng.permutation(len(pairs))]
        record(
            {
                "type": "epoch_started",
                "run_id": run_id,
                "epoch": epoch,
                "pair_count": len(ordered),
                "first_pairs": [list(pair) for pair in ordered[:5]],
                "subsample_rng_after_pairs": _rng_state(subsample_rng),
                "order_rng_after_shuffle": _rng_state(order_rng),
            }
        )

        losses: list[float] = []
        for sequence, (center_id, positive_id) in enumerate(ordered, start=1):
            rejected_negatives: list[dict] = []

            def record_negative_rejection(attempt: int, candidate_id: int) -> None:
                rejected_negatives.append(
                    {
                        "attempt": attempt,
                        "token_id": candidate_id,
                        "token": vocab.tokens[candidate_id],
                    }
                )

            negative_ids = draw_negative_indexes(
                negative_rng,
                vocab.negative_distribution,
                positive_index=positive_id,
                count=config.negatives,
                on_reject=record_negative_rejection,
            )
            detail = negative_sampling_step(
                input_matrix,
                output_matrix,
                center_id=center_id,
                positive_id=positive_id,
                negative_ids=negative_ids,
                learning_rate=config.learning_rate,
            )
            losses.append(detail["loss"])
            record(
                {
                    "type": "step",
                    "run_id": run_id,
                    "epoch": epoch,
                    "sequence": sequence,
                    "center": vocab.tokens[center_id],
                    "positive": vocab.tokens[positive_id],
                    "negatives": [vocab.tokens[i] for i in negative_ids],
                    "center_id": center_id,
                    "positive_id": positive_id,
                    "negative_ids": negative_ids,
                    "rejected_negative_candidates": rejected_negatives,
                    "loss": detail["loss"],
                    "scores": detail["scores"],
                    "probabilities": detail["probabilities"],
                    "input_gradient_norm": float(
                        np.linalg.norm(detail["input_gradient"])
                    ),
                    "output_gradient_norm": float(
                        np.linalg.norm(detail["output_gradients"])
                    ),
                    "negative_rng_after_step": _rng_state(negative_rng),
                }
            )
        last_mean_loss = float(np.mean(losses))
        record(
            {
                "type": "epoch_completed",
                "run_id": run_id,
                "epoch": epoch,
                "pair_count": len(ordered),
                "mean_loss": last_mean_loss,
                "first_loss": float(losses[0]),
                "last_loss": float(losses[-1]),
            }
        )

    artifact = ModelArtifact(
        run_id=run_id,
        tokens=vocab.tokens,
        counts=vocab.counts,
        config=config.to_dict(),
        corpus_fingerprint=fingerprint,
        input_matrix=input_matrix,
        output_matrix=output_matrix,
    )
    record({"type": "run_completed", "run_id": run_id, "mean_loss": last_mean_loss})
    return TrainingResult(
        run_id=run_id,
        artifact=artifact,
        vocab=vocab,
        events=tuple(events),
        final_loss=last_mean_loss,
    )
