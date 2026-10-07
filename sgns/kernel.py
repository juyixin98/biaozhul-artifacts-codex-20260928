"""Skip-gram negative-sampling training kernel.

Math (single event, one positive target plus ``K`` negatives):

    s_j      = u_center . v_j
    sigmoid  = stable 1 / (1 + exp(-s))
    g_pos    = lr * (1 - sigmoid(s_pos))
    g_neg    = -lr * sigmoid(s_neg)
    u_center += g_pos * v_pos + sum_k g_neg_k * v_neg_k
    v_j      += g_j * u_center

All output rows used inside one event are snapshotted *before* any update,
so duplicate negatives accumulate on the same pre-update row and the two
parameter sets (input vectors ``u`` and output vectors ``v``) can never
overwrite each other.

Replay order is fixed: epochs ascending, sentences in insertion order,
center positions ascending, context positions ascending. The dynamic window
for a center is one fresh integer draw in ``[1, window]``. Every random
draw (init, subsampling, window size, inverse-CDF negatives) comes from one
``numpy.random.Generator(default_rng(seed))`` stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import TrainConfig
from .errors import ComputationFailureError
from .sampling import NegativeTable
from .vocab import Vocabulary, apply_subsampling


def stable_sigmoid(x: float) -> float:
    """Numerically stable logistic; non-finite input raises ValueError."""
    if not np.isfinite(x):
        raise ValueError(f"sigmoid received non-finite value {x!r}")
    if x >= 0.0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def _require_finite(name: str, value: np.ndarray | float, event: dict[str, Any]) -> None:
    scalar = float(np.asarray(value).sum()) if np.ndim(value) else float(value)
    if not np.isfinite(scalar):
        raise ComputationFailureError(
            f"non-finite value encountered in {name}",
            stage=name,
            event={k: v for k, v in event.items() if k != "vectors"},
        )


@dataclass
class EventTrace:
    """Serializable intermediate state for one skip-gram event."""

    epoch: int
    sentence_index: int
    center_position: int
    dynamic_window: int
    center_id: int
    positive_id: int
    negative_ids: list[int]
    scores: list[float]
    coefficients: list[float]
    input_vector_before: list[float]
    input_vector_after: list[float]
    positive_output_before: list[float]
    positive_output_after: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "sentence_index": self.sentence_index,
            "center_position": self.center_position,
            "dynamic_window": self.dynamic_window,
            "center_id": self.center_id,
            "positive_id": self.positive_id,
            "negative_ids": self.negative_ids,
            "scores": self.scores,
            "coefficients": self.coefficients,
            "input_vector_before": self.input_vector_before,
            "input_vector_after": self.input_vector_after,
            "positive_output_before": self.positive_output_before,
            "positive_output_after": self.positive_output_after,
        }


@dataclass
class TrainStats:
    epochs: int = 0
    centers: int = 0
    events: int = 0
    tokens_in: int = 0
    tokens_kept: int = 0
    negative_redraws: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "epochs": self.epochs,
            "centers": self.centers,
            "events": self.events,
            "tokens_in": self.tokens_in,
            "tokens_kept": self.tokens_kept,
            "negative_redraws": self.negative_redraws,
        }


@dataclass
class TrainResult:
    input_vectors: np.ndarray
    output_vectors: np.ndarray
    stats: TrainStats
    trace: list[EventTrace] = field(default_factory=list)


def initialize_matrices(
    vocab_size: int,
    dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Input vectors uniform in [-0.5/dim, 0.5/dim); output vectors zero."""
    input_vectors = (rng.random((vocab_size, dim), dtype=np.float64) - 0.5) / dim
    output_vectors = np.zeros((vocab_size, dim), dtype=np.float64)
    return input_vectors, output_vectors


def skipgram_event(
    input_vectors: np.ndarray,
    output_vectors: np.ndarray,
    center_id: int,
    positive_id: int,
    negative_ids: list[int],
    lr: float,
) -> tuple[list[float], list[float]]:
    """Apply one skip-gram negative-sampling event in place.

    Returns ``(scores, coefficients)``: one score and signed learning-rate
    coefficient per target (positive first, then negatives).
    """
    targets = [positive_id, *negative_ids]
    event = {"center_id": center_id, "targets": targets}
    u_before = input_vectors[center_id].copy()
    v_before = {target: output_vectors[target].copy() for target in set(targets)}

    scores: list[float] = []
    coefficients: list[float] = []
    grad_u = np.zeros_like(u_before)
    row_deltas: dict[int, np.ndarray] = {}
    for rank, target_id in enumerate(targets):
        score = float(np.dot(u_before, v_before[target_id]))
        _require_finite(f"score.target_{rank}", score, event)
        if rank == 0:
            coefficient = lr * (1.0 - stable_sigmoid(score))
        else:
            coefficient = -lr * stable_sigmoid(score)
        _require_finite(f"coefficient.target_{rank}", coefficient, event)
        scores.append(score)
        coefficients.append(coefficient)
        grad_u += coefficient * v_before[target_id]
        row_deltas[target_id] = row_deltas.get(
            target_id,
            np.zeros_like(u_before),
        ) + coefficient * u_before

    _require_finite("input_gradient", grad_u, event)
    input_vectors[center_id] = u_before + grad_u
    _require_finite("input_vector_update", input_vectors[center_id], event)
    for target_id, delta in row_deltas.items():
        output_vectors[target_id] = v_before[target_id] + delta
        _require_finite("output_vector_update", output_vectors[target_id], event)
    return scores, coefficients


def train_skipgram(
    encoded_sentences: list[list[int]],
    vocab: Vocabulary,
    negatives_table: NegativeTable,
    config: TrainConfig,
    record_events: int = 0,
) -> TrainResult:
    """Run the full deterministic training loop and return both vector sets."""
    rng = np.random.default_rng(config.seed)
    input_vectors, output_vectors = initialize_matrices(vocab.size, config.dim, rng)
    stats = TrainStats()
    trace: list[EventTrace] = []

    for epoch in range(config.epochs):
        for sentence_index, raw_ids in enumerate(encoded_sentences):
            stats.tokens_in += len(raw_ids)
            kept_ids = apply_subsampling(raw_ids, vocab, rng)
            stats.tokens_kept += len(kept_ids)
            for position, center_id in enumerate(kept_ids):
                dynamic_window = int(rng.integers(1, config.window + 1))
                left = max(0, position - dynamic_window)
                right = min(len(kept_ids), position + dynamic_window + 1)
                context = kept_ids[left:right]
                stats.centers += 1
                for target_position, positive_id in enumerate(context):
                    target_abs = left + target_position
                    if target_abs == position:
                        continue
                    negatives, attempts = negatives_table.sample_negatives_tracked(
                        positive_id,
                        config.negatives,
                        rng,
                    )
                    stats.negative_redraws += attempts
                    u_before = input_vectors[center_id].copy()
                    v_pos_before = output_vectors[positive_id].copy()
                    scores, coefficients = skipgram_event(
                        input_vectors,
                        output_vectors,
                        center_id,
                        positive_id,
                        negatives,
                        config.lr,
                    )
                    stats.events += 1
                    if len(trace) < record_events:
                        trace.append(
                            EventTrace(
                                epoch=epoch,
                                sentence_index=sentence_index,
                                center_position=position,
                                dynamic_window=dynamic_window,
                                center_id=center_id,
                                positive_id=positive_id,
                                negative_ids=list(negatives),
                                scores=scores,
                                coefficients=coefficients,
                                input_vector_before=u_before.tolist(),
                                input_vector_after=input_vectors[center_id].tolist(),
                                positive_output_before=v_pos_before.tolist(),
                                positive_output_after=output_vectors[positive_id].tolist(),
                            )
                        )
        stats.epochs = epoch + 1
    return TrainResult(
        input_vectors=input_vectors,
        output_vectors=output_vectors,
        stats=stats,
        trace=trace,
    )
