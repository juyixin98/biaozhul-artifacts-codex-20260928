"""Evaluation metrics for a held-out evaluation set (never the train set)."""
from __future__ import annotations

import numpy as np

from .validation import validate_arrays

EPS = 1e-15


def log_loss(probs: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    p = np.clip(probs, EPS, 1.0 - EPS)
    loss = -(labels * np.log(p) + (1.0 - labels) * np.log1p(-p))
    return float(np.dot(weights, loss) / weights.sum())


def brier_score(probs: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(weights, (probs - labels) ** 2) / weights.sum())


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, weights: np.ndarray, n_bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    total = weights.sum()
    for b in range(n_bins):
        mask = idx == b
        wb = weights[mask].sum()
        if wb == 0:
            continue
        conf = np.dot(weights[mask], probs[mask]) / wb
        acc = np.dot(weights[mask], labels[mask]) / wb
        ece += (wb / total) * abs(acc - conf)
    return float(ece)


def evaluate(
    calibrator,
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Evaluate a fitted calibrator on a separate evaluation set."""
    s, y, w, summary = validate_arrays(scores, labels, weights)
    probs = calibrator.predict(s)
    return {
        "n_eval_rows": summary.n_used,
        "n_zero_weight_dropped": summary.n_zero_weight_dropped,
        "log_loss": log_loss(probs, y, w),
        "brier_score": brier_score(probs, y, w),
        "ece": expected_calibration_error(probs, y, w, n_bins),
        "base_rate": float(np.dot(w, y) / w.sum()),
        "mean_predicted": float(np.dot(w, probs) / w.sum()),
    }
