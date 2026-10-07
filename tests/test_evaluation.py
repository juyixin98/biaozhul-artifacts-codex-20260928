"""Synthetic co-occurrence probe, explicitly not a semantic benchmark."""
from __future__ import annotations

from app.config import TrainConfig
from app.evaluation import evaluate_pairs
from app.training import train_skipgram


def test_repeated_synthetic_pattern_recovers_expected_partner(evidence):
    # Words repeatedly sharing a center acquire comparable input embeddings.
    corpus = (
        [["cat", "pet"], ["pet", "cat"], ["dog", "pet"], ["pet", "dog"]] * 8
        + [["car", "road"], ["road", "car"], ["bus", "road"], ["road", "bus"]] * 8
        + [["cat", "pet"], ["dog", "pet"]] * 4
    )
    config = TrainConfig(
        seed=991,
        epochs=15,
        window=1,
        negatives=3,
        dim=8,
        learning_rate=0.08,
        subsample_t=1.0,
        min_count=1,
        init_range=0.1,
    )
    result = train_skipgram(corpus, config, run_id="synthetic-pattern")
    metrics = evaluate_pairs(result.artifact, [("cat", "dog"), ("car", "bus")])

    assert metrics["kind"] == "synthetic_cooccurrence_rank"
    assert metrics["results"][0]["rank"] == 1
    assert metrics["results"][1]["rank"] == 1
    assert metrics["top1_rate"] == 1.0
    assert metrics["mean_reciprocal_rank"] == 1.0
    assert "not a semantic understanding benchmark" in metrics["interpretation"]
    evidence(
        "synthetic_cooccurrence_ranks",
        {"ranks": [1, 1], "top1_rate": 1.0},
        {
            "ranks": [item["rank"] for item in metrics["results"]],
            "top1_rate": metrics["top1_rate"],
        },
        "repeated local pairs should be recovered without claiming language understanding",
    )
