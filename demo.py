#!/usr/bin/env python3
"""Local end-to-end demonstration using only a synthetic in-repo corpus.

Run:
    python3 demo.py

The script creates a fresh data directory under ``./demo_var``, validates a
small synthetic corpus, trains one frozen-vocabulary skip-gram run, verifies
bit-identical replay with the same seed, and then prints vectors, similarity
and nearest neighbours. It never contacts external services.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from sgns.artifact import load_artifact
from sgns.config import TrainConfig
from sgns.service import EmbeddingService

DEMO_DIR = Path(__file__).resolve().parent / "demo_var"
CORPUS_ID = "demo-capitals"
RUN_ID = "demo-run"

CORPUS = """london england london england city london
paris france paris france city paris
berlin germany berlin germany city berlin
london england city paris france city
berlin germany london paris city
"""

CONFIG = TrainConfig(
    dim=8,
    lr=0.05,
    epochs=40,
    window=2,
    negatives=5,
    min_count=1,
    subsample_t=0.0,
    negative_exponent=0.75,
    seed=20260101,
)


def main() -> None:
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    service = EmbeddingService(
        db_path=DEMO_DIR / "sgns.db",
        artifact_dir=DEMO_DIR / "artifacts",
    )

    corpus_info = service.create_corpus(CORPUS_ID, CORPUS)
    print("corpus:", json.dumps(corpus_info))

    first = service.start_run(
        RUN_ID,
        CORPUS_ID,
        CONFIG.to_dict(),
        record_events=1,
    )
    print("run status:", first["status"])
    print("stats:", json.dumps(first["stats"], sort_keys=True))
    print("artifact sha256:", first["artifact"]["sha256"])

    # Independent replay through a second service directory: same seed must
    # reproduce both parameter tables bit-for-bit.
    replay_service = EmbeddingService(
        db_path=DEMO_DIR / "replay.db",
        artifact_dir=DEMO_DIR / "replay_artifacts",
    )
    replay_service.create_corpus(CORPUS_ID, CORPUS)
    replay_service.start_run(RUN_ID, CORPUS_ID, CONFIG.to_dict())
    artifact_a = load_artifact(DEMO_DIR / "artifacts", RUN_ID)
    artifact_b = load_artifact(DEMO_DIR / "replay_artifacts", RUN_ID)
    print(
        "replay identical:",
        bool(
            np.array_equal(artifact_a.input_vectors, artifact_b.input_vectors)
            and np.array_equal(
                artifact_a.output_vectors, artifact_b.output_vectors
            )
        ),
    )

    trace = first["trace"][0]
    print(
        "first event:",
        json.dumps(
            {
                "epoch": trace["epoch"],
                "sentence_index": trace["sentence_index"],
                "center_position": trace["center_position"],
                "dynamic_window": trace["dynamic_window"],
                "center_id": trace["center_id"],
                "positive_id": trace["positive_id"],
                "negative_ids": trace["negative_ids"],
                "scores": trace["scores"],
                "coefficients": trace["coefficients"],
            },
            sort_keys=True,
        ),
    )

    for word in ("london", "england", "paris", "france"):
        vector = service.vector(RUN_ID, word)
        print(
            f"vector[{word}] norm={vector['norm']:.6f} "
            f"first3={[round(x, 4) for x in vector['vector'][:3]]}"
        )

    for pair in (("london", "england"), ("paris", "france"), ("london", "paris")):
        outcome = service.similarity(RUN_ID, *pair)
        print(f"similarity{pair}: {outcome['similarity']:.6f}")

    for word in ("london", "paris"):
        outcome = service.neighbours(RUN_ID, word, top_k=3)
        ranked = [
            f"{item['word']}({item['similarity']:.3f})"
            for item in outcome["neighbours"]
        ]
        print(f"neighbours[{word}]: {ranked}")

    print(
        "evaluation note: synthetic fixture only; no real semantic benchmark "
        "is claimed."
    )


if __name__ == "__main__":
    main()
