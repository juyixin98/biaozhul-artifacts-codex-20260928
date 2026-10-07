"""Local end-to-end demonstration without external services or accounts."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation import evaluate_pairs
from app.prediction import cosine_similarity, most_similar
from app.storage import ModelStore
from app.training import train_skipgram
from app.config import TrainConfig


CORPUS = [
    ["river", "bank"],
    ["bank", "river"],
    ["creek", "bank"],
    ["bank", "creek"],
    ["stream", "bank"],
    ["bank", "stream"],
    ["brook", "bank"],
    ["bank", "brook"],
    ["tree", "forest"],
    ["forest", "tree"],
    ["bush", "forest"],
    ["forest", "bush"],
    ["fern", "forest"],
    ["forest", "fern"],
    ["plant", "forest"],
    ["forest", "plant"],
]


def main() -> None:
    data_dir = ROOT / "demo_data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    config = TrainConfig(
        seed=20260401,
        epochs=12,
        window=1,
        negatives=3,
        dim=8,
        learning_rate=0.08,
        subsample_t=1.0,
        min_count=1,
        init_range=0.1,
    )
    with ModelStore(str(data_dir / "runs.sqlite"), str(data_dir / "artifacts")) as store:
        result = train_skipgram(CORPUS, config, run_id="demo-replay-001")
        store.save_training_run(result, CORPUS)
        artifact = store.load_artifact(result.run_id)
        payload = {
            "run_id": result.run_id,
            "seed": config.seed,
            "vocab": list(result.vocab.tokens),
            "final_mean_loss": result.final_loss,
            "river_creek_similarity": cosine_similarity(artifact, "river", "creek"),
            "river_neighbors": most_similar(artifact, "river", k=3),
            "synthetic_evaluation": evaluate_pairs(
                artifact, [("river", "creek"), ("tree", "bush")]
            ),
            "note": "Synthetic co-occurrence ranking only; no semantic understanding claim.",
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
