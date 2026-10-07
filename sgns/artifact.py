"""Model artifacts: frozen vocabulary, config, vectors and NPZ persistence.

An artifact is one ``.npz`` file (numpy zipped archive) containing:

    * input vectors  (the word vectors served for similarity)
    * output vectors (kept so training semantics and recovery are auditable)
    * vocabulary and config JSON, stored as fixed metadata arrays

A sidecar ``<name>.npz.sha256`` records the archive digest. Loading
recomputes the digest and rejects mismatches as ``COMPUTATION_FAILED``:
corrupted/mutated artifacts must never silently serve wrong vectors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TrainConfig
from .errors import ComputationFailureError, NotFoundError
from .vocab import Vocabulary

ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ModelArtifact:
    vocab: Vocabulary
    config: TrainConfig
    input_vectors: np.ndarray
    output_vectors: np.ndarray
    version: int = ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if self.input_vectors.shape != (self.vocab.size, self.config.dim):
            raise ComputationFailureError(
                "input vectors do not match vocabulary/config shape",
                observed=list(self.input_vectors.shape),
                expected=[self.vocab.size, self.config.dim],
            )
        if self.output_vectors.shape != self.input_vectors.shape:
            raise ComputationFailureError(
                "output vectors do not match input vectors shape",
                output_shape=list(self.output_vectors.shape),
                input_shape=list(self.input_vectors.shape),
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_artifact(
    directory: str | Path,
    name: str,
    artifact: ModelArtifact,
) -> dict[str, str | int]:
    """Write the NPZ archive atomically and create a sha256 sidecar."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not name.replace("-", "").replace("_", "").isalnum():
        from .errors import InputError

        raise InputError(
            "artifact name must contain only letters, digits, '-' or '_'",
            name=name,
        )
    final_path = directory / f"{name}.npz"
    tmp_path = directory / f"{name}.tmp.npz"
    vocab_json = json.dumps(
        artifact.vocab.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    config_json = json.dumps(
        artifact.config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    np.savez_compressed(
        tmp_path,
        input_vectors=artifact.input_vectors,
        output_vectors=artifact.output_vectors,
        vocab=np.frombuffer(vocab_json.encode("utf-8"), dtype=np.uint8),
        config=np.frombuffer(config_json.encode("utf-8"), dtype=np.uint8),
        version=np.int64(artifact.version),
    )
    tmp_path.replace(final_path)
    digest = _sha256_file(final_path)
    (directory / f"{name}.npz.sha256").write_text(
        f"{digest}  {name}.npz\n",
        encoding="ascii",
    )
    return {
        "path": str(final_path),
        "sha256": digest,
        "size_bytes": final_path.stat().st_size,
    }


def load_artifact(directory: str | Path, name: str) -> ModelArtifact:
    """Load and validate an artifact against its sha256 sidecar."""
    directory = Path(directory)
    path = directory / f"{name}.npz"
    sidecar = directory / f"{name}.npz.sha256"
    if not path.exists():
        raise NotFoundError("model artifact does not exist", path=str(path))
    if not sidecar.exists():
        raise ComputationFailureError(
            "model artifact is missing its sha256 sidecar",
            path=str(sidecar),
        )
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = _sha256_file(path)
    if expected != actual:
        raise ComputationFailureError(
            "model artifact digest mismatch",
            expected=expected,
            actual=actual,
        )
    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["version"])
        if version != ARTIFACT_VERSION:
            raise ComputationFailureError(
                "unsupported artifact version",
                expected=ARTIFACT_VERSION,
                observed=version,
            )
        vocab = Vocabulary.from_dict(
            json.loads(archive["vocab"].tobytes().decode("utf-8"))
        )
        config = TrainConfig.from_dict(
            json.loads(archive["config"].tobytes().decode("utf-8"))
        )
        return ModelArtifact(
            vocab=vocab,
            config=config,
            input_vectors=archive["input_vectors"].astype(np.float64),
            output_vectors=archive["output_vectors"].astype(np.float64),
            version=version,
        )
