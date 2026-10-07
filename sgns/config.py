"""Immutable training configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import InputError


@dataclass(frozen=True)
class TrainConfig:
    """Parameters for one training run.

    ``seed`` fully determines sample order, subsampling and negative draws.
    ``min_count`` discards words below a raw frequency before the vocabulary
    is frozen. ``subsample_t`` is the word2vec-style threshold; ``0.0``
    disables subsampling. ``negative_exponent`` defaults to unigram**0.75.
    """

    dim: int = 8
    lr: float = 0.05
    epochs: int = 5
    window: int = 2
    negatives: int = 3
    min_count: int = 1
    subsample_t: float = 1e-3
    negative_exponent: float = 0.75
    seed: int = 1234

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not isinstance(self.dim, int) or self.dim <= 0:
            problems.append("dim must be a positive integer")
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            problems.append("epochs must be a positive integer")
        if not isinstance(self.window, int) or self.window <= 0:
            problems.append("window must be a positive integer")
        if not isinstance(self.negatives, int) or self.negatives <= 0:
            problems.append("negatives must be a positive integer")
        if not isinstance(self.min_count, int) or self.min_count <= 0:
            problems.append("min_count must be a positive integer")
        if not self.lr > 0 or not self.lr < float("inf"):
            problems.append("lr must be finite and strictly positive")
        if not 0.0 <= self.subsample_t <= 1.0:
            problems.append("subsample_t must be in [0.0, 1.0]")
        if not 0.0 <= self.negative_exponent <= 2.0:
            problems.append("negative_exponent must be in [0.0, 2.0]")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            problems.append("seed must be a non-negative integer below 2**63")
        if problems:
            raise InputError("invalid training configuration", reasons=problems)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "dim": self.dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "window": self.window,
            "negatives": self.negatives,
            "min_count": self.min_count,
            "subsample_t": self.subsample_t,
            "negative_exponent": self.negative_exponent,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "TrainConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise InputError(
                "invalid training configuration",
                unknown_fields=unknown,
            )
        return cls(**raw)  # type: ignore[arg-type]
