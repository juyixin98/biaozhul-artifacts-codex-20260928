"""Validated training configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .errors import ValidationError
from .rng import validate_seed


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1729
    epochs: int = 5
    window: int = 2
    negatives: int = 5
    dim: int = 16
    learning_rate: float = 0.025
    subsample_t: float = 1e-3
    min_count: int = 1
    init_range: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> "TrainConfig":
        if not isinstance(data, dict):
            raise ValidationError("training configuration must be an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValidationError("unknown training configuration fields", {"fields": unknown})

        values = {name: field.default for name, field in cls.__dataclass_fields__.items()}
        values.update(data)
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        validate_seed(self.seed)
        integer_fields = {
            "epochs": (1, 1000),
            "window": (1, 32),
            "negatives": (1, 100),
            "dim": (2, 512),
            "min_count": (1, 1_000_000),
        }
        for name, (lower, upper) in integer_fields.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationError(f"{name} must be an integer", {"actual": repr(value)})
            if not lower <= value <= upper:
                raise ValidationError(
                    f"{name} must be in [{lower}, {upper}]",
                    {"actual": value, "min": lower, "max": upper},
                )
        for name, lower in (("learning_rate", 1e-6), ("subsample_t", 1e-6)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(f"{name} must be a number")
            if not lower <= float(value) <= 1.0:
                raise ValidationError(f"{name} must be in [{lower}, 1.0]", {"actual": value})
        if not 0.0 < float(self.init_range) <= 10.0:
            raise ValidationError("init_range must be in (0, 10]", {"actual": self.init_range})
