"""Deterministic random-number streams.

A single root seed produces named child streams.  Training therefore has a
recorded, replayable sequence even when subsampling, example ordering, and
negative sampling all consume randomness.
"""
from __future__ import annotations

import numpy as np

from .errors import ValidationError


STREAM_SUBSAMPLE = "subsample"
STREAM_ORDER = "example_order"
STREAM_NEGATIVE = "negative"
STREAM_INIT = "init"


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError("seed must be an integer", {"value": repr(seed)})
    if seed < 0 or seed > 2**32 - 1:
        raise ValidationError(
            "seed must be in [0, 2**32 - 1]",
            {"min": 0, "max": 2**32 - 1, "actual": seed},
        )
    return seed


def child_rng(root_seed: int, name: str) -> np.random.Generator:
    """Create a stable named stream from a root seed."""
    validate_seed(root_seed)
    if not isinstance(name, str) or not name:
        raise ValidationError("RNG stream name must be a non-empty string")
    root = np.random.SeedSequence([np.uint32(root_seed), np.uint32(19640409)])
    children = root.spawn(4)
    index = {
        STREAM_INIT: 0,
        STREAM_SUBSAMPLE: 1,
        STREAM_ORDER: 2,
        STREAM_NEGATIVE: 3,
    }[name]
    return np.random.default_rng(children[index])
