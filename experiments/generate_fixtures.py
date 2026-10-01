"""Deterministic local fixtures for reproduction and independent tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FIXTURE_VERSION = "1.0.0"
SEED = 20260930


def ar1_chains(
    *, n_chains: int = 4, n_draws: int = 5_000, phi: float = 0.8, seed: int = SEED
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n_chains, n_draws), dtype=float)
    for chain in range(n_chains):
        innovations = rng.normal(size=n_draws)
        out[chain, 0] = innovations[0]
        for draw in range(1, n_draws):
            out[chain, draw] = phi * out[chain, draw - 1] + innovations[draw]
    return out


def shifted_iid_chains(
    *,
    n_draws: int = 2_000,
    shifts: tuple[float, ...] = (-3.0, -1.0, 1.0, 3.0),
    seed: int = SEED + 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(len(shifts), n_draws)) + np.asarray(shifts)[:, None]


def constant_chains(*, n_draws: int = 100, constants=(1.0, 1.0, 2.0, 2.0)) -> np.ndarray:
    return np.tile(np.asarray(constants, dtype=float)[:, None], (1, n_draws))


def write_fixtures(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    fixtures = {
        "independent_noise.json": {
            "kind": "independent_noise",
            "seed": SEED,
            "chains": np.random.default_rng(SEED).normal(size=(4, 5_000)).tolist(),
        },
        "ar1_phi_0p8.json": {
            "kind": "ar1",
            "seed": SEED,
            "phi": 0.8,
            "expected_iat": (1.0 + 0.8) / (1.0 - 0.8),
            "chains": ar1_chains().tolist(),
        },
        "constant_and_misaligned.json": {
            "kind": "constant_chains_and_shifted_chains",
            "constants": constant_chains()[:, :20].tolist(),
            "shifted_iid": shifted_iid_chains(n_draws=2_000).tolist(),
        },
        "short.json": {
            "kind": "short_chains",
            "seed": SEED + 2,
            "chains": np.random.default_rng(SEED + 2).normal(size=(2, 5)).tolist(),
        },
    }
    for filename, payload in fixtures.items():
        payload["fixture_version"] = FIXTURE_VERSION
        path = output_dir / filename
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[filename] = path
    return paths


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    written = write_fixtures(base / "fixtures")
    for name, path in written.items():
        print(f"{name}: {path}")
