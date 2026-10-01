"""Independent exact reference implementation using fractions.

This module deliberately does not import the production NumPy core. It provides
an independently written oracle for tests and reproducible fixture experiments.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable


def _fraction(value: object) -> Fraction:
    return Fraction(str(value))


def fixed_reference_edges(n_bins: int) -> list[Fraction]:
    return [Fraction(k, n_bins) for k in range(n_bins + 1)]


def quantile_reference_edges(probabilities: list[Fraction], weights: list[Fraction], n_bins: int) -> list[Fraction]:
    grouped: dict[Fraction, Fraction] = {}
    for probability, weight in zip(probabilities, weights):
        grouped[probability] = grouped.get(probability, Fraction(0)) + weight
    values = sorted(grouped)
    positive = [(value, grouped[value]) for value in values if grouped[value] > 0]
    total = sum((weight for _, weight in positive), Fraction(0))
    cumulative = []
    running = Fraction(0)
    for _, weight in positive:
        running += weight
        cumulative.append(running)

    edges = [Fraction(0)]
    for k in range(1, n_bins + 1):
        if k == n_bins:
            edges.append(Fraction(1))
            continue
        target = total * k / n_bins
        index = next(i for i, mass in enumerate(cumulative) if mass >= target)
        if index >= len(positive) - 1:
            edges.append(Fraction(1))
        else:
            left = positive[index][0]
            right = positive[index + 1][0]
            edges.append((left + right) / 2)
    return edges


def assign_reference_bins(probabilities: Iterable[Fraction], edges: list[Fraction]) -> list[int]:
    cuts = edges[1:-1]
    assigned = []
    for probability in probabilities:
        index = 0
        for cut in cuts:
            if probability >= cut:
                index += 1
            else:
                break
        assigned.append(index)
    return assigned


def weighted_brier_reference(
    y: list[int], p: list[Fraction], w: list[Fraction], edges: list[Fraction]
) -> dict[str, Fraction]:
    total = sum(w, Fraction(0))
    labels = assign_reference_bins(p, edges)
    base_rate = sum((outcome * weight for outcome, weight in zip(y, w)), Fraction(0)) / total
    mean_prediction = sum((probability * weight for probability, weight in zip(p, w)), Fraction(0)) / total

    raw = Fraction(0)
    binned = Fraction(0)
    reliability = Fraction(0)
    resolution = Fraction(0)
    bins = []

    for bin_index in range(len(edges) - 1):
        selected = [i for i, label in enumerate(labels) if label == bin_index]
        count = len(selected)
        weight = sum((w[i] for i in selected), Fraction(0))
        if weight == 0:
            bins.append(
                {
                    "bin_index": bin_index,
                    "count": count,
                    "weight": weight,
                    "weight_fraction": weight / total,
                    "observed_rate": Fraction(0),
                    "mean_prediction": Fraction(0),
                    "empty": True,
                }
            )
            continue

        observed = sum((y[i] * w[i] for i in selected), Fraction(0)) / weight
        prediction = sum((p[i] * w[i] for i in selected), Fraction(0)) / weight
        raw += sum((w[i] * (p[i] - y[i]) ** 2 for i in selected), Fraction(0))
        binned += weight * ((prediction - observed) ** 2 + observed * (1 - observed))
        reliability += weight * (prediction - observed) ** 2
        resolution += weight * (observed - base_rate) ** 2
        bins.append(
            {
                "bin_index": bin_index,
                "count": count,
                "weight": weight,
                "weight_fraction": weight / total,
                "observed_rate": observed,
                "mean_prediction": prediction,
                "empty": False,
            }
        )

    raw /= total
    binned /= total
    reliability /= total
    resolution /= total
    uncertainty = base_rate * (1 - base_rate)
    return {
        "bins": bins,
        "total_weight": total,
        "ess": total**2 / sum((weight * weight for weight in w), Fraction(0)),
        "base_rate": base_rate,
        "mean_prediction": mean_prediction,
        "raw_brier": raw,
        "binned_brier": binned,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "decomposition_residual": raw - binned,
        "identity_residual": binned - (reliability - resolution + uncertainty),
        "sum_weight_fractions": sum((item["weight_fraction"] for item in bins), Fraction(0)),
    }


def reference_from_fixture(fixture: dict) -> dict[str, Fraction]:
    samples = fixture["samples"]
    y = [int(value) for value in samples["y"]]
    p = [_fraction(value) for value in samples["p"]]
    w = (
        [_fraction(value) for value in samples["w"]]
        if samples.get("w") is not None
        else [Fraction(1) for _ in y]
    )
    if "edges" in fixture:
        edges = [_fraction(value) for value in fixture["edges"]]
    else:
        edges = quantile_reference_edges(p, w, int(fixture["n_bins"]))
    result = weighted_brier_reference(y, p, w, edges)
    result["edges"] = edges
    return result
