"""Weighted calibration statistics and Brier decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binning import assign_bins, resolve_edges
from .contracts import (
    AnalysisResult,
    BinStats,
    Decomposition,
    WeightSummary,
)


@dataclass(frozen=True)
class NumpySamples:
    y: np.ndarray
    p: np.ndarray
    w: np.ndarray

    @classmethod
    def from_contract(cls, batch) -> "NumpySamples":
        return cls(
            y=np.asarray(batch.y, dtype=np.int8),
            p=np.asarray(batch.p, dtype=float),
            w=np.asarray(batch.w if batch.w is not None else [1.0] * len(batch.y), dtype=float),
        )


def effective_sample_size(w: np.ndarray) -> float:
    """Kish effective sample size: (sum w)^2 / sum(w^2)."""
    total = float(np.sum(w))
    square_sum = float(np.sum(np.square(w)))
    if square_sum <= 0.0:
        return 0.0
    return total * total / square_sum


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    return float(np.sum(values * weights) / total)


def compute_analysis(
    samples: NumpySamples,
    method: str,
    requested_bins: int,
    run_id: str,
    explicit_edges: list[float] | None = None,
) -> AnalysisResult:
    edges = resolve_edges(samples.p, samples.w, method, requested_bins, explicit_edges)
    assignments = assign_bins(samples.p, edges)
    total_weight = float(np.sum(samples.w))
    total_count = int(samples.w.shape[0])
    zero_weight_count = int(np.sum(samples.w == 0.0))

    bin_records: list[BinStats] = []
    traces = [
        f"run_id={run_id}; method={method}; requested_bins={requested_bins}",
        (
            "weight rule: omitted weights default to 1; total_weight=sum(w); "
            "ess=(sum w)^2/sum(w^2)"
        ),
        (
            "bin rule: intervals are [lower, upper); final upper=1.0 is inclusive; "
            "equal-frequency cuts are never placed inside a tied probability"
        ),
    ]

    weighted_brier_sum = 0.0
    weighted_binned_sum = 0.0
    weighted_reliability_sum = 0.0
    weighted_resolution_sum = 0.0
    occupied_bins = 0
    base_rate = weighted_mean(samples.y, samples.w)
    mean_prediction = weighted_mean(samples.p, samples.w)

    for index in range(len(edges) - 1):
        selected = assignments == index
        count = int(np.sum(selected))
        bw = samples.w[selected]
        by = samples.y[selected]
        bp = samples.p[selected]
        bin_weight = float(np.sum(bw))
        bin_zero_weight = int(np.sum(bw == 0.0))
        weight_fraction = bin_weight / total_weight
        bin_ess = effective_sample_size(bw)

        if bin_weight <= 0.0:
            observed = 0.0
            prediction = 0.0
            brier = 0.0
            binned = 0.0
            empty = True
            traces.append(
                f"bin {index}: empty; interval=[{edges[index]!r}, {edges[index + 1]!r}); "
                "all weighted terms are 0"
            )
        else:
            occupied_bins += 1
            observed = weighted_mean(by, bw)
            prediction = weighted_mean(bp, bw)
            brier = float(np.sum(bw * np.square(bp - by)) / total_weight)
            binned = bin_weight * ((prediction - observed) ** 2 + observed * (1.0 - observed)) / total_weight
            weighted_brier_sum += float(np.sum(bw * np.square(bp - by)))
            weighted_binned_sum += bin_weight * (
                (prediction - observed) ** 2 + observed * (1.0 - observed)
            )
            weighted_reliability_sum += bin_weight * (prediction - observed) ** 2
            weighted_resolution_sum += bin_weight * (observed - base_rate) ** 2
            empty = False
            traces.append(
                f"bin {index}: n={count}; weight={bin_weight:.12g}; frac={weight_fraction:.12g}; "
                f"ess={bin_ess:.12g}; observed={observed:.12g}; "
                f"mean_prediction={prediction:.12g}; raw_brier_term={brier:.12g}"
            )

        bin_records.append(
            BinStats(
                bin_index=index,
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                zero_weight_count=bin_zero_weight,
                weight=bin_weight,
                weight_fraction=weight_fraction,
                effective_sample_size=bin_ess,
                observed_rate=observed,
                mean_prediction=prediction,
                brier=brier,
                binned_brier=binned,
                empty=empty,
            )
        )

    raw_brier = weighted_brier_sum / total_weight
    binned_brier = weighted_binned_sum / total_weight
    reliability = weighted_reliability_sum / total_weight
    resolution = weighted_resolution_sum / total_weight
    uncertainty = base_rate * (1.0 - base_rate)
    residual = raw_brier - binned_brier
    identity_residual = binned_brier - (reliability - resolution + uncertainty)

    decomposition = Decomposition(
        raw_brier=raw_brier,
        binned_brier=binned_brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        decomposition_residual=residual,
        identity_residual=identity_residual,
    )
    weight_summary = WeightSummary(
        count=total_count,
        total_weight=total_weight,
        effective_sample_size=effective_sample_size(samples.w),
        sum_weight_fractions=float(sum(item.weight_fraction for item in bin_records)),
    )
    traces.append(
        "judgment: raw_brier is mean weighted (p-y)^2; binned_brier replaces each "
        "forecast with its bin's weighted mean prediction"
    )
    traces.append(
        "judgment: decomposition_residual=raw_brier-binned_brier; "
        "identity_residual=binned_brier-(reliability-resolution+uncertainty)"
    )

    return AnalysisResult(
        run_id=run_id,
        method=method,
        requested_bins=requested_bins,
        occupied_bins=occupied_bins,
        empty_bins=len(bin_records) - occupied_bins,
        edges=[float(value) for value in edges],
        bins=bin_records,
        decomposition=decomposition,
        weights=weight_summary,
        base_rate=base_rate,
        mean_prediction=mean_prediction,
        calculation_trace=traces,
    )
