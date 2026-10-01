"""Pure NumPy/SciPy statistical diagnostics.

The effective sample size estimator follows the Stan/BDA3 Geyer initial
positive and monotone sequence rules. Autocorrelation shown to callers uses
the conventional biased autocovariance estimator; it is not summed naively to
calculate ESS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import fft, stats

DiagnosticLevel = Literal["ok", "warning", "inconclusive", "not_estimated"]


class DiagnosticInputError(ValueError):
    """A sample request violates the documented input contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class DiagnosticFinding:
    code: str
    level: DiagnosticLevel
    message: str
    scope: str = "analysis"

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "level": self.level,
            "scope": self.scope,
            "message": self.message,
        }


def validate_samples(chains: list[list[float]] | np.ndarray) -> np.ndarray:
    """Convert and validate an M x N equal-length sample matrix."""
    if isinstance(chains, list):
        if not chains or not all(isinstance(chain, list) and chain for chain in chains):
            raise DiagnosticInputError(
                "EMPTY_SAMPLES",
                "At least one chain with at least one draw is required.",
            )
        first_length = len(chains[0])
        if any(len(chain) != first_length for chain in chains):
            raise DiagnosticInputError(
                "EQUAL_LENGTH_REQUIRED",
                "All chains must contain the same number of draws.",
            )
    try:
        matrix = np.asarray(chains, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DiagnosticInputError(
            "NON_NUMERIC_SAMPLES", "All samples must be finite numeric values."
        ) from exc

    if matrix.ndim != 2:
        raise DiagnosticInputError(
            "INVALID_CHAIN_SHAPE",
            "Samples must be a non-empty list of equal-length non-empty chains.",
        )
    if matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise DiagnosticInputError(
            "EMPTY_SAMPLES", "At least one chain with at least one draw is required."
        )
    if not np.isfinite(matrix).all():
        raise DiagnosticInputError(
            "NON_FINITE_SAMPLE", "NaN and infinite values are not supported."
        )
    return matrix


def validate_max_lag(max_lag: int | None, n_draws: int, default_max_lag: int) -> int:
    if max_lag is None:
        return min(default_max_lag, n_draws - 1)
    if not isinstance(max_lag, int) or isinstance(max_lag, bool):
        raise DiagnosticInputError(
            "INVALID_MAX_LAG", "max_acf_lag must be an integer or null."
        )
    if max_lag < 0 or max_lag >= n_draws:
        raise DiagnosticInputError(
            "INVALID_MAX_LAG",
            f"max_acf_lag must satisfy 0 <= lag < {n_draws} for these chains.",
        )
    return max_lag


def biased_autocovariance(matrix: np.ndarray) -> np.ndarray:
    """Per-chain autocovariance for every lag, with no cross-chain pooling."""
    n_draws = matrix.shape[-1]
    padded_n = fft.next_fast_len(2 * n_draws)
    centered = matrix - matrix.mean(axis=-1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=padded_n, axis=-1)
    covariance = np.fft.irfft(
        spectrum * np.conjugate(spectrum), n=padded_n, axis=-1
    )
    return covariance[..., :n_draws] / n_draws


def chain_autocorrelation(
    chain: np.ndarray, max_lag: int
) -> tuple[list[float] | None, bool]:
    """ACF after within-chain demeaning and division by lag-zero variance."""
    if chain.size < 2:
        return None, True
    if np.ptp(chain) == 0.0:
        return None, True
    covariance = biased_autocovariance(chain[np.newaxis, :])[0]
    values = covariance / covariance[0]
    return [float(x) for x in values[: max_lag + 1]], False


def split_chains(matrix: np.ndarray) -> np.ndarray:
    """Split every chain into first and second halves and stack them."""
    half = matrix.shape[1] // 2
    return np.vstack((matrix[:, :half], matrix[:, -half:]))


def z_scale(matrix: np.ndarray) -> np.ndarray:
    """Blom rank-normal transformation applied to all chains jointly."""
    ranks = stats.rankdata(matrix, method="average", nan_policy="omit")
    c = 3.0 / 8.0
    transformed = (ranks - c) / (ranks.size - 2.0 * c + 1.0)
    return stats.norm.ppf(transformed).reshape(matrix.shape)


def geyer_ess(matrix: np.ndarray) -> dict[str, Any]:
    """Geyer ESS for an already transformed M x N matrix."""
    n_chains, n_draws = matrix.shape
    total_draws = n_chains * n_draws
    if n_chains < 1 or n_draws < 4:
        return {
            "value": None,
            "status": "not_estimated_short_chain",
            "reason": "Geyer ESS requires at least four draws in each analyzed batch.",
        }
    if np.ptp(matrix) == 0.0:
        return {
            "value": None,
            "status": "not_estimated_constant",
            "reason": "All analyzed values are identical, so autocorrelation and ESS are undefined.",
        }

    autocov = biased_autocovariance(matrix)
    chain_means = matrix.mean(axis=1)
    mean_within_variance = float(
        np.mean(autocov[:, 0]) * n_draws / (n_draws - 1.0)
    )
    var_plus = mean_within_variance * (n_draws - 1.0) / n_draws
    if n_chains > 1:
        var_plus += float(np.var(chain_means, ddof=1))
    if var_plus <= 0.0:
        return {
            "value": None,
            "status": "not_estimated_constant",
            "reason": "The pooled ESS variance denominator is zero.",
        }

    rho = np.zeros(n_draws, dtype=float)
    rho_even = 1.0
    rho[0] = rho_even
    rho_odd = 1.0 - (mean_within_variance - float(np.mean(autocov[:, 1]))) / var_plus
    rho[1] = rho_odd

    t = 1
    while t < (n_draws - 3) and (rho_even + rho_odd) > 0.0:
        rho_even = (
            1.0 - (mean_within_variance - float(np.mean(autocov[:, t + 1]))) / var_plus
        )
        rho_odd = (
            1.0 - (mean_within_variance - float(np.mean(autocov[:, t + 2]))) / var_plus
        )
        if rho_even + rho_odd >= 0.0:
            rho[t + 1] = rho_even
            rho[t + 2] = rho_odd
        t += 2

    max_t = t - 2
    if rho_even > 0.0:
        rho[max_t + 1] = rho_even

    t = 1
    while t <= max_t - 2:
        if (rho[t + 1] + rho[t + 2]) > (rho[t - 1] + rho[t]):
            rho[t + 1] = (rho[t - 1] + rho[t]) / 2.0
            rho[t + 2] = rho[t + 1]
        t += 2

    tau = -1.0 + 2.0 * np.sum(rho[: max_t + 1]) + np.sum(
        rho[max_t + 1 : max_t + 2]
    )
    tau = max(float(tau), 1.0 / np.log10(total_draws))
    return {
        "value": float(total_draws / tau),
        "status": "estimated",
        "integrated_autocorrelation_time": float(tau),
        "geyer_max_internal_index": int(max_t),
        "cutoff_lag": max(0, int(max_t) + 1),
        "variance_denominator": float(var_plus),
        "rho_hat_preview": [float(x) for x in rho[: min(11, rho.size)]],
        "rule": (
            "Geyer initial positive sequence followed by initial monotone sequence; "
            "rho is pooled across chains after within-chain demeaning."
        ),
    }


def variance_components(matrix: np.ndarray) -> dict[str, Any]:
    """Separate within-batch and between-batch variance components."""
    n_chains, n_draws = matrix.shape
    means = matrix.mean(axis=1)
    if n_draws >= 2:
        variances = np.var(matrix, axis=1, ddof=1)
        within = float(np.mean(variances))
    else:
        variances = None
        within = None

    between_scaled = (
        float(n_draws * np.var(means, ddof=1)) if n_chains >= 2 else None
    )
    if within is not None and between_scaled is not None:
        marginal = (n_draws - 1.0) / n_draws * within + between_scaled / n_draws
    else:
        marginal = within

    return {
        "batch_count": int(n_chains),
        "draws_per_batch": int(n_draws),
        "batch_means": [float(x) for x in means],
        "batch_variances": (
            None if variances is None else [float(x) for x in variances]
        ),
        "within_batch_variance_W": within,
        "between_batch_variance_B_scaled_by_n": between_scaled,
        "marginal_variance_estimate_V_hat": (
            None if marginal is None else float(marginal)
        ),
        "variance_rule": (
            "W is the mean of unbiased per-chain variances; B is n times the "
            "unbiased variance of chain means."
        ),
    }


def raw_rhat(matrix: np.ndarray) -> dict[str, Any]:
    """Classic potential scale reduction for equal-length batches."""
    n_chains, n_draws = matrix.shape
    if n_chains < 2 or n_draws < 2:
        return {
            "value": None,
            "extended_value": None,
            "status": "not_estimated_short_chain",
            "reason": "R-hat requires at least two batches and two draws per batch.",
        }

    components = variance_components(matrix)
    within = components["within_batch_variance_W"]
    between = components["between_batch_variance_B_scaled_by_n"]
    if within == 0.0 and between == 0.0:
        return {
            "value": None,
            "extended_value": None,
            "status": "not_estimated_constant",
            "reason": "Within- and between-batch variance are both zero.",
            "variance_components": components,
        }
    if within == 0.0:
        return {
            "value": None,
            "extended_value": "plus_infinity",
            "status": "inconclusive_zero_within_variance",
            "reason": "Batch means differ while every batch is internally constant.",
            "variance_components": components,
        }

    value = float(np.sqrt(((n_draws - 1.0) * within + between) / (n_draws * within)))
    return {
        "value": value,
        "extended_value": None,
        "status": "estimated",
        "reason": None,
        "variance_components": components,
    }


def rank_split_rhat(matrix: np.ndarray, *, require_original_chains: bool) -> dict[str, Any]:
    """Rank-normalized split R-hat, including a folded location check."""
    n_chains, n_draws = matrix.shape
    if require_original_chains and n_chains < 2:
        return {
            "value": None,
            "extended_value": None,
            "status": "not_estimated_single_chain",
            "reason": "Cross-chain rank R-hat needs at least two original chains.",
        }
    if n_draws < 4:
        return {
            "value": None,
            "extended_value": None,
            "status": "not_estimated_short_chain",
            "reason": "Split rank R-hat requires at least four original draws.",
        }

    split = split_chains(matrix)
    bulk = raw_rhat(z_scale(split))
    folded = np.abs(split - np.median(split))
    tail = raw_rhat(z_scale(folded))
    finite_values = [
        x["value"] for x in (bulk, tail) if x["value"] is not None
    ]
    has_infinity = any(x["extended_value"] == "plus_infinity" for x in (bulk, tail))
    if not finite_values and not has_infinity:
        return {
            "value": None,
            "extended_value": None,
            "status": bulk["status"],
            "reason": bulk["reason"],
            "bulk_rhat": bulk,
            "folded_rhat": tail,
        }

    status = "estimated"
    reason = None
    if has_infinity:
        status = "inconclusive_zero_within_variance"
        reason = "Rank batches have zero within-batch variance but different locations."
    return {
        "value": max(finite_values) if finite_values else None,
        "extended_value": "plus_infinity" if has_infinity else None,
        "status": status,
        "reason": reason,
        "bulk_rhat": bulk,
        "folded_rhat": tail,
    }


def analyze_chains(
    chains: list[list[float]] | np.ndarray,
    *,
    max_acf_lag: int | None = None,
    default_max_acf_lag: int = 50,
    rhat_warning_threshold: float = 1.01,
    ess_low_fraction: float = 0.10,
) -> dict[str, Any]:
    """Run ACF, ESS, variance-component, and R-hat diagnostics."""
    matrix = validate_samples(chains)
    n_chains, n_draws = matrix.shape
    max_lag = validate_max_lag(max_acf_lag, n_draws, default_max_acf_lag)
    findings: list[DiagnosticFinding] = []
    steps: list[dict[str, str]] = [
        {"step": "validate", "detail": f"accepted {n_chains} chains x {n_draws} draws."},
        {
            "step": "demean_and_normalize_acf",
            "detail": (
                "Each chain was de-meaned separately; ACF denominator is that chain's "
                "lag-zero biased autocovariance. ESS does not truncate raw ACF at this lag."
            ),
        },
    ]

    chain_reports = []
    for index, chain in enumerate(matrix):
        acf_values, constant_acf = chain_autocorrelation(chain, max_lag)
        is_constant = bool(np.ptp(chain) == 0.0)
        chain_report = {
            "chain_index": index,
            "draw_count": int(n_draws),
            "mean": float(np.mean(chain)),
            "variance_ddof1": None if n_draws < 2 else float(np.var(chain, ddof=1)),
            "is_constant": is_constant,
            "acf": {
                "values": acf_values,
                "requested_max_lag": max_lag,
                "normalization": "biased_autocovariance_within_chain",
                "lag_0": 1.0 if acf_values is not None else None,
                "not_estimated": constant_acf,
            },
        }
        if is_constant:
            findings.append(
                DiagnosticFinding(
                    "CONSTANT_CHAIN",
                    "inconclusive",
                    f"Chain {index} is constant; ACF and ESS information are undefined or degenerate.",
                    scope=f"chain:{index}",
                )
            )
        if n_draws < 4:
            findings.append(
                DiagnosticFinding(
                    "SHORT_CHAIN",
                    "warning",
                    f"Chain {index} has fewer than four draws; split and Geyer diagnostics are limited.",
                    scope=f"chain:{index}",
                )
            )
        chain_reports.append(chain_report)

    original_variance = variance_components(matrix)
    split_variance = None
    if n_draws >= 4:
        split_variance = variance_components(split_chains(matrix))

    ess_identity = geyer_ess(matrix)
    if n_draws >= 4:
        ess_mean_split = geyer_ess(split_chains(matrix))
        ess_bulk_split_rank = geyer_ess(z_scale(split_chains(matrix)))
    else:
        missing_ess = {
            "value": None,
            "status": "not_estimated_short_chain",
            "reason": "Split diagnostics require at least four original draws.",
        }
        ess_mean_split = dict(missing_ess)
        ess_bulk_split_rank = dict(missing_ess)

    if n_chains == 1:
        findings.append(
            DiagnosticFinding(
                "SINGLE_CHAIN_CROSS_CHAIN_UNAVAILABLE",
                "warning",
                "One chain cannot diagnose disagreement between independent chain trajectories.",
            )
        )
        classic_rhat = {
            "value": None,
            "extended_value": None,
            "status": "not_estimated_single_chain",
            "reason": "Classic R-hat requires at least two chains.",
        }
        rank_rhat = {
            "value": None,
            "extended_value": None,
            "status": "not_estimated_single_chain",
            "reason": "Cross-chain rank R-hat requires at least two original chains.",
        }
        within_split_rhat = rank_split_rhat(
            matrix, require_original_chains=False
        )
        if within_split_rhat["status"] == "estimated":
            within_split_rhat["interpretation"] = (
                "A within-chain split diagnostic; it cannot replace disagreement checks "
                "between independently initialized chains."
            )
    else:
        classic_rhat = raw_rhat(matrix)
        rank_rhat = rank_split_rhat(matrix, require_original_chains=True)
        within_split_rhat = None

    total_draws = n_chains * n_draws
    for name, result in (
        ("ESS_IDENTITY", ess_identity),
        ("ESS_MEAN_SPLIT", ess_mean_split),
        ("ESS_BULK_SPLIT_RANK", ess_bulk_split_rank),
    ):
        if result["status"].startswith("not_estimated_constant"):
            findings.append(
                DiagnosticFinding(
                    f"{name}_NOT_ESTIMATED_CONSTANT",
                    "inconclusive",
                    result["reason"],
                )
            )
        elif result["status"].startswith("not_estimated_short"):
            findings.append(
                DiagnosticFinding(
                    f"{name}_NOT_ESTIMATED_SHORT",
                    "warning",
                    result["reason"],
                )
            )
        elif (
            result.get("value") is not None
            and result["value"] / total_draws < ess_low_fraction
        ):
            findings.append(
                DiagnosticFinding(
                    f"{name}_LOW",
                    "warning",
                    (
                        f"{name} is {result['value']:.3g}, below "
                        f"{ess_low_fraction:.0%} of {total_draws} nominal draws."
                    ),
                )
            )

    rhat_for_threshold = rank_rhat.get("value")
    if rhat_for_threshold is not None and rhat_for_threshold >= rhat_warning_threshold:
        findings.append(
            DiagnosticFinding(
                "RHAT_ABOVE_THRESHOLD",
                "warning",
                (
                    f"Rank split R-hat {rhat_for_threshold:.4f} is at or above "
                    f"{rhat_warning_threshold:.2f}; chain/batch disagreement is indicated."
                ),
            )
        )
    if rank_rhat.get("extended_value") == "plus_infinity":
        findings.append(
            DiagnosticFinding(
                "RHAT_PLUS_INFINITY_CONSTANT_BATCHES",
                "inconclusive",
                rank_rhat.get("reason") or "Constant batches have non-finite R-hat.",
            )
        )

    levels = [f.level for f in findings]
    if "inconclusive" in levels:
        overall_status = "inconclusive"
    elif "warning" in levels:
        overall_status = "warning"
    else:
        overall_status = "ok"

    steps.extend(
        [
            {
                "step": "separate_batch_variance",
                "detail": "Computed per-chain means, per-chain unbiased variances, W, B, and V-hat.",
            },
            {
                "step": "ess_truncation",
                "detail": ess_identity.get("rule", "ESS was not estimated."),
            },
            {
                "step": "convergence_diagnostic",
                "detail": "Computed classic and split rank R-hat where supported.",
            },
        ]
    )

    return {
        "schema_version": "mcmc-diagnostics/v1",
        "status": overall_status,
        "input_summary": {
            "chain_count": int(n_chains),
            "draws_per_chain": int(n_draws),
            "nominal_sample_count": int(total_draws),
        },
        "chains": chain_reports,
        "autocorrelation": {
            "demeaning": "within_chain_sample_mean",
            "covariance_estimator": "biased_1_over_n",
            "normalization": "divide_by_lag_zero_within_chain_covariance",
            "report_max_lag": max_lag,
            "note": "This display lag cutoff is separate from Geyer ESS truncation.",
        },
        "variance": {
            "original_batches": original_variance,
            "split_batches": split_variance,
        },
        "effective_sample_size": {
            "identity_original_chains": ess_identity,
            "mean_split_chains": ess_mean_split,
            "bulk_split_rank_normalized": ess_bulk_split_rank,
            "capping_policy": "Not capped at nominal sample count; finite-sample anti-correlation can exceed N.",
        },
        "convergence": {
            "classic_original_chains": classic_rhat,
            "rank_split_chains": rank_rhat,
            "within_chain_rank_split_for_single_chain": within_split_rhat,
            "warning_threshold": rhat_warning_threshold,
        },
        "findings": [f.as_dict() for f in findings],
        "uncertainty_conclusions": [
            "R-hat near 1 and non-degenerate ESS do not prove convergence; they fail to detect strong evidence of mixing failure in these samples.",
            "Diagnostics can miss multimodality shared by all chains or nonstationarity occurring after the supplied window.",
            "Short or constant chains reduce identifiability; inspect trace plots and use dispersed initial states.",
        ],
        "limitations": [
            "Only univariate scalar chains are supported.",
            "Input values are assumed ordered in time and independently initialized across chains.",
            "Geyer ESS estimates integrated autocorrelation and remains noisy in finite samples.",
        ],
        "key_steps": steps,
    }
