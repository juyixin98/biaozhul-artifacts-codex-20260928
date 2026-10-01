"""Statistical kernel.

The application kernel depends only on NumPy/SciPy and the package schema.
Autocorrelation shown to callers uses the conventional biased per-chain
autocovariance estimator. ESS uses finite-sample variance components and
Geyer truncation; those are related but not identical calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.stats import norm, rankdata

from .config import Settings, get_settings
from .models import (
    ACFResult,
    AutocovSummary,
    Diagnostic,
    ESSResult,
    MultiChainRequest,
    MultiChainResult,
    RhatResult,
    SingleChainRequest,
    SingleChainResult,
    VarianceComponents,
)
from .versioning import request_id_from_payload, version_payload


@dataclass(frozen=True)
class ESSInternals:
    ess: float
    tau: float
    cutoff_lag: int
    reached_end: bool


def _diag(
    code: str,
    level: str,
    message: str,
    scope: str = "general",
    uncertainty: bool = False,
) -> Diagnostic:
    return Diagnostic(
        code=code,
        level=level,  # type: ignore[arg-type]
        message=message,
        scope=scope,
        uncertainty=uncertainty,
    )


def default_max_lag(n: int, settings: Settings) -> int:
    return min(
        n - 1,
        max(
            settings.default_acf_min_lags,
            math.ceil(settings.default_acf_factor * math.log10(max(n, 1))),
        ),
    )


def autocovariance_chain(values: np.ndarray) -> np.ndarray:
    """Return C[t] = sum_i (x_i-m)(x_i+t-m)/n for one chain."""

    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("empty chain")
    centered = x - x.mean()
    if n == 1:
        return np.asarray([float(centered[0] ** 2)])
    fft_size = next_fast_len(2 * n, real=True)
    spectrum = rfft(centered, n=fft_size)
    full = irfft(spectrum * np.conjugate(spectrum), n=fft_size)
    return full[:n] / n


def _acf_result(
    chain_id: str, values: np.ndarray, requested_lag: int | None, settings: Settings
):
    n = values.size
    desired = default_max_lag(n, settings) if requested_lag is None else requested_lag
    clamped = max(0, min(desired, n - 1))
    cov = autocovariance_chain(values)
    constant = bool(cov[0] == 0.0)
    if constant:
        acf_values: list[float | None] = [None] * (clamped + 1)
    else:
        acf_values = [float(v) for v in (cov[: clamped + 1] / cov[0])]
    result = ACFResult(
        chain_id=chain_id,
        n=n,
        mean=float(values.mean()),
        max_lag=clamped,
        lags=list(range(clamped + 1)),
        acf=acf_values,
        constant=constant,
        method="per_chain_centered_biased_autocovariance_divided_by_lag_zero",
    )
    return result, cov, clamped, desired != clamped


def _geyer_ess(matrix: np.ndarray) -> ESSInternals:
    """Initial positive sequence followed by initial monotone sequence."""

    x = np.asarray(matrix, dtype=float)
    m, n = x.shape
    acov = np.vstack([autocovariance_chain(row) for row in x])
    chain_means = x.mean(axis=1)

    within = float(np.mean(acov[:, 0]) * n / (n - 1.0)) if n > 1 else 0.0
    var_plus = float(np.mean(acov[:, 0]))
    if m > 1:
        var_plus += float(np.var(chain_means, ddof=1))

    rho = np.zeros(n, dtype=float)
    rho[0] = 1.0
    if n > 1:
        rho[1] = 1.0 - (within - float(np.mean(acov[:, 1]))) / var_plus

    t = 1
    while t < (n - 3) and (rho[t - 1] + rho[t]) > 0.0:
        rho[t + 1] = 1.0 - (within - float(np.mean(acov[:, t + 1]))) / var_plus
        rho[t + 2] = 1.0 - (within - float(np.mean(acov[:, t + 2]))) / var_plus
        if (rho[t + 1] + rho[t + 2]) < 0.0:
            rho[t + 1] = 0.0
            rho[t + 2] = 0.0
        t += 2

    max_t = t - 2

    t = 1
    while t <= max_t - 2:
        previous_pair = rho[t - 1] + rho[t]
        if (rho[t + 1] + rho[t + 2]) > previous_pair:
            rho[t + 1] = previous_pair / 2.0
            rho[t + 2] = previous_pair / 2.0
        t += 2

    if max_t + 1 < n and rho[max_t + 1] > 0.0:
        rho[max_t + 1] = rho[max_t + 1]

    tau = -1.0 + 2.0 * np.sum(rho[: max_t + 1]) + float(np.sum(rho[max_t + 1 : max_t + 2]))
    tau = max(float(tau), 1.0 / math.log10(m * n))
    ess = float(m * n / tau)
    return ESSInternals(
        ess=ess,
        tau=float(tau),
        cutoff_lag=max(0, max_t + 1),
        reached_end=(max_t + 2 >= n - 2),
    )


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).ravel()
    ranks = rankdata(flat, method="average")
    transformed = (ranks - 3.0 / 8.0) / (flat.size - 2.0 * 3.0 / 8.0 + 1.0)
    return norm.ppf(transformed).reshape(values.shape)


def _split_chains(values: np.ndarray) -> np.ndarray:
    half = values.shape[1] // 2
    return np.vstack([values[:, :half], values[:, -half:]])


def _ess_result(
    kind: str,
    matrix: np.ndarray | None,
    total_draws: int,
    settings: Settings,
    raw_series_constant: bool = False,
) -> tuple[ESSResult | None, ESSInternals | None, list[Diagnostic]]:
    truncation_rule = (
        "scan complete lag pairs from lag 0/1; stop at first non-positive pair; "
        "apply Geyer monotone pair smoothing; tau=-1+2*sum(retained rho)"
    )
    if matrix is None:
        return (
            ESSResult(
                kind=kind,  # type: ignore[arg-type]
                ess=None,
                total_draws=total_draws,
                relative_ess=None,
                integrated_time=None,
                cutoff_lag=None,
                truncation_rule=truncation_rule,
                state="not_estimated",
            ),
            None,
            [],
        )

    m, n = matrix.shape
    if n < settings.min_ess_draws:
        return (
            ESSResult(
                kind=kind,  # type: ignore[arg-type]
                ess=None,
                total_draws=m * n,
                relative_ess=None,
                integrated_time=None,
                cutoff_lag=None,
                truncation_rule="Geyer pair truncation not applied to a too-short series",
                state="not_estimated",
            ),
            None,
            [
                _diag(
                    "SHORT_CHAIN_FOR_ESS",
                    "warning",
                    f"{kind} ESS needs at least {settings.min_ess_draws} draws after any split; "
                    f"received {n}. ESS is omitted because truncation is unstable.",
                    scope=f"ess.{kind}",
                    uncertainty=True,
                )
            ],
        )

    if raw_series_constant or bool(np.ptp(matrix) == 0.0):
        return (
            ESSResult(
                kind=kind,  # type: ignore[arg-type]
                ess=None,
                total_draws=m * n,
                relative_ess=None,
                integrated_time=None,
                cutoff_lag=None,
                truncation_rule="undefined for zero global variance",
                state="undefined",
            ),
            None,
            [
                _diag(
                    "CONSTANT_SERIES_FOR_ESS",
                    "inconclusive",
                    f"{kind} ESS is undefined because every source series is constant "
                    "(or the transformed diagnostic series has zero variance). "
                    "Constant output must not be labelled as N independent draws.",
                    scope=f"ess.{kind}",
                    uncertainty=True,
                )
            ],
        )

    internals = _geyer_ess(matrix)
    relative = internals.ess / (m * n)
    diagnostics: list[Diagnostic] = []
    if internals.reached_end:
        diagnostics.append(
            _diag(
                "ESS_TRUNCATION_AT_SERIES_END",
                "warning",
                f"{kind} ESS remained positive through every usable lag pair. "
                "The estimate may be unstable; use a longer run.",
                scope=f"ess.{kind}",
                uncertainty=True,
            )
        )
    if relative < settings.very_low_relative_ess:
        diagnostics.append(
            _diag(
                "ESS_SEVERE_DEGRADATION",
                "error",
                f"{kind} relative ESS is below {settings.very_low_relative_ess:.0%}.",
                scope=f"ess.{kind}",
            )
        )
    elif relative < settings.low_relative_ess:
        diagnostics.append(
            _diag(
                "ESS_LOW",
                "warning",
                f"{kind} relative ESS is below {settings.low_relative_ess:.0%}.",
                scope=f"ess.{kind}",
            )
        )

    return (
        ESSResult(
            kind=kind,  # type: ignore[arg-type]
            ess=internals.ess,
            total_draws=m * n,
            relative_ess=relative,
            integrated_time=internals.tau,
            cutoff_lag=internals.cutoff_lag,
            truncation_rule=truncation_rule,
            state="estimated",
        ),
        internals,
        diagnostics,
    )


def _variance_components(
    matrix: np.ndarray | list[np.ndarray], equal_length: bool
) -> VarianceComponents:
    rows = list(matrix)
    m = len(rows)
    lengths = [int(row.size) for row in rows]
    if equal_length and len(set(lengths)) == 1:
        n = lengths[0]
        matrix = np.asarray(matrix)
        means = matrix.mean(axis=1)
        variances = (
            matrix.var(axis=1, ddof=1)
            if n > 1
            else np.zeros(m, dtype=float)
        )
        within = float(variances.mean())
        between = float(n * np.var(means, ddof=1)) if m > 1 else None
        marginal = float((n - 1.0) / n * within + between / n) if between is not None else None
        return VarianceComponents(
            n_chains=m,
            draws_per_chain=n,
            equal_length=True,
            within_chain_variance=within,
            between_chain_variance=between,
            marginal_variance=marginal,
            degrees_of_freedom_between=m - 1 if m > 1 else None,
            degrees_of_freedom_within=m * (n - 1),
            formula=(
                "W=mean(s_i^2); B=n*var(chain means,ddof=1); "
                "Vhat=(n-1)/n*W+B/n"
            ),
        )

    n_total = sum(lengths)
    grand = float(np.average([row.mean() for row in rows], weights=lengths))
    weighted_between_sum = float(
        sum(lengths[i] * float((rows[i].mean() - grand) ** 2) for i in range(m))
    )
    within_sum = float(
        sum(float(np.sum((rows[i] - rows[i].mean()) ** 2)) for i in range(m))
    )
    within = within_sum / (n_total - m) if n_total > m else 0.0
    if m > 1:
        # Gelman unequal-n ANOVA component. It can be negative for short,
        # unbalanced chains; report it instead of coercing to zero.
        harmonic_draws = (
            n_total - sum(length * length for length in lengths) / n_total
        ) / (m - 1)
        mean_square_between = weighted_between_sum / (m - 1)
        between = (mean_square_between - within) / harmonic_draws
    else:
        between = None
    pooled = float(
        (within_sum + weighted_between_sum) / (n_total - 1) if n_total > 1 else None
    )
    return VarianceComponents(
        n_chains=m,
        draws_per_chain=None,
        equal_length=False,
        within_chain_variance=within,
        between_chain_variance=between,
        marginal_variance=pooled,
        degrees_of_freedom_between=m - 1 if m > 1 else None,
        degrees_of_freedom_within=n_total - m,
        formula=(
            "unequal-length weighted ANOVA components; pooled variance is shown but the "
            "equal-length R-hat/ESS estimator is not applied"
        ),
    )


def _classic_rhat_from_equal(
    matrix: np.ndarray, kind: str, threshold: float
) -> tuple[RhatResult, list[Diagnostic]]:
    m, n = matrix.shape
    if m < 2:
        return (
            RhatResult(
                kind=kind,  # type: ignore[arg-type]
                rhat=None,
                state="not_estimated",
                reason="at_least_two_series_required",
                threshold=threshold,
            ),
            [],
        )
    means = matrix.mean(axis=1)
    variances = matrix.var(axis=1, ddof=1)
    within = float(variances.mean())
    between = float(n * np.var(means, ddof=1))
    if within == 0.0:
        reason = (
            "global_constant"
            if between == 0.0
            else "zero_within_variance_nonzero_between_variance"
        )
        return (
            RhatResult(
                kind=kind,  # type: ignore[arg-type]
                rhat=None,
                state="undefined",
                reason=reason,
                threshold=threshold,
            ),
            [
                _diag(
                    "RHAT_ZERO_WITHIN_VARIANCE",
                    "inconclusive",
                    "R-hat is undefined because within-series variance is zero. "
                    "Differing constant series indicate divergence rather than a finite R-hat.",
                    scope=f"rhat.{kind}",
                    uncertainty=True,
                )
            ],
        )
    marginal = (n - 1.0) / n * within + between / n
    return (
        RhatResult(
            kind=kind,  # type: ignore[arg-type]
            rhat=float(math.sqrt(marginal / within)),
            state="estimated",
            reason=None,
            threshold=threshold,
        ),
        [],
    )


def _rhat_set(
    matrix: np.ndarray, equal_length: bool, settings: Settings
) -> tuple[list[RhatResult], list[Diagnostic]]:
    m, n = matrix.shape
    diagnostics: list[Diagnostic] = []
    if not equal_length:
        results = []
        for kind in ("classic", "split", "rank_normalized"):
            results.append(
                RhatResult(
                    kind=kind,  # type: ignore[arg-type]
                    rhat=None,
                    state="not_estimated",
                    reason="unequal_length_input_requires_balanced_estimator",
                    threshold=settings.rhat_warning_threshold,
                )
            )
        diagnostics.append(
            _diag(
                "UNEQUAL_LENGTH_NO_AGGREGATE_RHAT",
                "warning",
                "Batch means are still separated, but R-hat/ESS aggregation requires balanced "
                "chain lengths and was not silently approximated.",
                scope="rhat",
                uncertainty=True,
            )
        )
        return results, diagnostics

    if m < settings.min_rhat_chains:
        results = []
        for kind in ("classic", "split", "rank_normalized"):
            results.append(
                RhatResult(
                    kind=kind,  # type: ignore[arg-type]
                    rhat=None,
                    state="not_estimated",
                    reason="at_least_two_chains_required",
                    threshold=settings.rhat_warning_threshold,
                )
            )
        diagnostics.append(
            _diag(
                "SINGLE_CHAIN_NO_BETWEEN_CHAIN_RHAT",
                "info",
                "Between-chain R-hat requires at least two chains; one-chain ACF/ESS cannot "
                "establish mixing across dispersed initializations.",
                scope="rhat",
                uncertainty=True,
            )
        )
        return results, diagnostics

    if n < settings.min_split_draws:
        results = []
        for kind in ("classic", "split", "rank_normalized"):
            results.append(
                RhatResult(
                    kind=kind,  # type: ignore[arg-type]
                    rhat=None,
                    state="not_estimated",
                    reason="at_least_four_draws_required",
                    threshold=settings.rhat_warning_threshold,
                )
            )
        diagnostics.append(
            _diag(
                "SHORT_CHAIN_FOR_RHAT",
                "warning",
                f"R-hat needs at least {settings.min_split_draws} draws; received {n}.",
                scope="rhat",
                uncertainty=True,
            )
        )
        return results, diagnostics

    classic, classic_diags = _classic_rhat_from_equal(
        matrix, "classic", settings.rhat_warning_threshold
    )
    split_matrix = _split_chains(matrix)
    split, split_diags = _classic_rhat_from_equal(
        split_matrix, "split", settings.rhat_warning_threshold
    )

    rank_split = _rank_normalize(split_matrix)
    bulk_rank, _ = _classic_rhat_from_equal(
        rank_split, "rank_normalized", settings.rhat_warning_threshold
    )
    folded = np.abs(split_matrix - np.median(split_matrix))
    tail_rank, _ = _classic_rhat_from_equal(
        _rank_normalize(folded), "rank_normalized", settings.rhat_warning_threshold
    )
    if bulk_rank.rhat is not None and tail_rank.rhat is not None:
        rank = RhatResult(
            kind="rank_normalized",
            rhat=float(max(bulk_rank.rhat, tail_rank.rhat)),
            state="estimated",
            reason=None,
            threshold=settings.rhat_warning_threshold,
        )
    else:
        rank = RhatResult(
            kind="rank_normalized",
            rhat=None,
            state="undefined",
            reason=bulk_rank.reason or tail_rank.reason,
            threshold=settings.rhat_warning_threshold,
        )

    results = [classic, split, rank]
    diagnostics.extend(classic_diags)
    diagnostics.extend(split_diags)
    for result in results:
        if result.rhat is not None and result.rhat > settings.rhat_warning_threshold:
            diagnostics.append(
                _diag(
                    "RHAT_ABOVE_THRESHOLD",
                    "warning",
                    f"{result.kind} R-hat {result.rhat:.4f} exceeds "
                    f"{settings.rhat_warning_threshold:.2f}; batch means are not mutually consistent.",
                    scope=f"rhat.{result.kind}",
                    uncertainty=True,
                )
            )
    return results, diagnostics


def _overall_status(diagnostics: list[Diagnostic]) -> str:
    levels = {item.level for item in diagnostics}
    if "error" in levels:
        return "error"
    if "inconclusive" in levels:
        return "inconclusive"
    if "warning" in levels:
        return "warning"
    return "ok"


def _deduplicate_diagnostics(diagnostics: list[Diagnostic]) -> list[Diagnostic]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[Diagnostic] = []
    for diagnostic in diagnostics:
        key = (diagnostic.code, diagnostic.scope, diagnostic.message)
        if key not in seen:
            seen.add(key)
            unique.append(diagnostic)
    return unique


def _common_limitations() -> list[str]:
    return [
        "R-hat near 1 and high ESS do not prove convergence; they only fail to detect selected "
        "mixing problems in the supplied finite draws.",
        "ESS assumes a weakly stationary within-chain target and can be misleading under "
        "nonstationarity, multimodality, strong trends, or initialization burn-in.",
        "Autocorrelation and ESS diagnostics do not replace posterior validation, trace plots, "
        "divergence/gradient diagnostics, or adequate dispersed starts.",
        "Geyer truncation is an estimator of integrated autocorrelation time; long-memory or "
        "too-short runs can make its uncertainty substantial.",
    ]


def analyze_single_chain(
    request: SingleChainRequest,
    settings: Settings | None = None,
    payload: dict | None = None,
) -> SingleChainResult:
    settings = settings or get_settings()
    request_id = request_id_from_payload(payload or request.model_dump(), request.request_id)
    values = np.asarray(request.values, dtype=float)
    diagnostics: list[Diagnostic] = []
    steps = [
        "validated finite one-dimensional sample contract",
        "subtracted the chain mean before autocovariance",
        "normalized each autocovariance by the same-chain lag-zero variance",
        "applied ESS Geyer lag-pair truncation rather than assuming draws independent",
    ]

    acf_results = []
    if request.include_acf:
        acf, _cov, reported_lag, clamped = _acf_result(
            "chain-0", values, request.max_lag, settings
        )
        if request.max_lag is not None and request.max_lag > values.size - 1:
            diagnostics.append(
                _diag(
                    "MAX_LAG_CLAMPED",
                    "info",
                    f"Requested ACF lag {request.max_lag} exceeds n-1; reported lag is {reported_lag}.",
                    scope="acf",
                )
            )
        if values.size == 1:
            diagnostics.append(
                _diag(
                    "ACF_ONLY_ZERO_LAG",
                    "info",
                    "Only lag zero can be reported for a one-draw chain.",
                    scope="acf",
                )
            )
        acf_results.append(acf)

    constant = bool(np.ptp(values) == 0.0)
    if constant:
        diagnostics.append(
            _diag(
                "CONSTANT_CHAIN",
                "inconclusive",
                "The chain is constant. Autocorrelation after lag zero, ESS, and mixing cannot "
                "be estimated; its sample count is not effective sample size.",
                scope="sample",
                uncertainty=True,
            )
        )
    if values.size < settings.min_ess_draws:
        diagnostics.append(
            _diag(
                "SHORT_CHAIN",
                "warning",
                f"Chain has {values.size} draw(s); at least {settings.min_ess_draws} are required "
                "for this service's stable ESS truncation.",
                scope="sample",
                uncertainty=True,
            )
        )

    matrix = values.reshape(1, -1)
    mean_ess, _, mean_diags = _ess_result(
        "mean",
        None if constant or values.size < settings.min_ess_draws else matrix,
        values.size,
        settings,
        raw_series_constant=constant,
    )
    can_split = values.size >= max(settings.min_ess_draws, settings.min_split_draws)
    if can_split:
        split = _split_chains(matrix)
        split_mean_ess, _, split_mean_diags = _ess_result(
            "split_mean",
            None if constant else split,
            values.size,
            settings,
            raw_series_constant=constant,
        )
        # Split bulk combines two halves of one source chain. It checks
        # within-run stationarity but not mixing across separate chains.
        bulk_input = _rank_normalize(split)
    else:
        split = None
        bulk_input = None
        split_mean_ess = None
        split_mean_diags = []
    bulk_ess, _, bulk_diags = _ess_result(
        "bulk",
        None if not can_split or constant else bulk_input,
        values.size,
        settings,
        raw_series_constant=constant,
    )
    diagnostics.extend(mean_diags)
    diagnostics.extend(split_mean_diags)
    diagnostics.extend(bulk_diags)

    rhat, rhat_diags = _rhat_set(matrix, True, settings)
    diagnostics.extend(rhat_diags)

    components = _variance_components(matrix, True)
    diagnostics = _deduplicate_diagnostics(diagnostics)
    return SingleChainResult(
        request_id=request_id,
        status=_overall_status(diagnostics),  # type: ignore[arg-type]
        n_chains=1,
        total_draws=int(values.size),
        draws=[int(values.size)],
        autocovariance=AutocovSummary(
            lag_cutoff_rule=(
                "displayed ACF uses min(requested or default 10log10 rule,n-1); "
                "display cutoff never zeroes later lags"
            ),
            reported_max_lag=acf_results[0].max_lag if acf_results else 0,
            ess_scanned_lags=max(0, values.size - 1),
            centered=True,
            denominator="each chain C[0]; null when C[0]=0",
        ),
        acf=acf_results,
        ess=[
            item
            for item in (mean_ess, split_mean_ess, bulk_ess)
            if item is not None
        ],
        variance_components=components,
        rhat=rhat,
        diagnostics=diagnostics,
        limitations=_common_limitations()
        + [
            "A single chain has no genuine between-chain variance; split-chain diagnostics may "
            "miss failure to reach a separate mode."
        ],
        steps=steps,
        versions=version_payload(settings),
    )


def analyze_chains(
    request: MultiChainRequest,
    settings: Settings | None = None,
    payload: dict | None = None,
) -> MultiChainResult:
    settings = settings or get_settings()
    request_id = request_id_from_payload(payload or request.model_dump(), request.request_id)
    chain_ids = request.chain_ids or [f"chain-{i}" for i in range(len(request.chains))]
    arrays = [np.asarray(chain, dtype=float) for chain in request.chains]
    lengths = [int(array.size) for array in arrays]
    equal_length = len(set(lengths)) == 1
    diagnostics: list[Diagnostic] = []
    steps = [
        "validated finite multi-chain contract and chain identifiers",
        "computed each chain mean separately and centered within that chain",
        "distinguished within-chain variance from variance of chain means",
        "used Geyer autocorrelation truncation for ESS and separate classic/split/rank R-hat",
    ]

    acf_results = []
    if request.include_acf:
        for chain_id, array in zip(chain_ids, arrays):
            acf, _cov, reported_lag, clamped = _acf_result(
                chain_id, array, request.max_lag, settings
            )
            acf_results.append(acf)
            if request.max_lag is not None and request.max_lag > array.size - 1:
                diagnostics.append(
                    _diag(
                        "MAX_LAG_CLAMPED",
                        "info",
                        f"{chain_id}: requested lag {request.max_lag} exceeds n-1; "
                        f"reported lag is {reported_lag}.",
                        scope=f"acf.{chain_id}",
                    )
                )

    constant_flags = [bool(np.ptp(array) == 0.0) for array in arrays]
    if any(constant_flags):
        ids = [chain_ids[i] for i, flag in enumerate(constant_flags) if flag]
        diagnostics.append(
            _diag(
                "CONSTANT_CHAIN",
                "inconclusive",
                f"Constant chain(s): {', '.join(ids)}. Autocorrelation after lag zero and "
                "effective sample information are undefined for those chains.",
                scope="sample",
                uncertainty=True,
            )
        )
    min_length = min(lengths)
    if min_length < settings.min_ess_draws:
        diagnostics.append(
            _diag(
                "SHORT_CHAIN",
                "warning",
                f"The shortest chain has {min_length} draws; stable ESS requires at least "
                f"{settings.min_ess_draws}.",
                scope="sample",
                uncertainty=True,
            )
        )

    if request.require_equal_length and not equal_length:
        raise ValueError(
            "chains must have equal length when require_equal_length=true; "
            "set require_equal_length=false to receive weighted variance diagnostics only"
        )

    if not equal_length:
        diagnostics.append(
            _diag(
                "UNEQUAL_LENGTH_INPUT",
                "warning",
                "Chain lengths differ. Weighted within/between components are reported, but "
                "balanced ESS and R-hat are not estimated.",
                scope="sample",
                uncertainty=True,
            )
        )

    raw_matrix = np.vstack([array.reshape(1, -1) for array in arrays]) if equal_length else None
    all_constant = bool(raw_matrix is not None and np.ptp(raw_matrix) == 0.0)
    ess_results: list[ESSResult] = []
    if (
        raw_matrix is not None
        and min_length >= settings.min_ess_draws
        and not any(constant_flags)
    ):
        mean_ess, _, mean_diags = _ess_result(
            "mean",
            None if all(constant_flags) else raw_matrix,
            sum(lengths),
            settings,
            raw_series_constant=all(constant_flags),
        )
        split = _split_chains(raw_matrix)
        split_mean_ess, _, split_mean_diags = _ess_result(
            "split_mean",
            None if split.shape[1] < settings.min_ess_draws else split,
            sum(lengths),
            settings,
            raw_series_constant=all_constant,
        )
        bulk_input = _rank_normalize(split)
        bulk_ess, _, bulk_diags = _ess_result(
            "bulk",
            None if all(constant_flags) or split.shape[1] < settings.min_ess_draws else bulk_input,
            sum(lengths),
            settings,
            raw_series_constant=all(constant_flags),
        )
        if mean_ess is not None:
            ess_results.append(mean_ess)
        if split_mean_ess is not None:
            ess_results.append(split_mean_ess)
        if bulk_ess is not None:
            ess_results.append(bulk_ess)
        diagnostics.extend(mean_diags)
        diagnostics.extend(split_mean_diags)
        diagnostics.extend(bulk_diags)
    else:
        for kind in ("mean", "split_mean", "bulk"):
            result, _internals, kind_diags = _ess_result(
                kind, None, sum(lengths), settings
            )
            if result is not None:
                ess_results.append(result)
            diagnostics.extend(kind_diags)

    variance = (
        _variance_components(raw_matrix, True)
        if raw_matrix is not None
        else _variance_components(arrays, False)
    )
    rhat_results: list[RhatResult] = []
    if raw_matrix is not None:
        rhat_results, rhat_diags = _rhat_set(raw_matrix, True, settings)
        diagnostics.extend(rhat_diags)
    else:
        for kind in ("classic", "split", "rank_normalized"):
            rhat_results.append(
                RhatResult(
                    kind=kind,  # type: ignore[arg-type]
                    rhat=None,
                    state="not_estimated",
                    reason="unequal_length_input_requires_balanced_estimator",
                    threshold=settings.rhat_warning_threshold,
                )
            )

    if len(arrays) < settings.min_rhat_chains:
        steps.append("reported unavailable between-chain diagnostics rather than comparing a chain to itself")

    diagnostics = _deduplicate_diagnostics(diagnostics)
    return MultiChainResult(
        request_id=request_id,
        status=_overall_status(diagnostics),  # type: ignore[arg-type]
        n_chains=len(arrays),
        total_draws=sum(lengths),
        draws=lengths,
        autocovariance=AutocovSummary(
            lag_cutoff_rule=(
                "displayed ACF uses min(requested or default 10log10 rule,n-1); "
                "ESS separately scans complete autocorrelation lag pairs"
            ),
            reported_max_lag=max((item.max_lag for item in acf_results), default=0),
            ess_scanned_lags=max(lengths) - 1,
            centered=True,
            denominator="each chain C[0]; null when C[0]=0",
        ),
        acf=acf_results,
        ess=ess_results,
        variance_components=variance,
        rhat=rhat_results,
        diagnostics=diagnostics,
        limitations=_common_limitations(),
        steps=steps,
        versions=version_payload(settings),
        chain_ids=chain_ids,
    )
    internals = _geyer_ess(matrix)
    relative = internals.ess / (m * n)
    diagnostics: list[Diagnostic] = []
    if internals.reached_end:
        diagnostics.append(
            _diag(
                "ESS_TRUNCATION_AT_SERIES_END",
                "warning",
                f"{kind} ESS remained positive through every usable lag pair. "
                "The estimate may be unstable; use a longer run.",
                scope=f"ess.{kind}",
                uncertainty=True,
            )
        )
    if relative < settings.very_low_relative_ess:
        diagnostics.append(
            _diag(
                "ESS_SEVERE_DEGRADATION",
                "error",
                f"{kind} relative ESS is below {settings.very_low_relative_ess:.0%}.",
                scope=f"ess.{kind}",
            )
        )
    elif relative < settings.low_relative_ess:
        diagnostics.append(
            _diag(
                "ESS_LOW",
                "warning",
                f"{kind} relative ESS is below {settings.low_relative_ess:.0%}.",
                scope=f"ess.{kind}",
            )
        )

    return (
        ESSResult(
            kind=kind,  # type: ignore[arg-type]
            ess=internals.ess,
            total_draws=m * n,
            relative_ess=relative,
            integrated_time=internals.tau,
            cutoff_lag=internals.cutoff_lag,
            truncation_rule=truncation_rule,
            state="estimated",
        ),
        internals,
        diagnostics,
    )
