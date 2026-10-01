"""Versioned survival analysis orchestration."""

from __future__ import annotations

import socket

from survival_core import Observation, kaplan_meier, log_rank

from .config import Settings
from .contracts import SurvivalAnalysisRequest
from .tracing import failure, notice, trace_entry
from .versioning import current_versions


def _point_payload(point, group: str) -> dict:
    return {
        "time": point.time,
        "kind": point.kind,
        "risk_before": point.risk_before,
        "entered": point.entered,
        "events": point.events,
        "censored": point.censored,
        "risk_after": point.risk_after,
        "survival": None if point.survival is None else float(point.survival),
        "greenwood_variance": None
        if point.greenwood_variance is None
        else float(point.greenwood_variance),
        "ci_low": None if point.ci_low is None else float(point.ci_low),
        "ci_high": None if point.ci_high is None else float(point.ci_high),
        "ci_clipped": bool(point.ci_clipped),
    }


def run_analysis(
    request: SurvivalAnalysisRequest,
    settings: Settings,
    *,
    analysis_id: str,
    request_id: str,
    created_at: str,
) -> dict:
    trace = [
        trace_entry(
            "api",
            "request_received",
            request_id=request_id,
            analysis_id=analysis_id,
            client_request_id=request.client_request_id,
        ),
        trace_entry(
            "risk-kernel",
            "risk_set_order_declared",
            rule="events_and_tied_censored_subjects_remain_at_risk_during_same_timepoint",
            left_truncation_rule="subject_enters_when_entry_is_strictly_less_than_time",
        ),
    ]
    notices: list[dict] = []
    failures: list[dict] = []
    versions = current_versions(settings)

    observations = [
        Observation(
            observation_id=item.observation_id,
            group=item.group,
            time=item.time,
            event=item.event,
            entry=item.entry,
        )
        for item in request.observations
    ]
    observed_groups = sorted({item.group for item in observations})
    groups = request.groups if request.groups is not None else observed_groups

    trace.append(
        trace_entry(
            "service",
            "groups_resolved",
            group_order=groups,
            observed_groups=observed_groups,
            empty_groups=[
                group
                for group in groups
                if not any(item.group == group for item in observations)
            ],
        )
    )

    group_payloads: list[dict] = []
    for group in groups:
        group_observations = [item for item in observations if item.group == group]
        km = kaplan_meier(
            group_observations,
            confidence_level=request.confidence_level,
            confidence_interval_method=request.confidence_interval_method,
        )
        for code in km.notices:
            message = _notice_message(code)
            notices.append(notice(code, message, f"group:{group}"))
        group_payloads.append(
            {
                "group": group,
                "n": km.n,
                "events": km.events,
                "censored": km.censored,
                "median_survival_time": km.median_survival_time,
                "median_status": km.median_status,
                "points": [_point_payload(point, group) for point in km.points],
            }
        )
        trace.append(
            trace_entry(
                "kaplan-meier",
                "group_estimated",
                group=group,
                n=km.n,
                events=km.events,
                censored=km.censored,
                median_status=km.median_status,
                timepoints=len(km.points),
            )
        )

    comparison_core = log_rank(
        observations,
        groups,
        significance_level=request.significance_level,
    )
    comparison = {
        "status": comparison_core.status,
        "reason": comparison_core.reason,
        "method": comparison_core.method,
        "group_order": comparison_core.group_order,
        "observed": comparison_core.observed,
        "expected": comparison_core.expected,
        "variance_covariance_matrix": comparison_core.variance_covariance_matrix,
        "chi_square": comparison_core.chi_square,
        "degrees_freedom": comparison_core.degrees_freedom,
        "p_value": comparison_core.p_value,
    }
    if comparison_core.status == "failed":
        failures.append(
            failure(
                comparison_core.reason or "LOG_RANK_FAILED",
                "Log-rank comparison was not produced because the requested comparison is not statistically identifiable.",
                "comparison:log_rank",
            )
        )
    elif comparison_core.status == "not_run":
        notices.append(
            notice(
                comparison_core.reason or "LOG_RANK_NOT_RUN",
                "The log-rank test is reported as not run; do not interpret this as evidence of equality or difference.",
                "comparison:log_rank",
            )
        )
    trace.append(
        trace_entry(
            "log-rank",
            "comparison_completed",
            status=comparison_core.status,
            reason=comparison_core.reason,
            degrees_freedom=comparison_core.degrees_freedom,
        )
    )

    total_events = sum(group["events"] for group in group_payloads)
    total_censored = sum(group["censored"] for group in group_payloads)
    response = {
        "schema_version": versions.schema_version,
        "service_version": versions.service_version,
        "kernel_version": versions.kernel_version,
        "analysis_id": analysis_id,
        "request_id": request_id,
        "analysis_name": request.analysis_name,
        "created_at": created_at,
        "processing_location": f"local-process:{socket.gethostname()}",
        "config": {
            "confidence_level": request.confidence_level,
            "confidence_interval_method": request.confidence_interval_method,
            "significance_level": request.significance_level,
            "schema_migration_version": versions.schema_migration_version,
            "left_truncation": "supported_when_entry_is_provided",
            "risk_ties": "events_and_censoring_share_risk_set_at_same_time",
        },
        "summary": {
            "n_observations": len(observations),
            "n_groups": len(groups),
            "events": total_events,
            "censored": total_censored,
            "empty_groups": [
                group["group"] for group in group_payloads if group["n"] == 0
            ],
        },
        "groups": group_payloads,
        "comparison": comparison,
        "trace": trace,
        "notices": notices,
        "failures": failures,
        "disclaimer": (
            "Synthetic and demonstration-only output. This service is not validated "
            "for medical use and results must not support a real medical conclusion."
        ),
    }
    trace.append(
        trace_entry(
            "service",
            "response_serialized",
            schema_version=versions.schema_version,
            failure_count=len(failures),
            notice_count=len(notices),
        )
    )
    return response


def _notice_message(code: str) -> str:
    messages = {
        "boundary_survival_one": "Survival is exactly one before the first event; the boundary interval is reported as [1, 1].",
        "boundary_survival_zero_variance_undefined": "The curve reached zero. The terminal Greenwood variance and interval are undefined and are not fabricated.",
        "clipped_to_probability_range": "A linear Greenwood endpoint was clipped to the valid probability range [0, 1].",
        "empty_group_has_no_median": "The group has no observations, so no median survival time is estimated.",
        "no_events_median_not_reached": "The group has no events, so median survival is not reached and is not fabricated.",
        "latest_survival_above_half_median_not_reached": "The latest estimated survival remains above 0.5, so median survival is not reached.",
    }
    return messages.get(code, code.replace("_", " "))
