"""Application service orchestrating storage and the statistical kernel."""

from __future__ import annotations

from pathlib import Path

from app.domain import ensure_count, ensure_level, ensure_prior
from app.errors import NotFoundError, StateConflictError
from app.schemas import (
    BatchResponse,
    BatchSnapshot,
    IntervalResponse,
    PosteriorSummaryResponse,
    PredictionPointResponse,
    PredictionResponse,
    RunResponse,
    RunVersionResponse,
)
from app.stats import (
    beta_credible_interval,
    beta_moments,
    beta_binomial_predictions,
    wilson_confidence_interval,
)
from app.storage import SQLiteStore


class BetaBinomialService:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    @classmethod
    def with_file_database(cls, path: str | Path) -> "BetaBinomialService":
        return cls(SQLiteStore(path))

    def create_run(self, run_id: str, alpha: object, beta: object) -> RunResponse:
        prior_alpha, prior_beta = ensure_prior(alpha, beta)
        created = self.store.create_run(run_id, prior_alpha, prior_beta)
        return RunResponse(
            run_id=created["run_id"],
            alpha=created["alpha"],
            beta=created["beta"],
            created_at=created["created_at"],
        )

    def get_run(self, run_id: str) -> RunResponse:
        run = self.store.get_run(run_id)
        return RunResponse(**run)

    def submit_batch(
        self,
        run_id: str,
        batch_id: str,
        successes: object,
        failures: object,
        reason: str | None = None,
    ) -> BatchResponse:
        self.store.get_run(run_id)
        successes = ensure_count(successes, "successes")
        failures = ensure_count(failures, "failures")
        if self.store.get_active_batch(run_id, batch_id) is not None:
            raise StateConflictError(
                "Batch already exists; revise it rather than resubmitting.",
                {"run_id": run_id, "batch_id": batch_id},
            )
        result = self.store.apply_batch(
            run_id, batch_id, successes, failures, reason
        )
        return BatchResponse(**result)

    def revise_batch(
        self,
        run_id: str,
        batch_id: str,
        successes: object,
        failures: object,
        reason: str | None = None,
    ) -> BatchResponse:
        self.store.get_run(run_id)
        successes = ensure_count(successes, "successes")
        failures = ensure_count(failures, "failures")
        if self.store.get_active_batch(run_id, batch_id) is None:
            raise NotFoundError(
                "Cannot revise a batch that has never been submitted.",
                {"run_id": run_id, "batch_id": batch_id},
            )
        result = self.store.apply_batch(
            run_id, batch_id, successes, failures, reason
        )
        return BatchResponse(**result)

    def _latest_version(self, run_id: str) -> dict:
        versions = self.store.list_run_versions(run_id)
        if not versions:
            raise NotFoundError("Run has no versions.", {"run_id": run_id})
        return versions[-1]

    def _active_batch_snapshots(self, run_id: str) -> list[BatchSnapshot]:
        rows = self.store.active_batches(run_id)
        return [
            BatchSnapshot(
                batch_id=row["batch_id"],
                version=row["version"],
                successes=row["successes"],
                failures=row["failures"],
                status=row["status"],
                reason=row["reason"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def posterior_summary(
        self, run_id: str, confidence_level: object = 0.95
    ) -> PosteriorSummaryResponse:
        run = self.store.get_run(run_id)
        level = ensure_level(confidence_level)
        latest = self._latest_version(run_id)
        post_alpha = float(latest["posterior_alpha"])
        post_beta = float(latest["posterior_beta"])
        mean, variance = beta_moments(post_alpha, post_beta)
        credible = beta_credible_interval(post_alpha, post_beta, level)
        confidence = wilson_confidence_interval(
            latest["total_successes"], latest["total_failures"], level
        )
        credible_response = IntervalResponse(
            lower=credible.lower, upper=credible.upper, level=credible.level
        )
        confidence_response = None
        if confidence is not None:
            confidence_response = IntervalResponse(
                lower=confidence.lower,
                upper=confidence.upper,
                level=confidence.level,
            )
        return PosteriorSummaryResponse(
            run_id=run_id,
            prior_alpha=float(run["alpha"]),
            prior_beta=float(run["beta"]),
            posterior_alpha=post_alpha,
            posterior_beta=post_beta,
            total_successes=latest["total_successes"],
            total_failures=latest["total_failures"],
            mean=mean,
            variance=variance,
            credible_interval=credible_response,
            confidence_interval=confidence_response,
            active_batches=self._active_batch_snapshots(run_id),
        )

    def predict(
        self, run_id: str, future_n: object, confidence_level: object = 0.95
    ) -> PredictionResponse:
        level = ensure_level(confidence_level)
        latest = self._latest_version(run_id)
        post_alpha = float(latest["posterior_alpha"])
        post_beta = float(latest["posterior_beta"])
        distribution = beta_binomial_predictions(future_n, post_alpha, post_beta)
        credible = beta_credible_interval(post_alpha, post_beta, level)
        confidence = wilson_confidence_interval(
            latest["total_successes"], latest["total_failures"], level
        )
        credible_response = IntervalResponse(
            lower=credible.lower, upper=credible.upper, level=credible.level
        )
        confidence_response = None
        if confidence is not None:
            confidence_response = IntervalResponse(
                lower=confidence.lower,
                upper=confidence.upper,
                level=confidence.level,
            )
        return PredictionResponse(
            run_id=run_id,
            future_n=distribution.future_n,
            posterior_alpha=post_alpha,
            posterior_beta=post_beta,
            points=[
                PredictionPointResponse(
                    future_successes=point.future_successes,
                    probability=point.probability,
                )
                for point in distribution.points
            ],
            probability_sum=float(sum(point.probability for point in distribution.points)),
            mean=distribution.mean,
            variance=distribution.variance,
            credible_interval=credible_response,
            confidence_interval=confidence_response,
        )

    def list_versions(self, run_id: str) -> list[RunVersionResponse]:
        return [RunVersionResponse(**version) for version in self.store.list_run_versions(run_id)]
