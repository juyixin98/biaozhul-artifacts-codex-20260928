from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.config import Settings
from app.errors import (
    BatchNotFoundError,
    ModelNotFoundError,
    ResourceExhaustedError,
    StateConflictError,
)
from app.persistence.repository import SQLiteRepository, utc_now
from app.stats.kernel import (
    beta_binomial_distribution,
    credible_beta_interval,
    posterior_distribution,
    validate_counts,
    validate_prior,
    wilson_confidence_interval,
)


class BetaBinomialService:
    def __init__(self, repository: SQLiteRepository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings

    def _model_or_raise(self, model_id: str) -> dict[str, Any]:
        model = self.repository.get_model(model_id)
        if model is None:
            raise ModelNotFoundError("Model does not exist.", {"model_id": model_id})
        return model

    def _batch_or_raise(self, model_id: str, batch_id: str) -> dict[str, Any]:
        batch = self.repository.get_batch(model_id, batch_id)
        if batch is None:
            raise BatchNotFoundError(
                "Batch does not exist for this model.",
                {"model_id": model_id, "batch_id": batch_id},
            )
        return batch

    def _validate_prior(self, alpha: float, beta: float) -> None:
        validate_prior(alpha, beta, self.settings.max_prior_parameter)

    def _validate_batch_counts(self, success_count: int, failure_count: int) -> None:
        validate_counts(
            success_count,
            failure_count,
            maximum=self.settings.max_observation_count,
        )

    def _check_aggregate_capacity(
        self,
        success_count: int,
        failure_count: int,
        *,
        current_success: int = 0,
        current_failure: int = 0,
    ) -> None:
        future_success = current_success + success_count
        future_failure = current_failure + failure_count
        if (
            future_success > self.settings.max_observation_count
            or future_failure > self.settings.max_observation_count
        ):
            raise ResourceExhaustedError(
                "The requested update would exceed local sufficient-statistic capacity.",
                {
                    "maximum_each": self.settings.max_observation_count,
                    "current_success": current_success,
                    "current_failure": current_failure,
                    "future_success": future_success,
                    "future_failure": future_failure,
                },
            )

    @staticmethod
    def _next_version(connection, model_id: str) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(version), -1) AS max_version FROM events WHERE model_id = ?",
            (model_id,),
        ).fetchone()
        return int(row["max_version"]) + 1

    def _active_totals(self, connection, model_id: str) -> tuple[int, int]:
        rows = self.repository.list_active_batches(connection, model_id)
        success_count = sum(int(row["success_count"]) for row in rows)
        failure_count = sum(int(row["failure_count"]) for row in rows)
        return success_count, failure_count

    def _record_event(
        self,
        connection,
        *,
        model_id: str,
        event_type: str,
        batch_id: str | None,
        delta_success: int,
        delta_failure: int,
        reason: str | None,
        details: dict[str, Any],
    ) -> int:
        version = self._next_version(connection, model_id)
        total_success, total_failure = self._active_totals(connection, model_id)
        event_details = {
            "reason": reason,
            "intermediate": {
                "delta_success": delta_success,
                "delta_failure": delta_failure,
                "total_success_after_event": total_success,
                "total_failure_after_event": total_failure,
            },
            "judgement": details.pop("judgement", "Model state updated."),
        }
        event_details.update(details)
        self.repository.insert_event(
            connection,
            model_id=model_id,
            version=version,
            event_type=event_type,
            batch_id=batch_id,
            delta_success=delta_success,
            delta_failure=delta_failure,
            total_success=total_success,
            total_failure=total_failure,
            details=event_details,
            created_at=utc_now(),
        )
        return version

    def create_model(
        self, alpha: float, beta: float, reason: str | None = None
    ) -> dict[str, Any]:
        self._validate_prior(alpha, beta)
        model_id = f"bbm_{uuid4().hex}"
        created_at = utc_now()
        with self.repository.transaction() as connection:
            self.repository.insert_model(
                connection,
                model_id=model_id,
                prior_alpha=alpha,
                prior_beta=beta,
                created_at=created_at,
            )
            self.repository.insert_event(
                connection,
                model_id=model_id,
                version=0,
                event_type="MODEL_CREATED",
                batch_id=None,
                delta_success=0,
                delta_failure=0,
                total_success=0,
                total_failure=0,
                details={
                    "reason": reason,
                    "prior_alpha": float(alpha),
                    "prior_beta": float(beta),
                    "intermediate": {
                        "delta_success": 0,
                        "delta_failure": 0,
                        "total_success_after_event": 0,
                        "total_failure_after_event": 0,
                    },
                    "judgement": "Initialized an empty Beta conjugate model.",
                },
                created_at=created_at,
            )
        result = self.repository.get_model(model_id)
        assert result is not None
        result["version"] = 0
        result["success_count"] = 0
        result["failure_count"] = 0
        return result

    def add_batch(
        self,
        model_id: str,
        success_count: int,
        failure_count: int,
        reason: str | None = None,
    ) -> dict[str, Any]:
        self._model_or_raise(model_id)
        self._validate_batch_counts(success_count, failure_count)
        batch_id = f"bbb_{uuid4().hex}"
        created_at = utc_now()
        with self.repository.transaction() as connection:
            before_success, before_failure = self._active_totals(connection, model_id)
            self._check_aggregate_capacity(
                success_count,
                failure_count,
                current_success=before_success,
                current_failure=before_failure,
            )
            self.repository.insert_batch(
                connection,
                batch_id=batch_id,
                model_id=model_id,
                revision_of_batch_id=None,
                status="active",
                version=0,
                success_count=success_count,
                failure_count=failure_count,
                reason=reason,
                created_at=created_at,
            )
            version = self._record_event(
                connection,
                model_id=model_id,
                event_type="BATCH_ADDED",
                batch_id=batch_id,
                delta_success=success_count,
                delta_failure=failure_count,
                reason=reason,
                details={
                    "state_before": {
                        "total_success": before_success,
                        "total_failure": before_failure,
                    },
                    "judgement": "New active sufficient statistics were added.",
                },
            )
            connection.execute(
                "UPDATE batches SET version = ? WHERE id = ?", (version, batch_id)
            )
        batch = self.repository.get_batch(model_id, batch_id)
        assert batch is not None
        batch["version"] = version
        return batch

    def revise_batch(
        self,
        model_id: str,
        batch_id: str,
        success_count: int,
        failure_count: int,
        reason: str | None = None,
    ) -> dict[str, Any]:
        self._model_or_raise(model_id)
        old_batch = self._batch_or_raise(model_id, batch_id)
        if old_batch["status"] != "active":
            raise StateConflictError(
                "Only an active batch can be revised.",
                {"batch_id": batch_id, "current_status": old_batch["status"]},
            )
        self._validate_batch_counts(success_count, failure_count)

        replacement_id = f"bbb_{uuid4().hex}"
        created_at = utc_now()
        with self.repository.transaction() as connection:
            before_success, before_failure = self._active_totals(connection, model_id)
            withdrawn_success = before_success - int(old_batch["success_count"])
            withdrawn_failure = before_failure - int(old_batch["failure_count"])
            self._check_aggregate_capacity(
                success_count,
                failure_count,
                current_success=withdrawn_success,
                current_failure=withdrawn_failure,
            )

            # The original sufficient statistics are withdrawn first.
            self.repository.update_batch_status(connection, batch_id, "superseded")
            withdraw_version = self._record_event(
                connection,
                model_id=model_id,
                event_type="BATCH_REVISION_WITHDREW_ORIGINAL",
                batch_id=batch_id,
                delta_success=-int(old_batch["success_count"]),
                delta_failure=-int(old_batch["failure_count"]),
                reason=reason,
                details={
                    "state_before": {
                        "total_success": before_success,
                        "total_failure": before_failure,
                    },
                    "old_batch_id": batch_id,
                    "replacement_batch_id": replacement_id,
                    "judgement": "The original active batch was withdrawn before replacement.",
                },
            )

            self.repository.insert_batch(
                connection,
                batch_id=replacement_id,
                model_id=model_id,
                revision_of_batch_id=batch_id,
                status="active",
                version=0,
                success_count=success_count,
                failure_count=failure_count,
                reason=reason,
                created_at=created_at,
            )
            add_version = self._record_event(
                connection,
                model_id=model_id,
                event_type="BATCH_REVISION_ADDED_REPLACEMENT",
                batch_id=replacement_id,
                delta_success=success_count,
                delta_failure=failure_count,
                reason=reason,
                details={
                    "state_before": {
                        "total_success": withdrawn_success,
                        "total_failure": withdrawn_failure,
                    },
                    "revised_batch_id": batch_id,
                    "judgement": "The corrected sufficient statistics were inserted.",
                },
            )

            connection.execute(
                "UPDATE batches SET version = ? WHERE id = ?",
                (add_version, replacement_id),
            )

        replacement = self.repository.get_batch(model_id, replacement_id)
        assert replacement is not None
        replacement["withdraw_version"] = withdraw_version
        replacement["add_version"] = add_version
        return replacement

    def revert_batch(
        self,
        model_id: str,
        batch_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        self._model_or_raise(model_id)
        batch = self._batch_or_raise(model_id, batch_id)
        if batch["status"] != "active":
            raise StateConflictError(
                "Only an active batch can be reverted.",
                {"batch_id": batch_id, "current_status": batch["status"]},
            )

        with self.repository.transaction() as connection:
            before_success, before_failure = self._active_totals(connection, model_id)
            self.repository.update_batch_status(connection, batch_id, "reverted")
            event_type = (
                "BATCH_REVERTED_REVISION"
                if batch["revision_of_batch_id"] is not None
                else "BATCH_REVERTED"
            )
            version = self._record_event(
                connection,
                model_id=model_id,
                event_type=event_type,
                batch_id=batch_id,
                delta_success=-int(batch["success_count"]),
                delta_failure=-int(batch["failure_count"]),
                reason=reason,
                details={
                    "state_before": {
                        "total_success": before_success,
                        "total_failure": before_failure,
                    },
                    "revision_of_batch_id": batch["revision_of_batch_id"],
                    "judgement": "Active sufficient statistics were removed by explicit revert.",
                },
            )
            connection.execute(
                "UPDATE batches SET version = ? WHERE id = ?", (version, batch_id)
            )
        updated = self.repository.get_batch(model_id, batch_id)
        assert updated is not None
        updated["version"] = version
        return updated

    def restore_batch(
        self,
        model_id: str,
        batch_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        self._model_or_raise(model_id)
        batch = self._batch_or_raise(model_id, batch_id)
        if batch["status"] not in {"reverted", "superseded"}:
            raise StateConflictError(
                "Only a reverted or superseded batch can be restored.",
                {"batch_id": batch_id, "current_status": batch["status"]},
            )

        with self.repository.transaction() as connection:
            before_success, before_failure = self._active_totals(connection, model_id)
            details = {
                "state_before": {
                    "total_success": before_success,
                    "total_failure": before_failure,
                },
                "revision_of_batch_id": batch["revision_of_batch_id"],
            }

            if batch["status"] == "superseded":
                rows = connection.execute(
                    """
                    SELECT id FROM batches
                    WHERE model_id = ? AND revision_of_batch_id = ? AND status = 'active'
                    """,
                    (model_id, batch_id),
                ).fetchall()
                if not rows:
                    raise StateConflictError(
                        "Superseded batch has no active replacement to withdraw.",
                        {"batch_id": batch_id},
                    )
                active_replacement = rows[0]
                replacement = self.repository.get_batch(model_id, active_replacement["id"])
                assert replacement is not None

                self.repository.update_batch_status(
                    connection, replacement["id"], "reverted"
                )
                withdraw_version = self._record_event(
                    connection,
                    model_id=model_id,
                    event_type="BATCH_RECOVERY_WITHDREW_REPLACEMENT",
                    batch_id=replacement["id"],
                    delta_success=-int(replacement["success_count"]),
                    delta_failure=-int(replacement["failure_count"]),
                    reason=reason,
                    details={
                        **details,
                        "restored_batch_id": batch_id,
                        "judgement": "The later correction was withdrawn to restore the original.",
                    },
                )
                self.repository.update_batch_status(connection, batch_id, "active")
                add_version = self._record_event(
                    connection,
                    model_id=model_id,
                    event_type="BATCH_RECOVERY_RESTORED_ORIGINAL",
                    batch_id=batch_id,
                    delta_success=int(batch["success_count"]),
                    delta_failure=int(batch["failure_count"]),
                    reason=reason,
                    details={
                        "replacement_batch_id": replacement["id"],
                        "judgement": "The original sufficient statistics became active again.",
                    },
                )
                version = add_version
                details["withdraw_version"] = withdraw_version
                connection.execute(
                    "UPDATE batches SET version = ? WHERE id = ?",
                    (add_version, batch_id),
                )
                connection.execute(
                    "UPDATE batches SET version = ? WHERE id = ?",
                    (withdraw_version, replacement["id"]),
                )
            else:
                self.repository.update_batch_status(connection, batch_id, "active")
                version = self._record_event(
                    connection,
                    model_id=model_id,
                    event_type="BATCH_RESTORED",
                    batch_id=batch_id,
                    delta_success=int(batch["success_count"]),
                    delta_failure=int(batch["failure_count"]),
                    reason=reason,
                    details={
                        **details,
                        "judgement": "A previously reverted batch became active again.",
                    },
                )
                connection.execute(
                    "UPDATE batches SET version = ? WHERE id = ?", (version, batch_id)
                )

        updated = self.repository.get_batch(model_id, batch_id)
        assert updated is not None
        updated["version"] = version
        return updated

    def list_batches(self, model_id: str) -> list[dict[str, Any]]:
        self._model_or_raise(model_id)
        return self.repository.list_batches(model_id)

    def list_events(self, model_id: str) -> list[dict[str, Any]]:
        self._model_or_raise(model_id)
        return self.repository.list_events(model_id)

    def current_state(self, model_id: str) -> dict[str, Any]:
        model = self._model_or_raise(model_id)
        with self.repository.transaction() as connection:
            version_row = connection.execute(
                "SELECT MAX(version) AS version FROM events WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            total_success, total_failure = self._active_totals(connection, model_id)
        return {
            **model,
            "version": int(version_row["version"]),
            "success_count": total_success,
            "failure_count": total_failure,
        }

    def posterior_summary(self, model_id: str) -> dict[str, Any]:
        state = self.current_state(model_id)
        posterior = posterior_distribution(
            state["prior_alpha"],
            state["prior_beta"],
            int(state["success_count"]),
            int(state["failure_count"]),
        )
        credible_lower, credible_upper = credible_beta_interval(
            posterior.alpha, posterior.beta, self.settings.interval_level
        )
        confidence_lower, confidence_upper = wilson_confidence_interval(
            int(state["success_count"]),
            int(state["success_count"]) + int(state["failure_count"]),
            self.settings.interval_level,
        )
        return {
            "model_id": state["id"],
            "version": int(state["version"]),
            "prior_alpha": float(state["prior_alpha"]),
            "prior_beta": float(state["prior_beta"]),
            "success_count": int(state["success_count"]),
            "failure_count": int(state["failure_count"]),
            "posterior_alpha": posterior.alpha,
            "posterior_beta": posterior.beta,
            "posterior_mean": posterior.mean,
            "posterior_variance": posterior.variance,
            "credible_interval": {
                "level": self.settings.interval_level,
                "lower": credible_lower,
                "upper": credible_upper,
                "interpretation": "Bayesian credible interval",
            },
            "frequentist_confidence_interval": {
                "level": self.settings.interval_level,
                "lower": confidence_lower,
                "upper": confidence_upper,
                "method": "Wilson score confidence interval (frequentist)",
                "interpretation": "Frequentist confidence interval, not a Bayesian credible interval",
            },
        }

    def predict(
        self,
        model_id: str,
        trials: int,
        include_distribution: bool = False,
        reason: str | None = None,
    ) -> dict[str, Any]:
        validate_counts(trials, 0, names=("trials", "zero"))
        if trials > self.settings.max_predictive_trials:
            raise ResourceExhaustedError(
                "Predictive support is larger than the local response capacity.",
                {
                    "trials": trials,
                    "maximum_predictive_trials": self.settings.max_predictive_trials,
                },
            )
        summary = self.posterior_summary(model_id)
        probabilities, _log_probabilities, probability_sum = beta_binomial_distribution(
            trials,
            summary["posterior_alpha"],
            summary["posterior_beta"],
        )
        return {
            "posterior": summary,
            "trials": trials,
            "probabilities": [float(x) for x in probabilities] if include_distribution else None,
            "probability_sum": probability_sum,
            "distribution_tolerance": 1.0e-12,
        }
