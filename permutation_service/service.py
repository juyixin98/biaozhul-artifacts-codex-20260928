"""Application service that executes jobs and records their lifecycle."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

from .config import Settings
from .core import PermutationRequest, preflight_request, run_permutation_test
from .errors import (
    CancellationError,
    InvalidInputError,
    PermutationError,
    RunNotFoundError,
)
from .storage import RunStore
from .versions import runtime_versions

LOG = logging.getLogger("permutation_service")


class JobRunner:
    def __init__(
        self,
        store: RunStore,
        settings: Settings,
        *,
        max_workers: int = 2,
    ) -> None:
        self.store = store
        self.settings = settings
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="perm-run",
        )
        self._cancels: dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()

    def start(self) -> None:
        stale = self.store.reset_stale_runs()
        if stale:
            LOG.warning("reset %d interrupted run(s) as failed", stale)

    def stop(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def input_hash(run_input: dict[str, Any]) -> str:
        canonical = json.dumps(run_input, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _exact_limit(self, run_input: dict[str, Any]) -> int:
        requested = run_input.get("exact_max_combinations")
        limit = (
            self.settings.exact_max_combinations
            if requested is None
            else int(requested)
        )
        if limit > self.settings.requested_exact_limit_cap:
            raise InvalidInputError(
                "exact_max_combinations exceeds the configured safety cap",
                details={
                    "requested": limit,
                    "cap": self.settings.requested_exact_limit_cap,
                },
            )
        return limit

    def _build_core_request(
        self, run_input: dict[str, Any], exact_limit: int
    ) -> PermutationRequest:
        return PermutationRequest(
            values=run_input["values"],
            group1_size=int(run_input["group1_size"]),
            alpha=float(run_input["alpha"]),
            method=run_input["method"],
            monte_carlo_replications=(
                self.settings.default_monte_carlo_replications
                if run_input.get("monte_carlo_replications") is None
                else int(run_input["monte_carlo_replications"])
            ),
            exact_max_combinations=exact_limit,
            exact_limit_cap=self.settings.requested_exact_limit_cap,
            max_values=self.settings.max_values,
            max_monte_carlo_replications=(
                self.settings.max_monte_carlo_replications
            ),
            random_seed=run_input.get("random_seed"),
            return_null_distribution=bool(run_input.get("return_null_distribution", False)),
        )

    def create_run(self, run_input: dict[str, Any]) -> dict[str, Any]:
        exact_limit = self._exact_limit(run_input)
        b = run_input.get("monte_carlo_replications")
        if b is not None and int(b) > self.settings.max_monte_carlo_replications:
            raise InvalidInputError(
                "monte_carlo_replications exceeds configured limit",
                details={
                    "received": int(b),
                    "limit": self.settings.max_monte_carlo_replications,
                },
            )
        if len(run_input["values"]) > self.settings.max_values:
            raise InvalidInputError(
                "number of values must not exceed configured maximum",
                details={
                    "received": len(run_input["values"]),
                    "limit": self.settings.max_values,
                },
            )

        core_request = self._build_core_request(run_input, exact_limit)
        preflight_request(core_request)

        input_hash = self.input_hash(run_input)
        run = self.store.create_run(run_input, input_hash)
        run_id = run["run_id"]
        with self._cancel_lock:
            self._cancels[run_id] = threading.Event()
        future = self._executor.submit(self._execute, run_id, run_input, exact_limit)
        future.add_done_callback(lambda f: self._discard_future(run_id, f))
        return run

    def _discard_future(self, run_id: str, future: Future) -> None:
        with self._cancel_lock:
            self._cancels.pop(run_id, None)
        exception = future.exception()
        if exception is not None and not isinstance(exception, CancellationError):
            LOG.exception("background run %s failed unexpectedly", run_id)

    def _execute(
        self, run_id: str, run_input: dict[str, Any], exact_limit: int
    ) -> None:
        with self._cancel_lock:
            event = self._cancels.get(run_id)
        if event is None:
            event = threading.Event()

        run = self.store.mark_running(run_id)
        if run is None:
            raise RunNotFoundError("run disappeared before execution")
        if run["status"] != "running":
            self._record_cancellation(run_id, "canceled before computation started")
            return

        versions = runtime_versions()
        self.store.add_event(
            run_id,
            "versions",
            "computation started; "
            + ", ".join(f"{name}={version}" for name, version in versions.items()),
        )

        start_time = time.monotonic()
        progress_state = {"last_time": 0.0, "last_percent": -1}

        def progress(completed: int, total: int, phase: str, batch_size: int) -> None:
            now = time.monotonic()
            percent = -1 if total <= 0 else int(completed * 100 / total)
            should_log = (
                completed == 0
                or completed == total
                or percent != progress_state["last_percent"]
                or now - progress_state["last_time"] >= 0.25
            )
            if not should_log:
                return
            progress_state["last_time"] = now
            progress_state["last_percent"] = percent
            self.store.add_event(
                run_id,
                phase,
                f"{phase}: {completed}/{total} completed ({percent}%)",
                completed=completed,
                total=total,
                batch_size=batch_size,
            )
            LOG.info(
                "run_id=%s input_hash=%s phase=%s completed=%d total=%d batch=%d",
                run_id,
                run["input_hash"],
                phase,
                completed,
                total,
                batch_size,
            )

        try:
            core_request = self._build_core_request(run_input, exact_limit)
            result = run_permutation_test(
                core_request,
                progress=progress,
                is_canceled=event.is_set,
            )
            payload = asdict(result)
            payload["runtime_versions"] = versions
            payload["elapsed_ms"] = round((time.monotonic() - start_time) * 1000, 3)
            decision = "significant" if result.significant else "not_significant"
            self.store.add_event(
                run_id,
                "decision",
                (
                    f"method={result.method_used}; observed={result.statistic:.12g}; "
                    f"p_value={result.p_value:.12g}; alpha={result.alpha}; "
                    f"decision={decision}"
                ),
            )
            completed_run, did_complete = self.store.complete_run(run_id, payload)
            if did_complete:
                return
            if completed_run is None:
                raise RunNotFoundError("run disappeared while completing")
            self._record_cancellation(
                run_id,
                "cancellation arrived after final computation step; no result accepted",
            )
        except CancellationError:
            self._record_cancellation(run_id, "cancellation observed by statistical kernel")
        except PermutationError as exc:
            self.store.add_event(
                run_id,
                "error",
                f"{exc.error_code}: {exc.message}",
            )
            self.store.fail_run(
                run_id,
                {
                    "code": exc.error_code,
                    "message": exc.message,
                    "category": exc.category,
                    "details": exc.details,
                },
            )
        except Exception as exc:  # Defensive: unknown failures must not look successful.
            LOG.exception("run_id=%s unknown failure", run_id)
            self.store.add_event(run_id, "error", f"UNKNOWN_ERROR: {exc}")
            self.store.fail_run(
                run_id,
                {
                    "code": "UNKNOWN_ERROR",
                    "message": str(exc) or "unexpected computation error",
                    "category": "unknown",
                    "details": {},
                },
            )

    def _record_cancellation(self, run_id: str, message: str) -> None:
        self.store.add_event(run_id, "canceled", message)
        self.store.cancel_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"run {run_id} was not found")
        return run

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"run {run_id} was not found")
        updated = self.store.request_cancel(run_id)
        if updated is None:
            raise RunNotFoundError(f"run {run_id} was not found")
        with self._cancel_lock:
            event = self._cancels.get(run_id)
        if event is not None:
            event.set()
        return updated

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        events = self.store.list_events(run_id)
        if events is None:
            raise RunNotFoundError(f"run {run_id} was not found")
        return events
