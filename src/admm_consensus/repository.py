"""Durable JSON state and append-only evidence logs for local runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np

from .config import Settings
from .diagnostics import event_payload, state_snapshot
from .errors import NotFound, ResourceExhausted
from .models import AdaptiveRho, Problem, SolverConfig, SolverState, Tolerances
from .parser import parse_problem


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class RunRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.data_dir / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def run_dir(self, run_id: str) -> Path:
        if "/" in run_id or "\\" in run_id or not run_id:
            raise NotFound("Run ID is invalid.", {"run_id": run_id})
        return self.root / run_id

    def state_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "state.json"

    def event_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"

    def create(self, state: SolverState) -> None:
        with self._lock:
            run_dir = self.run_dir(state.run_id)
            try:
                existing = len([path for path in self.root.iterdir() if path.is_dir()])
                if existing >= self.settings.max_runs:
                    raise ResourceExhausted(
                        "Local run storage limit was reached.",
                        {"existing_runs": existing, "limit": self.settings.max_runs},
                    )
                run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise ResourceExhausted("Run ID already exists.", {"run_id": state.run_id}) from exc
            self.append_event(state, "run_created", {"status": state.status})
            self.save(state)

    def save(self, state: SolverState) -> None:
        with self._lock:
            run_dir = self.run_dir(state.run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            target = self.state_path(state.run_id)
            temporary = target.with_suffix(".json.tmp")
            payload = state_snapshot(state, include_problem=True)
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)

    def append_event(self, state: SolverState, event_type: str, details: dict[str, Any]) -> None:
        with self._lock:
            path = self.event_path(state.run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            event = event_payload(state, event_type, details)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def load(self, run_id: str) -> SolverState:
        with self._lock:
            path = self.state_path(run_id)
            if not path.exists():
                raise NotFound("Run was not found.", {"run_id": run_id})
            data = _load_json(path)
            problem: Problem = parse_problem(data["problem"], self.settings)
            adaptive_data = data["config"].get("adaptive_rho")
            adaptive = None
            if adaptive_data is not None:
                adaptive = AdaptiveRho(
                    mu=float(adaptive_data["mu"]),
                    tau_incr=float(adaptive_data["tau_incr"]),
                    tau_decr=float(adaptive_data["tau_decr"]),
                    change_rounds=frozenset(int(v) for v in adaptive_data["change_rounds"]),
                )
            tol_data = data["config"]["tolerances"]
            config = SolverConfig(
                max_iterations=int(data["config"]["max_iterations"]),
                initial_rho=float(data["config"]["rho"]),
                tol=Tolerances(
                    primal_abs=float(tol_data["primal_abs"]),
                    dual_abs=float(tol_data["dual_abs"]),
                    primal_rel=float(tol_data["primal_rel"]),
                    dual_rel=float(tol_data["dual_rel"]),
                ),
                adaptive_rho=adaptive,
            )
            return SolverState(
                run_id=data["run_id"],
                problem=problem,
                config=config,
                status=data["status"],
                round_number=int(data["round_number"]),
                rho=float(data["rho"]),
                x=[np.asarray(v, dtype=float) for v in data["x"]],
                z=np.asarray(data["z"], dtype=float),
                u=[np.asarray(v, dtype=float) for v in data["u"]],
                pending={int(worker_id): np.asarray(x, dtype=float) for worker_id, x in data.get("pending", {}).items()},
                trajectory=data["trajectory"],
                last_timeout=data.get("last_timeout"),
                failure_reason=data.get("failure_reason"),
            )

    def events(self, run_id: str) -> list[dict[str, Any]]:
        path = self.event_path(run_id)
        if not path.exists():
            raise NotFound("Run event log was not found.", {"run_id": run_id})
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def list_runs(self) -> list[str]:
        with self._lock:
            return sorted(path.name for path in self.root.iterdir() if path.is_dir())

    @property
    def lock(self) -> threading.RLock:
        return self._lock
