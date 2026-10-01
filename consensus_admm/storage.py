"""Durable append-only JSONL storage with snapshots and local file locks."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import threading
from pathlib import Path

from .config import ServerSettings
from .errors import ResourceLimitError, StateConflictError
from .state import CompletedRound, RunState, apply_event, initial_state_from_created


class FileStore:
    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self.data_dir = settings.data_dir
        self.runs_dir = self.data_dir / "runs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        self._creation_lock = threading.RLock()

    def _lock(self, run_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(run_id, threading.RLock())

    def _run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def event_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "events.jsonl"

    def snapshot_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "state.snapshot.json"

    def list_run_ids(self) -> list[str]:
        return sorted(
            path.name
            for path in self.runs_dir.iterdir()
            if path.is_dir() and (path / "events.jsonl").exists()
        )

    def ensure_capacity(self) -> None:
        if len(self.list_run_ids()) >= self.settings.max_runs:
            raise ResourceLimitError(
                "maximum number of stored runs reached",
                code="run_limit_exceeded",
                details={"max_runs": self.settings.max_runs},
            )

    def exists(self, run_id: str) -> bool:
        return self.event_path(run_id).exists()

    def create_run(
        self, event: dict, follow_up_events: list[dict] | None = None
    ) -> RunState:
        run_id = event["data"]["run_id"]
        run_dir = self._run_dir(run_id)
        with self._creation_lock, self._lock(run_id):
            if run_dir.exists():
                raise StateConflictError(
                    "run already exists", code="run_already_exists"
                )
            self.ensure_capacity()
            run_dir.mkdir(parents=True)
            locked_event = dict(event)
            locked_event["seq"] = 1
            locked_event["run_id"] = run_id
            self._append_event_unlocked(run_id, locked_event)
            state = initial_state_from_created(locked_event)
            for follow_up in follow_up_events or []:
                state.last_event_id += 1
                stored = dict(follow_up)
                stored["seq"] = state.last_event_id
                stored["run_id"] = run_id
                self._append_event_unlocked(run_id, stored)
                apply_event(state, stored)
            self._write_snapshot_unlocked(run_id, state)
            return state

    def transaction(self, run_id: str):
        @contextlib.contextmanager
        def _transaction():
            lock_path = self._run_dir(run_id) / ".lock"
            with self._lock(run_id), lock_path.open("a+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                state = self.load_run(run_id)

                def commit(events: list[dict]) -> RunState:
                    return self.commit_events(run_id, state, events)

                yield state, commit

        return _transaction()

    def commit_events(
        self, run_id: str, state: RunState, events: list[dict]
    ) -> RunState:
        """Append events under the transaction's file lock, then snapshot."""
        for event in events:
            state.last_event_id += 1
            stored = dict(event)
            stored["seq"] = state.last_event_id
            stored["run_id"] = run_id
            self._append_event_unlocked(run_id, stored)
            apply_event(state, stored)
        self._write_snapshot_unlocked(run_id, state)
        return state

    def _append_event_unlocked(self, run_id: str, event: dict) -> None:
        path = self.event_path(run_id)
        size = path.stat().st_size if path.exists() else 0
        serialized = json.dumps(event, sort_keys=True, ensure_ascii=False)
        if size + len(serialized.encode("utf-8")) + 1 > self.settings.max_event_log_bytes:
            raise ResourceLimitError(
                "event log size limit reached",
                code="event_log_limit_exceeded",
                details={"max_bytes": self.settings.max_event_log_bytes},
            )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_snapshot_unlocked(self, run_id: str, state: RunState) -> None:
        snapshot = self._snapshot_dict(state)
        final_path = self.snapshot_path(run_id)
        temp_path = final_path.with_name(".state.snapshot.json.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, sort_keys=True, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, final_path)

    def load_run(self, run_id: str) -> RunState:
        event_file = self.event_path(run_id)
        if not event_file.exists():
            from .errors import NotFoundError

            raise NotFoundError(
                "run not found", code="run_not_found", details={"run_id": run_id}
            )

        snapshot_state: RunState | None = None
        snapshot_path = self.snapshot_path(run_id)
        if snapshot_path.exists():
            try:
                with snapshot_path.open("r", encoding="utf-8") as handle:
                    snapshot = json.load(handle)
                snapshot_state = self._state_from_snapshot(snapshot)
                expected_seq = int(snapshot_state.last_event_id) + 1
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
                snapshot_state = None
                expected_seq = 1

        state = snapshot_state
        with event_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ResourceLimitError(
                        "event log is corrupt",
                        code="corrupt_event_log",
                        details={"line": line_number},
                    ) from exc
                seq = int(event["seq"])
                if state is None:
                    if event["type"] != "run_created" or seq != 1:
                        raise StateConflictError(
                            "event log must begin with run_created seq=1"
                        )
                    state = initial_state_from_created(event)
                    expected_seq = 2
                else:
                    if seq < expected_seq:
                        continue
                    if seq != expected_seq:
                        raise StateConflictError(
                            "event sequence is not contiguous",
                            code="invalid_event_sequence",
                            details={"expected": expected_seq, "received": seq},
                        )
                    if event.get("run_id") and event["run_id"] != run_id:
                        raise StateConflictError(
                            "event belongs to a different run",
                            code="event_run_id_mismatch",
                        )
                    apply_event(state, event)
                    expected_seq += 1
        if state is None:
            raise StateConflictError("event log is empty", code="empty_event_log")
        return state

    def read_events(self, run_id: str) -> list[dict]:
        event_file = self.event_path(run_id)
        if not event_file.exists():
            from .errors import NotFoundError

            raise NotFoundError(
                "run not found", code="run_not_found", details={"run_id": run_id}
            )
        result = []
        with event_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    result.append(json.loads(line))
        return result

    @staticmethod
    def _snapshot_dict(state: RunState) -> dict:
        return {
            "run_id": state.run_id,
            "name": state.name,
            "problem": state.problem,
            "config": state.config,
            "problem_digest": state.problem_digest,
            "central_reference": state.central_reference,
            "status": state.status,
            "iteration": state.iteration,
            "rho": state.rho,
            "z": state.z,
            "u": state.u,
            "end_reason": state.end_reason,
            "created_at": state.created_at,
            "last_event_id": state.last_event_id,
            "rounds": {
                str(key): {
                    "iteration": value.iteration,
                    "rho": value.rho,
                    "rho_previous": value.rho_previous,
                    "opened_at": value.opened_at,
                    "deadline": value.deadline,
                    "deadline_ms": value.deadline_ms,
                    "token": value.token,
                    "z_pre": value.z_pre,
                    "u_pre": value.u_pre,
                    "status": value.status,
                    "timeout_at": value.timeout_at,
                    "timeout_reason": value.timeout_reason,
                    "resumed_at": value.resumed_at,
                    "submissions": {
                        worker_id: {
                            "worker_id": sub.worker_id,
                            "iteration": sub.iteration,
                            "rho": sub.rho,
                            "x": sub.x,
                            "objective": sub.objective,
                            "stationarity_residual": sub.stationarity_residual,
                            "solve_seconds": sub.solve_seconds,
                        }
                        for worker_id, sub in value.submissions.items()
                    },
                }
                for key, value in state.rounds.items()
            },
            "history": [
                {
                    "iteration": item.iteration,
                    "rho": item.rho,
                    "rho_previous": item.rho_previous,
                    "z_pre": item.z_pre,
                    "z": item.z,
                    "x": item.x,
                    "u_pre": item.u_pre,
                    "u": item.u,
                    "primal_residual": item.primal_residual,
                    "dual_residual": item.dual_residual,
                    "eps_primal": item.eps_primal,
                    "eps_dual": item.eps_dual,
                    "primal_ok": item.primal_ok,
                    "dual_ok": item.dual_ok,
                    "converged": item.converged,
                    "reason": item.reason,
                    "objective_total": item.objective_total,
                    "objectives": item.objectives,
                }
                for item in state.history
            ],
        }

    @staticmethod
    def _state_from_snapshot(snapshot: dict) -> RunState:
        from .state import RoundState, SubmissionState

        state = RunState(
            run_id=str(snapshot["run_id"]),
            name=str(snapshot["name"]),
            problem=snapshot["problem"],
            config=snapshot["config"],
            problem_digest=str(snapshot["problem_digest"]),
            central_reference=snapshot.get("central_reference"),
            status=str(snapshot["status"]),
            iteration=int(snapshot["iteration"]),
            rho=float(snapshot["rho"]),
            z=list(snapshot["z"]),
            u={key: list(value) for key, value in snapshot["u"].items()},
            end_reason=snapshot.get("end_reason"),
            created_at=snapshot.get("created_at"),
            last_event_id=int(snapshot["last_event_id"]),
        )
        for key, raw_round in snapshot["rounds"].items():
            round_state = RoundState(
                iteration=int(raw_round["iteration"]),
                rho=float(raw_round["rho"]),
                rho_previous=float(raw_round["rho_previous"]),
                opened_at=float(raw_round["opened_at"]),
                deadline=float(raw_round["deadline"]),
                deadline_ms=int(raw_round["deadline_ms"]),
                token=str(raw_round["token"]),
                z_pre=list(raw_round["z_pre"]),
                u_pre={
                    worker_id: list(value)
                    for worker_id, value in raw_round["u_pre"].items()
                },
                status=str(raw_round["status"]),
                timeout_at=raw_round.get("timeout_at"),
                timeout_reason=raw_round.get("timeout_reason"),
                resumed_at=raw_round.get("resumed_at"),
            )
            round_state.submissions = {
                worker_id: SubmissionState(
                    worker_id=str(sub["worker_id"]),
                    iteration=int(sub["iteration"]),
                    rho=float(sub["rho"]),
                    x=list(sub["x"]),
                    objective=float(sub["objective"]),
                    stationarity_residual=float(sub["stationarity_residual"]),
                    solve_seconds=float(sub.get("solve_seconds", 0.0)),
                )
                for worker_id, sub in raw_round["submissions"].items()
            }
            state.rounds[int(key)] = round_state
        for raw_item in snapshot.get("history", []):
            state.history.append(
                CompletedRound(
                    iteration=int(raw_item["iteration"]),
                    rho=float(raw_item["rho"]),
                    rho_previous=float(raw_item["rho_previous"]),
                    z_pre=list(raw_item["z_pre"]),
                    z=list(raw_item["z"]),
                    x={key: list(value) for key, value in raw_item["x"].items()},
                    u_pre={
                        key: list(value) for key, value in raw_item["u_pre"].items()
                    },
                    u={
                        key: list(value) for key, value in raw_item["u"].items()
                    },
                    primal_residual=float(raw_item["primal_residual"]),
                    dual_residual=float(raw_item["dual_residual"]),
                    eps_primal=float(raw_item["eps_primal"]),
                    eps_dual=float(raw_item["eps_dual"]),
                    primal_ok=bool(raw_item["primal_ok"]),
                    dual_ok=bool(raw_item["dual_ok"]),
                    converged=bool(raw_item["converged"]),
                    reason=str(raw_item["reason"]),
                    objective_total=float(raw_item["objective_total"]),
                    objectives={
                        key: float(value)
                        for key, value in raw_item["objectives"].items()
                    },
                )
            )
        return state
