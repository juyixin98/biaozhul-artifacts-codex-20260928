"""Mutation analysis orchestration kernel."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .equivalence import equivalence_basis
from .isolation import copy_project, remove_workspace, source_snapshot
from .models import (
    MutantFate,
    MutationReport,
    MutationRequest,
    MutantResult,
    RunEvent,
    RunStatus,
    SourceLocation,
)
from .mutator import Mutation, generate_mutations
from .runner import run_test_command


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _location(mutation: Mutation) -> SourceLocation:
    return SourceLocation(
        path=mutation.relative_path,
        line=mutation.location.line,
        column=mutation.location.column,
        end_line=mutation.location.end_line,
        end_column=mutation.location.end_column,
        node_type=mutation.location.node_type,
        original_snippet=mutation.before,
    )


@dataclass
class EventLog:
    events: list[RunEvent]

    def add(self, step: str, status: str, detail: str, location: str | None = None) -> None:
        self.events.append(
            RunEvent(
                sequence=len(self.events) + 1,
                timestamp=_utc_now(),
                step=step,
                status=status,
                detail=detail,
                location=location,
                engine_version=__version__,
            )
        )


def _result_from_command(
    mutation: Mutation,
    command_result,
    reason: str | None = None,
) -> MutantResult:
    stderr = _tail(command_result.stderr)
    if command_result.timed_out:
        fate = MutantFate.TIMEOUT
        category = "timeout"
        final_reason = reason or "mutant test process exceeded its mutant-specific timeout"
    elif command_result.category == "test_failure":
        fate = MutantFate.KILLED
        category = "test_failure"
        final_reason = reason or "one or more tests failed for the mutated copy"
    elif command_result.category == "startup_failure":
        fate = MutantFate.STARTUP_FAILED
        combined_logs = f"{command_result.stdout}\n{command_result.stderr}"
        category = "syntax_error" if "SyntaxError" in combined_logs else "startup_failure"
        final_reason = reason or "pytest collection or interpreter startup failed in the mutated copy"
    elif command_result.returncode == 0:
        fate = MutantFate.SURVIVED
        category = None
        final_reason = reason or "all tests passed for the mutated copy"
    else:
        fate = MutantFate.RUNNER_ERROR
        category = command_result.category
        final_reason = reason or f"test runner exited with unclassified code {command_result.returncode}"
    return MutantResult(
        mutant_id=mutation.mutant_id,
        operator_family=mutation.operator_family,
        before=mutation.before,
        after=mutation.after,
        location=_location(mutation),
        fate=fate,
        determination="undetermined" if fate == MutantFate.EQUIVALENT_UNDETERMINED else "determined",
        exit_code=command_result.returncode,
        duration_seconds=round(command_result.duration_seconds, 6),
        reason=final_reason,
        failure_category=category,
        stderr_tail=stderr if stderr else None,
        workspace=None,
    )


def _static_result(
    mutation: Mutation,
    fate: MutantFate,
    reason: str,
    *,
    basis: str | None = None,
    failure_category: str | None = None,
    stderr_tail: str | None = None,
) -> MutantResult:
    return MutantResult(
        mutant_id=mutation.mutant_id,
        operator_family=mutation.operator_family,
        before=mutation.before,
        after=mutation.after,
        location=_location(mutation),
        fate=fate,
        determination="undetermined" if fate == MutantFate.EQUIVALENT_UNDETERMINED else "determined",
        exit_code=None,
        duration_seconds=None,
        reason=reason,
        failure_category=failure_category,
        stderr_tail=stderr_tail,
        equivalent_basis=basis,
        workspace=None,
    )


def _snapshot_payload(snapshot) -> dict:
    return {
        "file_count": len(snapshot),
        "files": [item.model_dump() for item in snapshot],
    }


def run_mutation_analysis(
    request: MutationRequest,
    *,
    work_root: Path,
    request_id: str | None = None,
) -> MutationReport:
    request_id = request_id or f"req-{uuid.uuid4().hex[:16]}"
    project_root = request.project_path.resolve()
    work_root = work_root.resolve()
    if work_root == project_root or work_root.is_relative_to(project_root):
        raise ValueError("work_root must be outside project_path to prevent recursive project copies")
    run_root = work_root / request_id
    run_root.mkdir(parents=True, exist_ok=True)
    events = EventLog([])
    created_at = _utc_now()
    before = source_snapshot(request.project_path)
    events.add(
        "snapshot",
        "completed",
        f"recorded {len(before)} original Python file hashes",
        str(request.project_path),
    )

    mutations = generate_mutations(
        request.project_path,
        request.source_paths,
        request.exclude_patterns,
        request.operators,
    )
    events.add(
        "generate",
        "completed",
        f"generated {len(mutations)} AST mutants",
        f"{__version__}:{request.project_path}",
    )

    baseline_workspace = run_root / "baseline"
    copy_project(request.project_path, baseline_workspace)
    events.add("baseline-copy", "completed", "created isolated baseline copy", str(baseline_workspace))
    baseline_result = run_test_command(
        baseline_workspace,
        request.test_command,
        request.baseline_timeout_seconds,
    )
    baseline_payload = {
        "returncode": baseline_result.returncode,
        "category": baseline_result.category,
        "duration_seconds": round(baseline_result.duration_seconds, 6),
        "timed_out": baseline_result.timed_out,
        "stderr_tail": _tail(baseline_result.stderr),
    }
    if baseline_result.timed_out:
        remove_workspace(baseline_workspace)
        after = source_snapshot(request.project_path)
        events.add("baseline", "timeout", "baseline exceeded its distinct baseline timeout")
        return MutationReport(
            request_id=request_id,
            status=RunStatus.BASELINE_TIMEOUT,
            project_path=str(request.project_path),
            engine_version=__version__,
            baseline=baseline_payload,
            timeouts={
                "baseline_seconds": request.baseline_timeout_seconds,
                "mutant_seconds": request.mutant_timeout_seconds,
            },
            summary={fate.value: 0 for fate in MutantFate},
            uncertainties=[],
            failures=[],
            mutants=[],
            source_before=_snapshot_payload(before),
            source_after=_snapshot_payload(after),
            source_unchanged=before == after,
            events=events.events,
            created_at=created_at,
            completed_at=_utc_now(),
        )
    if baseline_result.returncode != 0:
        remove_workspace(baseline_workspace)
        after = source_snapshot(request.project_path)
        events.add("baseline", "failed", f"baseline exited {baseline_result.returncode}")
        return MutationReport(
            request_id=request_id,
            status=RunStatus.BASELINE_FAILED,
            project_path=str(request.project_path),
            engine_version=__version__,
            baseline=baseline_payload,
            timeouts={
                "baseline_seconds": request.baseline_timeout_seconds,
                "mutant_seconds": request.mutant_timeout_seconds,
            },
            summary={fate.value: 0 for fate in MutantFate},
            uncertainties=[],
            failures=[],
            mutants=[],
            source_before=_snapshot_payload(before),
            source_after=_snapshot_payload(after),
            source_unchanged=before == after,
            events=events.events,
            created_at=created_at,
            completed_at=_utc_now(),
        )
    remove_workspace(baseline_workspace)
    events.add("baseline", "passed", "baseline tests passed in an isolated copy")

    results: list[MutantResult] = []
    for index, mutation in enumerate(mutations, start=1):
        workspace = run_root / f"mutant-{index:04d}"
        copy_project(request.project_path, workspace)
        target = workspace / mutation.relative_path
        target.write_text(mutation.mutated_source, encoding="utf-8")
        try:
            compile(mutation.mutated_source, str(target), "exec")
        except SyntaxError as exc:
            result = _static_result(
                mutation,
                MutantFate.SYNTAX_INVALID,
                f"mutated source does not compile: {exc.msg}",
                failure_category="syntax_error",
                stderr_tail=f"{exc.msg} at line {exc.lineno}",
            )
            result = result.model_copy(update={"workspace": str(workspace)})
            results.append(result)
            remove_workspace(workspace)
            events.add(
                "mutant",
                "syntax_invalid",
                f"{mutation.mutant_id} rejected before pytest: {exc.msg}",
                str(target),
            )
            continue

        basis = equivalence_basis(mutation)
        if basis is not None:
            result = _static_result(
                mutation,
                MutantFate.EQUIVALENT_UNDETERMINED,
                "static evidence indicates an equivalent mutant; fate remains undetermined",
                basis=basis,
            )
            result = result.model_copy(update={"workspace": str(workspace)})
            results.append(result)
            remove_workspace(workspace)
            events.add("mutant", "equivalent_undetermined", f"{mutation.mutant_id}: {basis}", str(target))
            continue

        command_result = run_test_command(
            workspace,
            request.test_command,
            request.mutant_timeout_seconds,
        )
        result = _result_from_command(mutation, command_result)
        result = result.model_copy(update={"workspace": str(workspace)})
        results.append(result)
        remove_workspace(workspace)
        events.add(
            "mutant",
            result.fate.value,
            f"{mutation.mutant_id} exited {result.exit_code} in {result.duration_seconds}s",
            str(target),
        )

    after = source_snapshot(request.project_path)
    summary = {fate.value: 0 for fate in MutantFate}
    for result in results:
        summary[result.fate.value] += 1
    failures = [
        item
        for item in results
        if item.fate in {MutantFate.SYNTAX_INVALID, MutantFate.STARTUP_FAILED, MutantFate.RUNNER_ERROR}
    ]
    uncertainties = [item for item in results if item.fate == MutantFate.EQUIVALENT_UNDETERMINED]
    completed_at = _utc_now()
    events.add("snapshot", "completed", f"verified {len(after)} original Python file hashes after run")
    return MutationReport(
        request_id=request_id,
        status=RunStatus.COMPLETED,
        project_path=str(request.project_path),
        engine_version=__version__,
        baseline=baseline_payload,
        timeouts={
            "baseline_seconds": request.baseline_timeout_seconds,
            "mutant_seconds": request.mutant_timeout_seconds,
        },
        summary=summary,
        uncertainties=uncertainties,
        failures=failures,
        mutants=results,
        source_before=_snapshot_payload(before),
        source_after=_snapshot_payload(after),
        source_unchanged=before == after,
        events=events.events,
        created_at=created_at,
        completed_at=completed_at,
    )
