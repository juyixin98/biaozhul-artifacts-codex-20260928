"""Application service that coordinates scheduler, runner, reduction, reports."""

from __future__ import annotations

from collections import defaultdict

from .domain import RunRequest, RunResult, StepRecord
from .http_adapter import HttpSutAdapter
from .replay import build_replay_script
from .runner import (
    execute_plan,
    make_run_result,
    reduce_failing_plan,
)
from .scheduler import create_plan
from .storage import EvidenceStore


def run_model_test(
    request: RunRequest,
    store: EvidenceStore,
    *,
    save: bool = True,
) -> RunResult:
    plans = create_plan(request)
    by_episode: dict[int, list] = defaultdict(list)
    for plan in plans:
        by_episode[plan.episode].append(plan)

    all_records: list[StepRecord] = []
    first_failure: StepRecord | None = None
    replay_script: str | None = None
    reduced_count: int | None = None

    def factory() -> HttpSutAdapter:
        return HttpSutAdapter(request.base_url)

    for episode in sorted(by_episode):
        episode_plans = by_episode[episode]
        with factory() as adapter:
            records, failure = execute_plan(
                request.model,
                episode_plans,
                adapter,
                request.synthetic_fault.value
                if request.synthetic_fault
                else None,
            )
        all_records.extend(records)
        if failure is not None and first_failure is None:
            first_failure = failure
            reduced = reduce_failing_plan(
                request.model,
                episode_plans,
                failure,
                factory,
                request.synthetic_fault.value
                if request.synthetic_fault
                else None,
            )
            reduced_count = len(episode_plans)
            replay_script = build_replay_script(
                request.base_url,
                request.model.resource_id_placeholder,
                reduced,
                request.synthetic_fault.value
                if request.synthetic_fault
                else None,
            )

    result = make_run_result(
        run_id=None,
        seed=request.seed,
        scheduler_name=request.scheduler.value,
        records=all_records,
        failure=first_failure,
        replay_script=replay_script,
        reduced_from_step_count=reduced_count,
    )
    if save:
        result.run_id = store.save_run(request, result)
    return result
