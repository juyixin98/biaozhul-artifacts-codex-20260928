"""Campaign orchestration.

A run executes:

1. three *guided* scenarios - forced histories for repeated-close,
   out-of-order-submit and cancel-after-delivery, each annotated with its
   scheduling rationale;
2. ``fuzz_steps`` seeded random commands (legal and deliberately illegal).

Every executed command produces a persisted step containing the scheduling
decision (seed/RNG state), model state before, pure expectation, raw
observation, model state after, and the verdict with its rationale.
"""
from __future__ import annotations

from pathlib import Path

from .contracts import Command, Finding
from .errors import ErrorCategory, MdbtError, ResourceExhausted
from .executor import Executor, InProcessTransport, Transport
from .oracle import FSMModel
from .reducer import discrepancy_signature, reduce_sequence
from .replay import replay_commands
from .scheduler import SchedulingDecision, Scheduler
from .scriptgen import render_replay_script
from .spec import MachineSpec
from .storage import EvidenceStore


GUIDED_SCENARIOS: list[tuple[str, list[str]]] = [
    (
        "guided_repeated_close",
        ["create", "submit", "approve", "deliver", "close", "close"],
    ),
    (
        "guided_out_of_order_submit",
        ["create", "submit", "approve", "submit"],
    ),
    (
        "guided_cancel_after_delivery",
        ["create", "submit", "approve", "deliver", "cancel"],
    ),
    (
        # Illegal construction on an existing resource; both SUTs must reject.
        "guided_double_create",
        ["create", "create"],
    ),
]


def _forced_decision(
    seq: int, command: Command, scenario: str, rationale: str, *, exists: bool, state: str
) -> SchedulingDecision:
    return SchedulingDecision(
        seq=seq,
        rng_state_before=[],
        mode=scenario,
        candidates=[command.action],
        chosen=command,
        rationale=rationale,
        rng_state_after=[],
        resource_exists=exists,
        model_state_hint=state,
    )


async def run_campaign(
    *,
    spec: MachineSpec,
    sut_name: str,
    transport_factory,
    store: EvidenceStore,
    run_id: str,
    seed: int = 20261007,
    fuzz_steps: int = 40,
    illegal_fraction: float = 0.45,
    replay_dir: str | Path = "artifacts/replays",
    sut_target: str = "in-process",
) -> dict:
    """Run guided scenarios plus fuzzing, persist evidence, reduce findings."""
    config = {
        "seed": seed,
        "fuzz_steps": fuzz_steps,
        "illegal_fraction": illegal_fraction,
        "guided_scenarios": [s for s, _ in GUIDED_SCENARIOS],
    }
    store.create_run(
        run_id=run_id,
        sut_name=sut_name,
        sut_target=sut_target,
        seed=seed,
        spec_json=spec.to_dict(),
        config=config,
    )

    replay_dir = Path(replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)

    # Each guided scenario gets a pristine model/executor so resource
    # bookkeeping cannot leak between forced histories.  The fuzz phase then
    # owns its own fresh pair as well.
    scheduler = Scheduler(spec, seed=seed, illegal_fraction=illegal_fraction)
    model = FSMModel.fresh(spec)
    executor = Executor.fresh(model, transport_factory())

    steps: list[dict] = []
    findings: list[Finding] = []
    commands_all: list[Command] = []
    seq = 0
    status = "completed"
    error_category = None
    error_message = None

    try:
        # ---------- guided scenarios (fresh resource per scenario) ----------
        for scenario_index, (scenario, actions) in enumerate(GUIDED_SCENARIOS):
            rid = f"guided-{scenario_index:02d}"
            model = FSMModel.fresh(spec)
            executor = Executor.fresh(model, transport_factory())
            state_hint = spec.initial_state
            for local_i, action in enumerate(actions, start=1):
                seq += 1
                exists = local_i > 1
                command = Command(resource_id=rid, action=action)
                rationale = (
                    f"guided scenario {scenario}: forced action {local_i}/{len(actions)}"
                )
                decision = _forced_decision(
                    seq, command, scenario, rationale,
                    exists=exists, state=state_hint,
                )
                record = await executor.step(seq, command, decision)
                steps.append(record)
                commands_all.append(command)
                store.append_step(run_id, record)
                obs_state = record["observation"].get("state")
                if record["observation"]["accepted"] and obs_state:
                    state_hint = obs_state
                if record["verdict"]["kind"] == "discrepancy":
                    await _register_finding(
                        finding_index=len(findings),
                        record=record,
                        commands=commands_all,
                        spec=spec,
                        transport_factory=transport_factory,
                        store=store,
                        run_id=run_id,
                        replay_dir=replay_dir,
                        findings=findings,
                    )

        # ---------- seeded fuzzing ----------
        model = FSMModel.fresh(spec)
        executor = Executor.fresh(model, transport_factory())
        for _ in range(fuzz_steps):
            if len(findings) >= 20:
                raise ResourceExhausted(
                    "finding cap (20) reached; stopping early",
                    detail={"findings": len(findings)},
                )
            seq += 1
            command, decision = scheduler.next_command()
            record = await executor.step(seq, command, decision)
            steps.append(record)
            commands_all.append(command)
            store.append_step(run_id, record)
            obs = record["observation"]
            scheduler.observe_result(command, obs["accepted"], obs.get("state"))
            if record["verdict"]["kind"] == "discrepancy":
                await _register_finding(
                    finding_index=len(findings),
                    record=record,
                    commands=commands_all,
                    spec=spec,
                    transport_factory=transport_factory,
                    store=store,
                    run_id=run_id,
                    replay_dir=replay_dir,
                    findings=findings,
                )

    except MdbtError as exc:
        status = "failed"
        error_category = exc.category.value
        error_message = exc.message

    summary = {
        "steps_executed": len(steps),
        "findings": len(findings),
        "finding_categories": sorted({f.category.value for f in findings}),
        "consistent_steps": sum(
            1 for s in steps if s["verdict"]["kind"] == "consistent"
        ),
        "discrepancy_steps": sum(
            1 for s in steps if s["verdict"]["kind"] == "discrepancy"
        ),
        "resyncs": sum(1 for s in steps if s["resynced"]),
        "final_model_states": dict(model.states),
        "seed": seed,
    }
    store.finish_run(
        run_id,
        status=status,
        summary=summary,
        error_category=error_category,
        error_message=error_message,
    )
    return {
        "run_id": run_id,
        "status": status,
        "summary": summary,
        "findings": [f.to_dict() for f in findings],
        "error": None
        if error_category is None
        else {"category": error_category, "message": error_message},
    }


async def _register_finding(
    *,
    finding_index: int,
    record: dict,
    commands: list[Command],
    spec: MachineSpec,
    transport_factory,
    store: EvidenceStore,
    run_id: str,
    replay_dir: Path,
    findings: list[Finding],
) -> None:
    verdict = record["verdict"]
    cat = (
        ErrorCategory(verdict["category"])
        if verdict.get("category")
        else ErrorCategory.COMPUTATION_FAILURE
    )
    title = _finding_title(cat, record)
    finding = Finding(
        category=cat,
        title=title,
        rationale=verdict["rationale"],
        trigger_seq=record["seq"],
        command=Command(**record["command"]),
        expected=record["expected"],
        observation=record["observation"],
    )

    trigger_seq = record["seq"]
    target_signature = discrepancy_signature(record)
    reduced_commands = list(commands[:trigger_seq])
    try:
        result = await reduce_sequence(
            spec,
            list(commands),
            trigger_seq=trigger_seq,
            target_signature=target_signature,
            transport_factory=transport_factory,
            id_prefix=f"reduce-{run_id}-"
        )
        reduced_commands = result.commands
        finding.reduced = True
        finding.reduced_length = len(result.commands)
        finding.reduction_attempted = True
        finding.reduction_log = result.log
        trigger_step = result.trigger_step
    except Exception as exc:  # reduction failure is recorded, not swallowed
        finding.reduction_attempted = True
        finding.reduction_log = [
            {"accepted": False, "reason": f"reduction error: {type(exc).__name__}: {exc}"}
        ]
        trigger_step = record

    script_name = f"replay_{run_id}_f{finding_index:02d}.py"
    # Must match the prefix used by reducer replays so the generated script
    # exercises exactly the validated minimised history.
    id_prefix = f"reduce-{run_id}-"
    script = render_replay_script(
        run_id=run_id,
        commands=reduced_commands,
        category=cat.value,
        trigger_step=trigger_step,
        script_name=script_name,
        id_prefix=id_prefix,
    )
    script_path = replay_dir / script_name
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    finding.replay_script = script
    finding.replay_path = str(script_path)

    finding_dict = finding.to_dict()
    store.add_finding(
        run_id=run_id,
        index=finding_index,
        finding_json=finding_dict,
        replay_script=script,
        replay_path=str(script_path),
    )
    findings.append(finding)


def _finding_title(category: ErrorCategory, record: dict) -> str:
    action = record["command"]["action"]
    state = record["model_state_before"]
    expected = record["expected"]
    obs = record["observation"]
    if expected["accepted"] and not obs["accepted"]:
        return f"SUT wrongly rejected {action!r} in state {state!r}"
    if not expected["accepted"] and obs["accepted"]:
        return f"SUT wrongly accepted illegal {action!r} in state {state!r}"
    return f"wrong outcome for {action!r} in state {state!r} ({category.value})"
