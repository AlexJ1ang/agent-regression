"""Execute fixed multi-turn tasks and persist evidence after every turn."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from copy import deepcopy
from pathlib import Path

from agent_harness.tasks.checks import check, combine_status
from agent_harness.tasks.models import (
    AgentReply,
    CheckResult,
    TaskAdapter,
    TaskResult,
    TaskRun,
    TaskSpec,
    TaskSuite,
    TurnResult,
)


def fingerprint(suite: TaskSuite, target_identity: str) -> str:
    """Bind saved evidence to execution inputs, independently of check revisions."""
    if not target_identity.strip():
        raise ValueError("target_identity must not be empty")
    definition = suite.model_dump(mode="json")
    for task in definition["tasks"]:
        task.pop("checks")
        for turn in task["turns"]:
            turn.pop("checks")
    data = {"suite": definition, "target": target_identity}
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def checks_fingerprint(suite: TaskSuite) -> str:
    checks = [
        {"id": task.id, "checks": [item.model_dump(mode="json") for item in task.checks],
         "turns": [[item.model_dump(mode="json") for item in turn.checks] for turn in task.turns]}
        for task in suite.tasks
    ]
    return hashlib.sha256(json.dumps(checks, sort_keys=True).encode()).hexdigest()


def save_run(run: TaskRun, output_dir: Path) -> Path:
    """Atomically replace run.json, keeping the last intact checkpoint on failure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "run.json"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_dir, prefix=".run-", delete=False,
        ) as file:
            temporary = file.name
            file.write(run.model_dump_json(indent=2))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _validate_run(suite: TaskSuite, run: TaskRun, target_identity: str) -> None:
    if run.suite_id != suite.id or run.fingerprint != fingerprint(suite, target_identity):
        raise ValueError("saved run does not match this suite and target")
    tasks = {task.id: task for task in suite.tasks}
    seen: set[str] = set()
    for result in run.results:
        if result.task_id not in tasks or result.task_id in seen:
            raise ValueError("saved run contains unknown or duplicate task ids")
        seen.add(result.task_id)
        task = tasks[result.task_id]
        if len(result.turns) > len(task.turns):
            raise ValueError("saved run contains too many turns")
        for index, turn in enumerate(result.turns):
            if turn.index != index + 1 or turn.message != task.turns[index].message:
                raise ValueError("saved run turn messages do not match the suite")
        if result.completed and (
            len(result.turns) != len(task.turns)
            or result.error
            or any(turn.reply is None or turn.error for turn in result.turns)
        ):
            raise ValueError("saved completed task has incomplete execution evidence")


def _session_error(task: TaskSpec, result: TaskResult, turn: TurnResult) -> str:
    if turn.reply is None or len(task.turns) == 1:
        return ""
    session_id = turn.reply.conversation_id
    if not session_id or not session_id.strip():
        return "multi-turn task requires a nonempty conversation_id"
    previous = [item.reply for item in result.turns[:-1] if item.reply is not None]
    if previous and session_id != previous[-1].conversation_id:
        return "conversation_id changed within the task"
    return ""


def _score_task(task: TaskSpec, result: TaskResult) -> None:
    """Rebuild check results from stored evidence, without calling the agent."""
    all_checks: list[CheckResult] = []
    for index, turn in enumerate(result.turns):
        turn.checks = []
        for spec in task.turns[index].checks:
            if turn.reply is None or turn.error:
                item = CheckResult(
                    name=spec.name, status="review", expected=spec.expected,
                    reason="turn has no valid execution evidence",
                )
            else:
                item = check(spec, turn.reply)
            turn.checks.append(item)
        all_checks.extend(turn.checks)

    result.checks = []
    replies = [turn.reply for turn in result.turns if turn.reply is not None]
    aggregate: AgentReply | None = None
    if replies:
        aggregate = replies[-1].model_copy(update={
            "tool_calls": [call for reply in replies for call in reply.tool_calls],
            "tools_observed": all(reply.tools_observed for reply in replies),
        })
    for spec in task.checks:
        if aggregate is not None and result.completed:
            item = check(spec, aggregate)
        else:
            item = CheckResult(
                name=spec.name, status="review", expected=spec.expected,
                reason="task did not reach its final turn",
            )
            # An observed forbidden call is conclusive even in a partial run.
            if aggregate is not None and spec.kind == "tool_not_called":
                observed = check(spec, aggregate)
                if observed.status == "failed":
                    item = observed
        result.checks.append(item)
    all_checks.extend(result.checks)
    if result.error:
        all_checks.append(CheckResult(name="execution", status="error", reason=result.error))
    elif not result.completed:
        all_checks.append(CheckResult(name="execution", status="review", reason="task is incomplete"))
    result.status = combine_status(all_checks)


async def run_suite(
    suite: TaskSuite,
    adapter: TaskAdapter,
    output_dir: Path,
    *,
    target_identity: str,
    resume: bool = False,
) -> TaskRun:
    """Run tasks in fresh conversations, retaining every response and check.

    Resume reuses completed tasks only. An interrupted task starts again in a
    fresh conversation; arbitrary remote agent state is never assumed recoverable.
    """
    output_dir = Path(output_dir)
    path = output_dir / "run.json"
    expected_fingerprint = fingerprint(suite, target_identity)
    if resume:
        if not path.exists():
            raise ValueError("cannot resume: run.json does not exist")
        run = TaskRun.model_validate_json(path.read_text(encoding="utf-8"))
        _validate_run(suite, run, target_identity)
        if run.checks_fingerprint != checks_fingerprint(suite):
            raise ValueError("saved checks changed; use rejudge or a new output directory")
    else:
        if path.exists():
            raise ValueError("run.json already exists; choose a new output directory or resume")
        run = TaskRun(
            suite_id=suite.id, fingerprint=expected_fingerprint,
            checks_fingerprint=checks_fingerprint(suite),
        )
    existing = {result.task_id: result for result in run.results}
    for task in suite.tasks:
        previous = existing.get(task.id)
        if previous is not None and previous.completed:
            continue
        result = TaskResult(task_id=task.id)
        if previous is not None:
            run.results[run.results.index(previous)] = result
        else:
            run.results.append(result)
        save_run(run, output_dir)
        conversation_id: str | None = None
        for index, step in enumerate(task.turns, 1):
            turn = TurnResult(index=index, message=step.message)
            result.turns.append(turn)
            # Save before awaiting so cancellation leaves an incomplete checkpoint.
            save_run(run, output_dir)
            started = time.perf_counter()
            try:
                # Only public inputs cross the adapter boundary.
                reply = await adapter.send(step.message, deepcopy(task.context), conversation_id)
                turn.reply = AgentReply.model_validate(reply)
                turn.error = _session_error(task, result, turn)
            except Exception as exc:
                turn.error = f"{type(exc).__name__}: {exc}"
            finally:
                turn.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            if turn.error:
                result.error = turn.error
                _score_task(task, result)
                save_run(run, output_dir)
                break
            assert turn.reply is not None
            conversation_id = turn.reply.conversation_id
            result.completed = index == len(task.turns)
            _score_task(task, result)
            save_run(run, output_dir)
    return run


def rejudge(suite: TaskSuite, run: TaskRun, *, target_identity: str) -> TaskRun:
    """Recheck recorded replies without sending any request to the target."""
    _validate_run(suite, run, target_identity)
    judged = run.model_copy(deep=True)
    judged.checks_fingerprint = checks_fingerprint(suite)
    recorded = {result.task_id: result for result in judged.results}
    judged.results = []
    for task in suite.tasks:
        result = recorded.get(task.id, TaskResult(task_id=task.id))
        _score_task(task, result)
        judged.results.append(result)
    return judged
