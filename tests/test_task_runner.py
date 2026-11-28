from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from agent_harness.models import ToolCallRecord
from agent_harness.tasks.models import (
    AgentReply,
    CheckSpec,
    TaskRun,
    TaskSpec,
    TaskSuite,
    TaskTurn,
)
from agent_harness.tasks.report import render_report
from agent_harness.tasks.runner import rejudge, run_suite


class SequenceAdapter:
    def __init__(self, replies: list[AgentReply | BaseException]):
        self.replies = iter(replies)
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []

    async def send(
        self, message: str, context: dict[str, Any], conversation_id: str | None,
    ) -> AgentReply:
        self.calls.append((message, context, conversation_id))
        item = next(self.replies)
        if isinstance(item, BaseException):
            raise item
        return item


def contains(value: str = "ok") -> CheckSpec:
    return CheckSpec(name="answer", kind="answer_contains", expected=value)


def suite(*tasks: TaskSpec) -> TaskSuite:
    return TaskSuite(id="business", tasks=list(tasks))


def task(task_id: str = "one", messages: tuple[str, ...] = ("hello",)) -> TaskSpec:
    return TaskSpec(
        id=task_id, goal="private goal", context={"role": "reader"},
        turns=[TaskTurn(message=message, checks=[contains()]) for message in messages],
    )


@pytest.mark.asyncio
async def test_real_session_is_reused_but_tasks_are_isolated(tmp_path: Path) -> None:
    definitions = suite(task("one", ("first", "second")), task("two"))
    adapter = SequenceAdapter([
        AgentReply(answer="ok", conversation_id="server-generated"),
        AgentReply(answer="ok", conversation_id="server-generated"),
        AgentReply(answer="ok"),
    ])
    result = await run_suite(definitions, adapter, tmp_path, target_identity="endpoint-v1")
    assert [call[2] for call in adapter.calls] == [None, "server-generated", None]
    assert adapter.calls[0] == ("first", {"role": "reader"}, None)
    assert [row.status for row in result.results] == ["passed", "passed"]
    assert TaskRun.model_validate_json((tmp_path / "run.json").read_text()) == result


@pytest.mark.asyncio
@pytest.mark.parametrize("session", [None, "", " "])
async def test_multiturn_missing_session_stops_before_next_message(
    tmp_path: Path, session: str | None,
) -> None:
    adapter = SequenceAdapter([AgentReply(answer="ok", conversation_id=session)])
    result = await run_suite(suite(task(messages=("a", "b"))), adapter, tmp_path, target_identity="v1")
    assert len(adapter.calls) == 1
    assert result.results[0].status == "error"
    assert not result.results[0].completed
    assert "conversation_id" in result.results[0].error


@pytest.mark.asyncio
async def test_changed_session_keeps_evidence_and_stops_group(tmp_path: Path) -> None:
    adapter = SequenceAdapter([
        AgentReply(answer="ok", conversation_id="one"),
        AgentReply(answer="ok", conversation_id="two"),
    ])
    result = await run_suite(suite(task(messages=("a", "b", "c"))), adapter, tmp_path, target_identity="v1")
    item = result.results[0]
    assert item.status == "error"
    assert len(adapter.calls) == 2
    assert item.turns[1].reply.conversation_id == "two"
    assert item.turns[1].checks[0].status == "review"


@pytest.mark.asyncio
async def test_tool_checks_span_task_but_turn_checks_only_see_their_turn(tmp_path: Path) -> None:
    first_call = ToolCallRecord(name="lookup", arguments={"device": "A"})
    task_spec = task(messages=("a", "b"))
    task_spec.turns[0].checks = [CheckSpec(name="lookup A", kind="tool_call", tool="lookup", arguments={"device": "A"})]
    task_spec.checks = [
        CheckSpec(name="queried A", kind="tool_call", tool="lookup", arguments={"device": "A"}),
        CheckSpec(name="final total", kind="answer_json", path="total", expected=20),
        CheckSpec(name="actual ticket", kind="state", path="ticket.status", expected="open"),
    ]
    task_spec.turns[1].checks = []
    adapter = SequenceAdapter([
        AgentReply(answer="looking up", conversation_id="s", tools_observed=True, tool_calls=[first_call]),
        AgentReply(answer='{"total":20}', conversation_id="s", tools_observed=True, state={"ticket": {"status": "open"}}),
    ])
    result = await run_suite(suite(task_spec), adapter, tmp_path, target_identity="v1")
    assert result.results[0].status == "passed"
    assert [item.status for item in result.results[0].checks] == ["passed"] * 3
    assert result.results[0].turns[1].reply.tool_calls == []


@pytest.mark.asyncio
async def test_unobservable_tools_are_review_even_with_good_answer(tmp_path: Path) -> None:
    definition = task()
    definition.checks = [CheckSpec(name="lookup", kind="tool_call", tool="lookup")]
    result = await run_suite(suite(definition), SequenceAdapter([AgentReply(answer="ok")]), tmp_path, target_identity="v1")
    assert result.results[0].status == "review"
    assert result.results[0].checks[0].status == "review"


@pytest.mark.asyncio
async def test_no_checks_never_becomes_passed(tmp_path: Path) -> None:
    definition = task()
    definition.turns[0].checks = []
    result = await run_suite(suite(definition), SequenceAdapter([AgentReply(answer="ok")]), tmp_path, target_identity="v1")
    assert result.results[0].status == "review"


@pytest.mark.asyncio
async def test_known_failed_check_is_preserved_after_execution_error(tmp_path: Path) -> None:
    definitions = suite(task("one", ("first", "second")), task("two"))
    adapter = SequenceAdapter([
        AgentReply(answer="wrong", conversation_id="s"), TimeoutError("service timeout"),
        AgentReply(answer="ok"),
    ])
    result = await run_suite(definitions, adapter, tmp_path, target_identity="v1")
    first = result.results[0]
    assert first.status == "failed"
    assert first.turns[0].checks[0].status == "failed"
    assert "TimeoutError" in first.error
    assert not first.completed
    assert result.results[1].status == "passed"
    assert adapter.calls[-1][2] is None


@pytest.mark.asyncio
async def test_execution_error_does_not_claim_final_answer_failure(tmp_path: Path) -> None:
    definition = task(messages=("first", "second"))
    definition.checks = [CheckSpec(name="final", kind="answer_json", path="value", expected=3)]
    adapter = SequenceAdapter([AgentReply(answer="ok", conversation_id="s"), RuntimeError("unavailable")])
    result = await run_suite(suite(definition), adapter, tmp_path, target_identity="v1")
    assert result.results[0].status == "error"
    assert result.results[0].checks[0].status == "review"


@pytest.mark.asyncio
async def test_resume_skips_completed_and_restarts_incomplete_in_fresh_session(tmp_path: Path) -> None:
    definitions = suite(task("complete"), task("retry", ("a", "b")))
    initial = SequenceAdapter([
        AgentReply(answer="ok"), AgentReply(answer="ok", conversation_id="old"),
        asyncio.CancelledError(),
    ])
    with pytest.raises(asyncio.CancelledError):
        await run_suite(definitions, initial, tmp_path, target_identity="v1")
    checkpoint = TaskRun.model_validate_json((tmp_path / "run.json").read_text())
    assert checkpoint.results[0].completed
    assert not checkpoint.results[1].completed
    assert checkpoint.results[1].turns[-1].reply is None
    resumed = SequenceAdapter([
        AgentReply(answer="ok", conversation_id="new"),
        AgentReply(answer="ok", conversation_id="new"),
    ])
    result = await run_suite(definitions, resumed, tmp_path, target_identity="v1", resume=True)
    assert [call[0] for call in resumed.calls] == ["a", "b"]
    assert [call[2] for call in resumed.calls] == [None, "new"]
    assert [row.status for row in result.results] == ["passed", "passed"]


@pytest.mark.asyncio
async def test_resume_rejects_changed_suite_target_and_output_overwrite(tmp_path: Path) -> None:
    definitions = suite(task())
    await run_suite(definitions, SequenceAdapter([AgentReply(answer="ok")]), tmp_path, target_identity="v1")
    unused = SequenceAdapter([])
    with pytest.raises(ValueError, match="already exists"):
        await run_suite(definitions, unused, tmp_path, target_identity="v1")
    with pytest.raises(ValueError, match="does not match"):
        await run_suite(definitions, unused, tmp_path, target_identity="v2", resume=True)
    changed = definitions.model_copy(deep=True)
    changed.tasks[0].turns[0].message = "changed"
    with pytest.raises(ValueError, match="does not match"):
        await run_suite(changed, unused, tmp_path, target_identity="v1", resume=True)
    assert not unused.calls


@pytest.mark.asyncio
async def test_rejudge_uses_saved_full_evidence_and_keeps_original(tmp_path: Path) -> None:
    definitions = suite(task())
    original = await run_suite(definitions, SequenceAdapter([AgentReply(answer="ok")]), tmp_path, target_identity="v1")
    original.results[0].status = "failed"
    original.results[0].turns[0].checks[0].status = "failed"
    judged = rejudge(definitions, original, target_identity="v1")
    assert judged.results[0].status == "passed"
    assert original.results[0].status == "failed"
    with pytest.raises(ValueError, match="does not match"):
        rejudge(definitions, original, target_identity="another target")


@pytest.mark.asyncio
async def test_report_escapes_messages_answers_and_check_evidence(tmp_path: Path) -> None:
    attack = '<script>alert("x")</script>'
    definition = task(messages=(attack,))
    definition.turns[0].checks = [contains(attack)]
    result = await run_suite(suite(definition), SequenceAdapter([AgentReply(answer=attack)]), tmp_path, target_identity="v1")
    report = render_report(result, tmp_path).read_text()
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert "passed: 1" in report
    assert "review: 0" in report


@pytest.mark.asyncio
async def test_rejudge_can_change_checks_without_rerunning_agent(tmp_path: Path) -> None:
    definitions = suite(task())
    original = await run_suite(definitions, SequenceAdapter([AgentReply(answer="ok")]), tmp_path, target_identity="v1")
    revised = definitions.model_copy(deep=True)
    revised.tasks[0].turns[0].checks = [contains("not present")]
    judged = rejudge(revised, original, target_identity="v1")
    assert judged.results[0].status == "failed"
    assert original.results[0].status == "passed"
    assert judged.checks_fingerprint != original.checks_fingerprint
    with pytest.raises(ValueError, match="checks changed"):
        await run_suite(revised, SequenceAdapter([]), tmp_path, target_identity="v1", resume=True)
    revised.tasks[0].turns[0].message = "different execution"
    with pytest.raises(ValueError, match="does not match"):
        rejudge(revised, original, target_identity="v1")


@pytest.mark.asyncio
async def test_incomplete_checkpoint_cannot_be_rejudged_as_passed(tmp_path: Path) -> None:
    definitions = suite(task(messages=("a", "b")))
    adapter = SequenceAdapter([AgentReply(answer="ok", conversation_id="s"), asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await run_suite(definitions, adapter, tmp_path, target_identity="v1")
    saved = TaskRun.model_validate_json((tmp_path / "run.json").read_text())
    assert rejudge(definitions, saved, target_identity="v1").results[0].status == "review"


@pytest.mark.asyncio
async def test_rejudge_exposes_tasks_not_started_before_interruption(tmp_path: Path) -> None:
    definitions = suite(task("first"), task("not-started"))
    original = await run_suite(
        definitions, SequenceAdapter([AgentReply(answer="ok"), AgentReply(answer="ok")]),
        tmp_path, target_identity="v1",
    )
    original.results = original.results[:1]
    judged = rejudge(definitions, original, target_identity="v1")
    assert [item.task_id for item in judged.results] == ["first", "not-started"]
    assert [item.status for item in judged.results] == ["passed", "review"]
    assert not judged.results[1].completed
