import pytest
from pydantic import ValidationError

from agent_harness.models import ToolCallRecord
from agent_harness.tasks.checks import check, combine_status
from agent_harness.tasks.models import AgentReply, CheckResult, CheckSpec, TaskSuite


def test_right_tool_wrong_device_fails():
    spec = CheckSpec(name="correct scope", kind="tool_call", tool="query", arguments={"device": "B"})
    reply = AgentReply(answer="done", tools_observed=True, tool_calls=[
        ToolCallRecord(name="query", arguments={"device": "A"}),
    ])
    assert check(spec, reply).status == "failed"
    reply.tool_calls.append(ToolCallRecord(name="query", arguments={"device": "B", "month": 9}))
    assert check(spec, reply).status == "passed"


def test_unavailable_tools_differ_from_observed_empty_tools():
    spec = CheckSpec(name="no mutation", kind="tool_not_called", tool="delete")
    assert check(spec, AgentReply(answer="ok")).status == "review"
    assert check(spec, AgentReply(answer="ok", tools_observed=True)).status == "passed"


def test_swapped_numbers_and_boolean_number_substitution_fail():
    spec = CheckSpec(name="A value", kind="answer_json", path="A", expected=10)
    assert check(spec, AgentReply(answer='{"A":20,"B":10}')).status == "failed"
    assert check(spec, AgentReply(answer='{"A":10,"B":20}')).status == "passed"
    spec.expected = True
    assert check(spec, AgentReply(answer='{"A":1}')).status == "failed"


def test_state_must_be_observed_and_cannot_be_inferred_from_answer():
    spec = CheckSpec(name="ticket exists", kind="state", path="ticket.status", expected="open")
    assert check(spec, AgentReply(answer='{"ticket":{"status":"open"}}')).status == "review"
    reply = AgentReply(answer="done", state={"ticket": {"status": "closed"}})
    assert check(spec, reply).status == "failed"


def test_missing_json_field_is_failure_but_no_checks_is_review():
    spec = CheckSpec(name="result", kind="answer_json", path="result", expected=1)
    assert check(spec, AgentReply(answer='{}')).status == "failed"
    assert combine_status([]) == "review"
    assert combine_status([CheckResult(name="missing", status="review")]) == "review"
    assert combine_status([CheckResult(name="bad", status="failed"), CheckResult(name="missing", status="review")]) == "failed"


@pytest.mark.parametrize("tasks", [[], [{"id": "../escape", "goal": "g", "turns": [{"message": "q"}]}]])
def test_invalid_suites_cannot_silently_pass(tasks):
    with pytest.raises(ValidationError):
        TaskSuite(id="suite", tasks=tasks)


def test_duplicate_task_ids_and_unknown_fields_are_rejected():
    task = {"id": "a", "goal": "g", "turns": [{"message": "q"}]}
    with pytest.raises(ValidationError):
        TaskSuite(id="suite", tasks=[task, task])
    with pytest.raises(ValidationError):
        TaskSuite.model_validate({"id": "suite", "tasks": [task], "typo": True})


@pytest.mark.parametrize("kind", ["answer_json", "state", "answer_contains"])
def test_missing_expected_value_cannot_accidentally_certify_null(kind):
    with pytest.raises(ValidationError, match="explicit expected"):
        CheckSpec(name="missing expected", kind=kind)


def test_explicit_null_is_a_valid_expectation():
    spec = CheckSpec(name="null result", kind="answer_json", expected=None)
    assert check(spec, AgentReply(answer="null")).status == "passed"
    state_spec = CheckSpec(name="not assigned", kind="state", path="assignee", expected=None)
    assert check(state_spec, AgentReply(answer="ok", state={"assignee": None})).status == "passed"


def test_observed_call_is_conclusive_even_when_other_tool_evidence_is_missing():
    reply = AgentReply(answer="ok", tools_observed=False, tool_calls=[
        ToolCallRecord(name="delete", arguments={"id": "A"}),
    ])
    forbidden = CheckSpec(name="no delete", kind="tool_not_called", tool="delete")
    assert check(forbidden, reply).status == "failed"
    required = CheckSpec(name="delete A", kind="tool_call", tool="delete", arguments={"id": "A"})
    assert check(required, reply).status == "passed"
    required.arguments = {"id": "B"}
    assert check(required, reply).status == "review"
