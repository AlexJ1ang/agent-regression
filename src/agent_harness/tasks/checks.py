"""Small deterministic checks over observed evidence, independent of the agent."""

from __future__ import annotations

import json
from typing import Any

from agent_harness.tasks.models import AgentReply, CheckResult, CheckSpec, Status


def read_path(value: Any, path: str) -> Any:
    """Read a dotted object path; an empty path selects the entire value."""
    for key in path.split(".") if path else []:
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(value, list) and key.isdigit():
            value = value[int(key)]
        else:
            raise KeyError(path)
    return value


def equal(expected: Any, actual: Any) -> bool:
    """JSON equality with booleans distinct from numbers."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and expected.keys() == actual.keys()
            and all(equal(value, actual[key]) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list) and len(expected) == len(actual)
            and all(equal(a, b) for a, b in zip(expected, actual))
        )
    return expected == actual


def arguments_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Expected object keys are required; extra agent arguments are allowed."""
    for key, value in expected.items():
        if key not in actual:
            return False
        if isinstance(value, dict):
            if not isinstance(actual[key], dict) or not arguments_match(value, actual[key]):
                return False
        elif not equal(value, actual[key]):
            return False
    return True


def check(spec: CheckSpec, reply: AgentReply) -> CheckResult:
    result = CheckResult(name=spec.name, status="review", expected=spec.expected)
    if spec.kind in {"tool_call", "tool_not_called"}:
        result.expected = {"tool": spec.tool, "arguments": spec.arguments}
        calls = [call for call in reply.tool_calls if call.name == spec.tool]
        result.actual = [call.arguments for call in calls]
        if spec.kind == "tool_not_called":
            passed = not calls
        else:
            passed = any(arguments_match(spec.arguments, call.arguments) for call in calls)
        conclusive = bool(calls) if spec.kind == "tool_not_called" else passed
        if not conclusive and not reply.tools_observed:
            result.reason = "Tool evidence is incomplete; absence cannot be inferred."
            return result
        result.status = "passed" if passed else "failed"
        result.reason = "Observed tool calls match the requirement." if passed else (
            "Required tool/arguments were not observed." if spec.kind == "tool_call"
            else "A forbidden tool was called."
        )
        return result

    try:
        if spec.kind == "answer_contains":
            actual: Any = reply.answer
            passed = spec.expected in actual
        else:
            if spec.kind == "state":
                if reply.state is None:
                    result.reason = "No independently observed environment state is available."
                    return result
                source: Any = reply.state
            else:
                source = json.loads(reply.answer)
            actual = read_path(source, spec.path)
            passed = equal(spec.expected, actual)
        result.actual = actual
        result.status = "passed" if passed else "failed"
        result.reason = "Observed value matches." if passed else "Observed value differs."
    except (KeyError, IndexError):
        result.status = "failed"
        result.reason = f"Required field is missing: {spec.path}"
    except (ValueError, TypeError) as exc:
        result.status = "failed"
        result.reason = f"Expected structured evidence could not be read: {exc}"
    return result


def combine_status(results: list[CheckResult]) -> Status:
    """An unresolved check never becomes success through an empty all()."""
    statuses = {result.status for result in results}
    for status in ("invalid", "failed", "error", "review"):
        if status in statuses:
            return status  # type: ignore[return-value]
    return "passed" if statuses else "review"
