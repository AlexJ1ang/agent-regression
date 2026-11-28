"""Contracts for fixed multi-turn tasks and agent adapters.

Expected answers live in task checks, never in the messages sent to an agent.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_harness.models import ToolCallRecord

Status = Literal["passed", "failed", "review", "error", "invalid"]


class CheckSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["answer_contains", "answer_json", "tool_call", "tool_not_called", "state"]
    expected: Any = None
    path: str = ""
    tool: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_check(self) -> CheckSpec:
        if self.kind in {"answer_contains", "answer_json", "state"} and "expected" not in self.model_fields_set:
            raise ValueError("value checks require an explicit expected value (null is allowed)")
        if self.kind in {"tool_call", "tool_not_called"} and not self.tool:
            raise ValueError("tool checks require a tool name")
        if self.kind == "answer_contains" and not isinstance(self.expected, str):
            raise ValueError("answer_contains requires a string expected value")
        if self.kind == "answer_contains" and not self.expected.strip():
            raise ValueError("answer_contains requires a nonempty expected string")
        if self.kind == "tool_not_called" and self.arguments:
            raise ValueError("tool_not_called forbids the entire tool, not selected arguments")
        if self.kind not in {"tool_call", "tool_not_called"} and (self.tool or self.arguments):
            raise ValueError("only tool checks accept tool names and arguments")
        return self


class TaskTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    checks: list[CheckSpec] = Field(default_factory=list)


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    goal: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    turns: list[TaskTurn] = Field(min_length=1)
    max_turns: int = Field(default=10, ge=1, le=100)
    checks: list[CheckSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_turns(self) -> TaskSpec:
        if len(self.turns) > self.max_turns:
            raise ValueError("fixed turns exceed max_turns")
        return self


class TaskSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    id: str = Field(min_length=1)
    tasks: list[TaskSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> TaskSuite:
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        return self


class AgentReply(BaseModel):
    answer: str
    conversation_id: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    # False means the adapter cannot observe tools, not that none were called.
    tools_observed: bool = False
    # Observed by the adapter from a trusted test environment, not parsed from prose.
    state: dict[str, Any] | None = None


class TaskAdapter(Protocol):
    async def send(
        self, message: str, context: dict[str, Any], conversation_id: str | None,
    ) -> AgentReply: ...


class CheckResult(BaseModel):
    name: str
    status: Status
    expected: Any = None
    actual: Any = None
    reason: str = ""


class TurnResult(BaseModel):
    index: int
    message: str
    reply: AgentReply | None = None
    checks: list[CheckResult] = Field(default_factory=list)
    latency_ms: float = 0
    error: str = ""


class TaskResult(BaseModel):
    task_id: str
    status: Status = "review"
    turns: list[TurnResult] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    error: str = ""
    completed: bool = False


class TaskRun(BaseModel):
    suite_id: str
    # Execution inputs and target identity; mismatch rejects resume/rejudge.
    fingerprint: str
    # Checks may change for an offline rejudge, but not for resume.
    checks_fingerprint: str = ""
    results: list[TaskResult] = Field(default_factory=list)
