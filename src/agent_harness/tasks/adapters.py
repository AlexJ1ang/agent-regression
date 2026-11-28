"""Explicit JSON/HTTP integration for an already deployed business agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_harness.models import ToolCallRecord
from agent_harness.tasks.models import AgentReply


class TargetConfig(BaseModel):
    """Only transport and observation settings; no expected answers or secrets."""

    model_config = ConfigDict(extra="forbid")

    url: str
    timeout_seconds: float = Field(default=30, gt=0)
    headers_env: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    query_field: str = "query"
    context_field: str = "inputs"
    conversation_id_field: str = "conversation_id"
    answer_path: str = "answer"
    conversation_id_path: str = "conversation_id"
    tools_path: str = "tool_calls"
    state_url: str | None = None
    state_path: str = ""
    revision: str = ""

    @field_validator("url", "state_url")
    @classmethod
    def http_url(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("target URLs must use http or https")
            if parsed.username or parsed.password:
                raise ValueError("put authentication in headers_env, not the URL")
        return value

    @model_validator(mode="after")
    def distinct_fields(self) -> TargetConfig:
        names = [self.query_field, self.context_field, self.conversation_id_field]
        if not all(names) or len(set(names)) != len(names):
            raise ValueError("request field names must be nonempty and distinct")
        if set(names) & self.extra_body.keys():
            raise ValueError("extra_body cannot override message, context or conversation fields")
        if self.state_url:
            rest = self.state_url.replace("{conversation_id}", "")
            if "{" in rest or "}" in rest:
                raise ValueError("state_url only supports {conversation_id}")
        return self

    def identity(self) -> str:
        """Configuration identity for result reuse; never resolve credential values."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True)


def load_target(path: Path | str) -> TargetConfig:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("target configuration must be an object")
    return TargetConfig.model_validate(data)


def _read_path(data: Any, path: str) -> Any:
    for key in path.split(".") if path else []:
        if isinstance(data, list):
            data = data[int(key)]
        elif isinstance(data, dict):
            data = data[key]
        else:
            raise TypeError(f"cannot read path {path!r}")
    return data


class HttpJsonAdapter:
    """Send turns and normalize explicit observations, without guessing tool use.

    ``state_url`` must point to a trusted read-only test-environment observer.
    A response's own ``state`` field is deliberately ignored.
    """

    def __init__(
        self, config: TargetConfig, client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._owned = client is None
        self.client = client or httpx.AsyncClient()

    async def __aenter__(self) -> HttpJsonAdapter:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owned:
            await self.client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        for name, variable in self.config.headers_env.items():
            value = os.environ.get(variable)
            if value is None:
                raise ValueError(f"missing credential environment variable: {variable}")
            headers[name] = value
        return headers

    async def send(
        self, message: str, context: dict[str, Any], conversation_id: str | None,
    ) -> AgentReply:
        config = self.config
        payload: dict[str, Any] = {
            **config.extra_body,
            config.query_field: message,
            config.context_field: context,
        }
        if conversation_id is not None:
            payload[config.conversation_id_field] = conversation_id
        headers = self._headers()
        response = await self.client.post(
            config.url, json=payload, headers=headers, timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("agent response must be a JSON object")
        try:
            answer = _read_path(data, config.answer_path)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("agent response is missing the configured answer") from exc
        if not isinstance(answer, str):
            raise ValueError("agent answer must be a string")
        try:
            actual_id = _read_path(data, config.conversation_id_path)
        except (KeyError, IndexError):
            actual_id = None
        if actual_id is not None and (not isinstance(actual_id, str) or not actual_id):
            raise ValueError("conversation ID must be a nonempty string or null")

        try:
            raw_tools = _read_path(data, config.tools_path)
        except (KeyError, IndexError):
            tools_observed, raw_tools = False, []
        else:
            tools_observed = True
        if not isinstance(raw_tools, list):
            raise ValueError("tool observations must be a list")
        tools: list[ToolCallRecord] = []
        for item in raw_tools:
            if not isinstance(item, dict):
                raise ValueError("each tool observation must be an object")
            name, arguments = item.get("name"), item.get("arguments")
            if not isinstance(name, str) or not name or not isinstance(arguments, dict):
                raise ValueError("tool observations require a name and argument object")
            call_id = item.get("call_id", "")
            if not isinstance(call_id, str):
                raise ValueError("tool call_id must be a string")
            tools.append(ToolCallRecord(name=name, arguments=arguments, call_id=call_id))

        state = None
        if config.state_url and actual_id is not None:
            state_url = config.state_url.replace("{conversation_id}", quote(actual_id, safe=""))
            observation = await self.client.get(
                state_url, headers=headers, timeout=config.timeout_seconds,
            )
            observation.raise_for_status()
            state = _read_path(observation.json(), config.state_path)
            if not isinstance(state, dict):
                raise ValueError("test-environment state must be an object")
        return AgentReply(
            answer=answer, conversation_id=actual_id, tool_calls=tools,
            tools_observed=tools_observed, state=state,
        )
