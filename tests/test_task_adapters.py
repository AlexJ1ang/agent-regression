from __future__ import annotations

import json

import httpx
import pytest

from agent_harness.tasks.adapters import HttpJsonAdapter, TargetConfig
from agent_harness.tasks.loader import load_task_suite
from agent_harness.tasks.runner import run_suite
from examples.task_demo import DemoService


@pytest.mark.asyncio
async def test_explicit_tools_and_independent_state(monkeypatch):
    monkeypatch.setenv("TEST_AGENT_TOKEN", "private-token")
    requests = []

    def respond(request):
        requests.append(request)
        assert request.headers["Authorization"] == "private-token"
        if request.method == "GET":
            return httpx.Response(200, json={"orders": []})
        body = json.loads(request.content)
        assert body == {"query": "hello", "inputs": {"role": "reader"}, "conversation_id": "old"}
        return httpx.Response(200, json={
            "answer": "created", "conversation_id": "new", "tool_calls": [],
            "state": {"orders": ["invented"]},
        })

    config = TargetConfig(
        url="https://example.test/chat", state_url="https://example.test/state/{conversation_id}",
        headers_env={"Authorization": "TEST_AGENT_TOKEN"},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with HttpJsonAdapter(config, client) as adapter:
            reply = await adapter.send("hello", {"role": "reader"}, "old")
        assert not client.is_closed
    assert reply.tools_observed
    assert reply.state == {"orders": []}
    assert str(requests[1].url) == "https://example.test/state/new"
    assert "private-token" not in config.identity()


@pytest.mark.asyncio
async def test_missing_tool_evidence_is_unknown_and_prose_state_is_ignored():
    def respond(request):
        return httpx.Response(200, json={"answer": "used lookup", "state": {"done": True}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        reply = await HttpJsonAdapter(TargetConfig(url="https://example.test"), client).send("q", {}, None)
    assert not reply.tools_observed
    assert reply.state is None


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [None, {}, [{"name": "lookup", "arguments": "{}"}]])
async def test_malformed_tool_records_are_not_silently_empty(tools):
    def respond(request):
        return httpx.Response(200, json={"answer": "ok", "tool_calls": tools})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError):
            await HttpJsonAdapter(TargetConfig(url="https://example.test"), client).send("q", {}, None)


@pytest.mark.asyncio
async def test_http_failure_is_not_business_failure():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await HttpJsonAdapter(TargetConfig(url="https://example.test"), client).send("q", {}, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault,expected", [(None, ["passed", "passed"]), ("wrong-device", ["failed", "passed"]), ("wrong-month", ["failed", "passed"]), ("no-tools", ["review", "review"])])
async def test_full_demo_tasks_detect_injected_faults(tmp_path, fault, expected):
    service = DemoService(fault)

    def respond(request):
        if request.method == "POST":
            data = service.chat(json.loads(request.content))
        else:
            data = service.state(request.url.path.removeprefix("/state/"))
        return httpx.Response(200, json=data)

    from pathlib import Path
    suite = load_task_suite(Path(__file__).parents[1] / "examples/tasks/suite.yaml")
    target = TargetConfig(url="https://example.test/chat", state_url="https://example.test/state/{conversation_id}")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await run_suite(suite, HttpJsonAdapter(target, client), tmp_path, target_identity=target.identity())
    assert [task.status for task in result.results] == expected
    assert all(task.completed and len(task.turns) == 3 for task in result.results)
