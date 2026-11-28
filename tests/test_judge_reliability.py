"""Judge request caching and score normalization, without external model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from agent_harness.config import HarnessConfig
from agent_harness.eval.llm_judge import JUDGE_PROMPT_VERSION_COT
from agent_harness.eval.pipeline import EvalPipeline
from agent_harness.models import ExpectationSpec, SSEStreamResult
from agent_harness.models import TestCase as Case


def make_evaluator(tmp_path: Path, *, cache_enabled: bool = True) -> EvalPipeline:
    config = HarnessConfig(output_dir=tmp_path)
    config.eval.scorers_enabled = ["llm_judge"]
    config.eval.judge.enabled = True
    config.eval.judge.cache_enabled = cache_enabled
    return EvalPipeline(config)


def make_case() -> Case:
    return Case(id="same-case", question="Which values belong to A and B?", expectations=ExpectationSpec(
        reference_answer="A=10, B=20", is_open_ended=True,
    ))


@pytest.mark.asyncio
async def test_perfect_judge_score_normalizes_once(tmp_path):
    evaluator = make_evaluator(tmp_path)

    async def judge(*_args, **_kwargs):
        return {"weighted_total": 5}

    evaluator.llm_judge._call_judge = judge
    async with httpx.AsyncClient() as client:
        result = await evaluator._score_stream(
            make_case(), SSEStreamResult(answer_parts=["A=10, B=20"]), client,
        )
    assert result.status == "passed"
    assert result.overall_score == 1
    assert result.dimensions[0].score == result.dimensions[0].max_score == 5


@pytest.mark.asyncio
async def test_judge_weight_combines_with_failed_required_check(tmp_path):
    evaluator = make_evaluator(tmp_path)
    evaluator.config.eval.scorers_enabled = ["fact", "llm_judge"]
    evaluator.enabled = {"fact", "llm_judge"}
    evaluator.config.eval.dimension_weights = {"fact": 0.75, "llm_judge": 0.25}

    async def judge(*_args, **_kwargs):
        return {"weighted_total": 5}

    evaluator.llm_judge._call_judge = judge
    case = make_case()
    case.expectations.expected_keywords = ["required-keyword"]
    async with httpx.AsyncClient() as client:
        result = await evaluator._score_stream(
            case, SSEStreamResult(answer_parts=["A=10, B=20"]), client,
        )
    assert result.overall_score == 0.25
    assert result.status == "failed"


@pytest.mark.asyncio
async def test_identical_complete_request_hits_cache(tmp_path):
    evaluator = make_evaluator(tmp_path)
    calls = []

    async def judge(system, user, _client=None):
        calls.append((system, user))
        return {"weighted_total": 5}

    evaluator.llm_judge._call_judge = judge
    case = make_case()
    stream = SSEStreamResult(answer_parts=["A=10, B=20"])
    async with httpx.AsyncClient() as client:
        first = await evaluator._score_stream(case, stream, client)
        second = await evaluator._score_stream(case, stream, client)
    assert first.status == second.status == "passed"
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [
    "reference", "question", "context", "system_prompt", "model", "api_base",
    "chat_path", "temperature", "max_tokens", "cot_enabled",
])
async def test_changed_evaluation_request_does_not_reuse_old_judgment(tmp_path, changed):
    evaluator = make_evaluator(tmp_path)
    calls = []

    async def judge(system, user, _client=None):
        calls.append((system, user))
        return {"weighted_total": 5 if len(calls) == 1 else 0}

    evaluator.llm_judge._call_judge = judge
    case = make_case()
    stream = SSEStreamResult(answer_parts=["A=10, B=20"])
    async with httpx.AsyncClient() as client:
        first = await evaluator._score_stream(case, stream, client)
        if changed == "reference":
            case.expectations.reference_answer = "A=20, B=10"
        elif changed == "question":
            case.question = "Explain a different task."
        elif changed == "context":
            case.context = {"project": "different"}
        else:
            values = {
                "system_prompt": "Use a different rubric.",
                "model": "different-model", "api_base": "https://different.example/v1",
                "chat_path": "/different/completions", "temperature": 0.5,
                "max_tokens": 4096, "cot_enabled": False,
            }
            setattr(evaluator.config.eval.judge, changed, values[changed])
        second = await evaluator._score_stream(case, stream, client)
    assert first.status == "passed"
    assert second.status == "failed"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_legacy_case_answer_cache_is_never_reused(tmp_path):
    evaluator = make_evaluator(tmp_path)
    case = make_case()
    stream = SSEStreamResult(answer_parts=["A=10, B=20"])
    old_key = hashlib.sha256(
        f"{case.id}:{stream.full_answer}:{JUDGE_PROMPT_VERSION_COT}".encode()
    ).hexdigest()[:16]
    (tmp_path / "judge_cache" / f"{old_key}.json").write_text(
        json.dumps({"weighted_total": 5}), encoding="utf-8",
    )
    calls = []

    async def judge(*_args, **_kwargs):
        calls.append(True)
        return {"weighted_total": 0}

    evaluator.llm_judge._call_judge = judge
    async with httpx.AsyncClient() as client:
        result = await evaluator._score_stream(case, stream, client)
    assert len(calls) == 1
    assert result.status == "failed"


@pytest.mark.asyncio
async def test_failed_judge_call_is_not_cached_as_a_verdict(tmp_path):
    evaluator = make_evaluator(tmp_path)
    calls = []

    async def judge(*_args, **_kwargs):
        calls.append(True)
        return None if len(calls) == 1 else {"weighted_total": 5}

    evaluator.llm_judge._call_judge = judge
    case = make_case()
    stream = SSEStreamResult(answer_parts=["answer"])
    async with httpx.AsyncClient() as client:
        first = await evaluator._score_stream(case, stream, client)
        assert first.status == "review"
        assert not list((tmp_path / "judge_cache").glob("*.json"))
        second = await evaluator._score_stream(case, stream, client)
    assert second.status == "passed"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_disabled_cache_executes_each_request(tmp_path):
    evaluator = make_evaluator(tmp_path, cache_enabled=False)
    calls = []

    async def judge(*_args, **_kwargs):
        calls.append(True)
        return {"weighted_total": 5}

    evaluator.llm_judge._call_judge = judge
    async with httpx.AsyncClient() as client:
        for _ in range(2):
            await evaluator._score_stream(
                make_case(), SSEStreamResult(answer_parts=["answer"]), client,
            )
    assert len(calls) == 2
    assert evaluator.llm_judge.cache is None


@pytest.mark.asyncio
async def test_disabled_judge_does_not_use_or_populate_cache(tmp_path):
    evaluator = make_evaluator(tmp_path)
    evaluator.config.eval.judge.enabled = False

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled judge should not call a model")

    evaluator.llm_judge._call_judge = forbidden
    async with httpx.AsyncClient() as client:
        result = await evaluator._score_stream(
            make_case(), SSEStreamResult(answer_parts=["answer"]), client,
        )
    assert result.status == "review"
    assert not list((tmp_path / "judge_cache").glob("*.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("total", [-1, 6, float("nan"), float("inf"), True, "invalid"])
async def test_invalid_judge_total_cannot_certify_an_answer(tmp_path, total):
    evaluator = make_evaluator(tmp_path, cache_enabled=False)
    async def judge(*_args, **_kwargs):
        return {"weighted_total": total}
    evaluator.llm_judge._call_judge = judge
    async with httpx.AsyncClient() as client:
        score = await evaluator._score_stream(make_case(), SSEStreamResult(answer_parts=["answer"]), client)
    assert score.status == "review"
    assert not score.overall_passed


def test_explicit_zero_is_not_replaced_by_dimension_scores(tmp_path):
    evaluator = make_evaluator(tmp_path)
    result = evaluator.llm_judge._parse_judge_result({
        "weighted_total": 0, "dimensions": {"accuracy": {"score": 5}},
    })
    assert result.score == 0
    assert not result.passed


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [{}, {"weighted_total": 6}])
async def test_missing_or_invalid_judgment_is_not_cached(tmp_path, invalid):
    evaluator = make_evaluator(tmp_path)
    calls = []
    async def judge(*_args, **_kwargs):
        calls.append(True)
        return invalid if len(calls) == 1 else {"weighted_total": 5}
    evaluator.llm_judge._call_judge = judge
    async with httpx.AsyncClient() as client:
        first = await evaluator._score_stream(make_case(), SSEStreamResult(answer_parts=["answer"]), client)
        assert first.status == "review"
        assert not list((tmp_path / "judge_cache").glob("*.json"))
        second = await evaluator._score_stream(make_case(), SSEStreamResult(answer_parts=["answer"]), client)
    assert second.status == "passed"
    assert len(calls) == 2
