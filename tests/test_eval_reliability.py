"""Regression tests for false passes and unavailable evidence, with no model calls."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from agent_harness.config import HarnessConfig
from agent_harness.core.gate import GateRule, evaluate_gate
from agent_harness.eval.pipeline import EvalPipeline
from agent_harness.models import (
    CaseResult,
    DimensionScore,
    EvalScore,
    ExpectationSpec,
    RepeatResult,
    RunResult,
    ScorerType,
    SSEStreamResult,
    ToolCallRecord,
)
from agent_harness.models import TestCase as Case


def pipeline(tmp_path: Path) -> EvalPipeline:
    config = HarnessConfig(output_dir=tmp_path)
    config.eval.judge.enabled = False
    return EvalPipeline(config)


def keyword_case(case_id: str = "keyword") -> Case:
    return Case(id=case_id, question="q", expectations=ExpectationSpec(
        expected_keywords=["required"],
    ))


@pytest.mark.asyncio
async def test_only_active_failed_check_scores_zero(tmp_path):
    async with httpx.AsyncClient() as client:
        score = await pipeline(tmp_path)._score_stream(
            keyword_case(), SSEStreamResult(answer_parts=["wrong"]), client,
        )
    assert score.overall_score == 0
    assert score.status == "failed"
    assert not score.overall_passed


@pytest.mark.asyncio
async def test_no_executable_expectation_needs_review(tmp_path):
    async with httpx.AsyncClient() as client:
        score = await pipeline(tmp_path)._score_stream(
            Case(id="no-checks", question="q"),
            SSEStreamResult(answer_parts=["plausible response"]), client,
        )
    assert score.status == "review"
    assert score.overall_score == 0
    assert not score.overall_passed


@pytest.mark.asyncio
@pytest.mark.parametrize("flags", [
    {"stream_interrupted": True}, {"error_events": ["tool service failed"]},
])
async def test_partial_correct_text_does_not_hide_transport_error(tmp_path, flags):
    async with httpx.AsyncClient() as client:
        score = await pipeline(tmp_path)._score_stream(
            keyword_case(), SSEStreamResult(answer_parts=["required"], **flags), client,
        )
    assert score.status == "error"
    assert not score.overall_passed


@pytest.mark.asyncio
async def test_failed_request_cannot_pass_using_residual_answer(tmp_path):
    result = RunResult(run_id="r", mode="eval", case_results=[CaseResult(
        case=keyword_case(), success=False, error="timeout",
        stream=SSEStreamResult(answer_parts=["required"]),
    )])
    await pipeline(tmp_path).evaluate(result)
    score = result.case_results[0].eval_score
    assert score.status == "error"
    assert result.summary["eval_error"] == 1
    assert result.summary["eval_decided"] == 0


@pytest.mark.asyncio
async def test_extra_tool_fails_even_when_expected_tool_is_present(tmp_path):
    case = Case(id="tools", question="q", expectations=ExpectationSpec(
        expected_tools=["lookup"],
    ))
    stream = SSEStreamResult(answer_parts=["done"], tool_calls=[
        ToolCallRecord(name="lookup"), ToolCallRecord(name="delete_record"),
    ])
    async with httpx.AsyncClient() as client:
        score = await pipeline(tmp_path)._score_stream(case, stream, client)
    assert score.status == "failed"
    assert not score.overall_passed


@pytest.mark.asyncio
async def test_reference_with_matching_numbers_is_unverified(tmp_path):
    case = Case(id="numbers", question="q", expectations=ExpectationSpec(
        reference_answer="A=10, B=20", expected_keywords=["A"],
    ))
    async with httpx.AsyncClient() as client:
        score = await pipeline(tmp_path)._score_stream(
            case, SSEStreamResult(answer_parts=["A=20, B=10"]), client,
        )
    assert score.status == "review"
    assert not score.overall_passed


@pytest.mark.asyncio
async def test_available_independent_judge_can_resolve_reference(tmp_path):
    evaluator = pipeline(tmp_path)
    evaluator.config.eval.judge.enabled = True

    async def judge(*_args):
        return DimensionScore(scorer=ScorerType.LLM_JUDGE, score=5, max_score=5)

    evaluator.llm_judge.score = judge
    case = Case(id="judged", question="q", expectations=ExpectationSpec(
        reference_answer="A=10", is_open_ended=True,
    ))
    async with httpx.AsyncClient() as client:
        score = await evaluator._score_stream(
            case, SSEStreamResult(answer_parts=["A=10"]), client,
        )
    assert score.status == "passed"


@pytest.mark.asyncio
async def test_judge_failure_is_unavailable_evidence_not_business_failure(tmp_path):
    evaluator = pipeline(tmp_path)
    evaluator.config.eval.judge.enabled = True

    async def unavailable(*_args, **_kwargs):
        return None

    evaluator.llm_judge._call_judge_cached = unavailable
    case = Case(id="judge-error", question="q", expectations=ExpectationSpec(
        is_open_ended=True,
    ))
    async with httpx.AsyncClient() as client:
        score = await evaluator._score_stream(
            case, SSEStreamResult(answer_parts=["answer"]), client,
        )
    assert score.status == "review"
    assert not score.overall_passed


@pytest.mark.asyncio
@pytest.mark.parametrize("second_success,second_answer,expected_status", [
    (False, "required", "error"),
    (True, "wrong", "failed"),
])
async def test_later_repeat_cannot_hide_behind_first_pass(
    tmp_path, second_success, second_answer, expected_status,
):
    first = SSEStreamResult(answer_parts=["required"])
    second = SSEStreamResult(answer_parts=[second_answer])
    case_result = CaseResult(
        case=keyword_case(), stream=first, success=True, repeat_count=2,
        repeats=[
            RepeatResult(attempt=1, success=True, stream=first),
            RepeatResult(attempt=2, success=second_success, stream=second),
        ],
    )
    result = RunResult(run_id="repeats", mode="eval", case_results=[case_result])
    await pipeline(tmp_path).evaluate(result)
    assert case_result.repeats[0].status == "passed"
    assert case_result.repeats[1].status == expected_status
    assert not case_result.repeats[1].overall_passed
    assert case_result.eval_score.status == expected_status
    assert not case_result.eval_score.overall_passed


def test_summary_excludes_unknown_outcomes_and_skipped_dimensions():
    outcomes = [
        EvalScore(case_id="p", status="passed", overall_score=1,
                  dimensions=[DimensionScore(scorer=ScorerType.FACT, score=1)]),
        EvalScore(case_id="f", status="failed", overall_score=0),
        EvalScore(case_id="r", status="review", overall_score=1,
                  dimensions=[DimensionScore(scorer=ScorerType.TOOL, score=1, skipped=True)]),
        EvalScore(case_id="e", status="error", overall_score=0),
    ]
    result = RunResult(run_id="summary", mode="eval", case_results=[
        CaseResult(case=keyword_case(s.case_id), eval_score=s) for s in outcomes
    ])
    summary = EvalPipeline._build_eval_summary(result)
    assert summary["eval_decided"] == 2
    assert summary["eval_pass_rate"] == 0.5
    assert summary["eval_avg_score"] == 0.5
    assert summary["eval_passed"] == summary["eval_failed"] == 1
    assert summary["eval_review"] == summary["eval_error"] == 1
    assert summary["eval_unresolved"] == 2
    assert "tool" not in summary["dimension_stats"]


@pytest.mark.parametrize("decided,unresolved", [(0, 0), (1, 1)])
def test_gate_refuses_incomplete_new_summary(decided, unresolved):
    gate = evaluate_gate({
        "eval_avg_score": 1, "eval_pass_rate": 1,
        "eval_decided": decided, "eval_unresolved": unresolved,
    }, None, GateRule(min_score=0.9))
    assert not gate.passed
    assert not next(check for check in gate.checks if check.name == "quality_complete").passed


def test_old_serialized_scores_keep_their_boolean_verdict():
    assert EvalScore(case_id="old", overall_passed=False).status == "failed"
    assert EvalScore(case_id="old", overall_passed=True).status == "passed"


def test_single_hard_regression_is_not_suppressed_by_insufficient_sample_size(tmp_path):
    from agent_harness.core.baseline import BaselineStore
    def run(status):
        return RunResult(run_id=status, mode="eval", case_results=[CaseResult(
            case=keyword_case(), eval_score=EvalScore(case_id="keyword", status=status, overall_score=0.9),
        )])
    store = BaselineStore(tmp_path)
    store.save("good", run("passed"))
    comparison = store.compare(run("failed"), "good")
    assert comparison["summary"]["hard_regression_count"] == 1
    assert not evaluate_gate({}, comparison, GateRule(max_regressions=0)).passed


def test_changed_scoring_policy_cannot_be_compared_to_legacy_scores(tmp_path):
    from agent_harness.core.baseline import BaselineStore
    store = BaselineStore(tmp_path)
    legacy = RunResult(run_id="old", mode="eval", case_results=[CaseResult(
        case=keyword_case(), eval_score=EvalScore(case_id="keyword", status="passed", overall_score=0.2),
    )])
    store.save("legacy", legacy)
    current = legacy.model_copy(update={"summary": {"scoring_version": "evidence-v2"}})
    assert "scoring policy changed" in store.compare(current, "legacy")["error"]
