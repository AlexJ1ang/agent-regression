"""评测管线。

按配置编排启用的评分维度，生成完整的评测报告。
"""

from __future__ import annotations

import logging
from statistics import mean, stdev
from typing import Any

import httpx

from agent_harness.config import HarnessConfig
from agent_harness.core.stats import wilson_interval
from agent_harness.eval.fact_scorer import FactScorer
from agent_harness.eval.intent_scorer import IntentScorer
from agent_harness.eval.llm_judge import LLMJudge
from agent_harness.eval.tool_scorer import ToolScorer
from agent_harness.models import (
    CaseResult,
    DimensionScore,
    EvalScore,
    EvalStatus,
    RunResult,
    ScorerType,
    SSEStreamResult,
    TestCase,
)

logger = logging.getLogger(__name__)


class EvalPipeline:
    """编排多维度评分的管线。"""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.enabled = set(config.eval.scorers_enabled)
        self.intent_scorer = IntentScorer()
        self.tool_scorer = ToolScorer()
        self.fact_scorer = FactScorer()
        self.llm_judge = LLMJudge(
            config.eval.judge,
            cache_dir=config.output_dir / "judge_cache",
        )

    async def evaluate(self, run_result: RunResult) -> RunResult:
        """对运行结果中的每条用例进行评分。"""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0)
        ) as judge_client:
            for case_result in run_result.case_results:
                case_result.eval_score = await self._score_case(
                    case_result, judge_client
                )
                # 重复采样：对每次执行的流独立评分，回填分数并聚合统计
                if case_result.repeat_count > 1:
                    await self._score_repeats(case_result, judge_client)

        run_result.summary.update(self._build_eval_summary(run_result))
        return run_result

    async def _score_case(
        self,
        case_result: CaseResult,
        judge_client: httpx.AsyncClient,
    ) -> EvalScore:
        """对单条用例进行多维度评分（基于代表 stream）。"""
        if not case_result.success or case_result.error:
            return EvalScore(
                case_id=case_result.case.id,
                overall_passed=False,
                status="error",
                latency=case_result.latency,
                error=case_result.error or "请求未成功完成",
            )
        eval_score = await self._score_stream(
            case_result.case, case_result.stream, judge_client
        )
        # 代表 stream 的评分额外携带延迟与执行错误信息
        return eval_score.model_copy(
            update={
                "latency": case_result.latency,
                "error": case_result.error,
            }
        )

    async def _score_stream(
        self,
        case: TestCase,
        stream: SSEStreamResult,
        judge_client: httpx.AsyncClient,
    ) -> EvalScore:
        """对（case, stream）组合进行多维度评分。"""
        if stream.stream_interrupted or stream.error_events:
            return EvalScore(
                case_id=case.id,
                overall_passed=False,
                status="error",
                error="；".join(stream.error_events) or "stream interrupted",
            )
        dimensions: list[DimensionScore] = []

        if ScorerType.INTENT.value in self.enabled:
            dimensions.append(self.intent_scorer.score(case, stream))

        if ScorerType.TOOL.value in self.enabled:
            dimensions.append(self.tool_scorer.score(case, stream))

        if ScorerType.FACT.value in self.enabled:
            dimensions.append(self.fact_scorer.score(case, stream))

        if (
            ScorerType.LLM_JUDGE.value in self.enabled
            and self.config.eval.judge.enabled
            and case.expectations.is_open_ended
        ):
            judge_score = await self.llm_judge.score(case, stream, judge_client)
            dimensions.append(judge_score)

        overall = self._compute_overall_score(dimensions)
        active = [d for d in dimensions if not d.skipped]
        judged = any(
            d.scorer == ScorerType.LLM_JUDGE
            and not d.skipped and not d.needs_review and not d.low_confidence
            for d in dimensions
        )
        unresolved = any(
            (d.needs_review and not (d.scorer == ScorerType.FACT and judged))
            or d.low_confidence
            for d in dimensions
        )
        status: EvalStatus
        if any(not d.passed for d in active):
            status = "failed"
        elif not active or unresolved:
            status = "review"
        else:
            status = "passed"

        return EvalScore(
            case_id=case.id,
            dimensions=dimensions,
            overall_score=overall,
            overall_passed=status == "passed",
            status=status,
        )

    async def _score_repeats(
        self,
        case_result: CaseResult,
        judge_client: httpx.AsyncClient,
    ) -> None:
        """对每个 repeat 的 stream 独立评分并回填，然后聚合统计。

        评分前 repeat 只持有 stream，无分数；此方法补齐 overall_score /
        overall_passed，再计算 mean/std/pass_rate/flaky。
        """
        primary_score = case_result.eval_score
        scores: list[float] = []
        passes: list[bool] = []
        outcomes: list[EvalScore] = []

        for repeat in case_result.repeats:
            if repeat.attempt == 1:
                # 首次执行与 primary 共享 stream/分数
                eval_score = primary_score or EvalScore(
                    case_id=case_result.case.id, status="error",
                    error="首次执行缺少评分", overall_passed=False,
                )
            elif not repeat.success or repeat.error:
                eval_score = EvalScore(
                    case_id=case_result.case.id, status="error",
                    error=repeat.error or "请求未成功完成", overall_passed=False,
                )
            else:
                eval_score = await self._score_stream(
                    case_result.case, repeat.stream, judge_client
                )
            score = eval_score.overall_score
            passed = eval_score.overall_passed
            repeat.overall_score = round(score, 4)
            repeat.overall_passed = passed
            repeat.status = eval_score.status
            outcomes.append(eval_score)
            if eval_score.status in {"passed", "failed"}:
                scores.append(score)
                passes.append(passed)

        case_result.mean_score = round(mean(scores), 4) if scores else 0.0
        case_result.std_score = (
            round(stdev(scores), 4) if len(scores) >= 2 else 0.0
        )
        case_result.pass_rate = round(sum(passes) / max(len(passes), 1), 4)
        case_result.flaky = len(set(passes)) > 1
        # A later failure or unavailable trial must not disappear behind the
        # first successful response. Business pass rate excludes unknown/error
        # trials; their statuses remain explicit in the summary.
        if primary_score and outcomes:
            statuses = {outcome.status for outcome in outcomes}
            aggregate = next(
                status for status in ("failed", "error", "review", "passed")
                if status in statuses
            )
            case_result.eval_score = primary_score.model_copy(update={
                "status": aggregate,
                "overall_passed": aggregate == "passed",
                "overall_score": case_result.mean_score,
            })

    def _compute_overall_score(self, dimensions: list[DimensionScore]) -> float:
        """按 config.eval.dimension_weights 计算综合得分。"""
        if not dimensions:
            return 0.0
        weights = self.config.eval.dimension_weights
        total = 0.0
        weight_sum = 0.0
        for d in dimensions:
            if d.skipped:
                continue
            w = weights.get(d.scorer.value, 0.1)
            normalized = d.score / d.max_score if d.max_score > 0 else 0
            total += normalized * w
            weight_sum += w
        return round(total / max(weight_sum, 0.01), 4)

    @staticmethod
    def _build_eval_summary(run_result: RunResult) -> dict[str, Any]:
        """构建评测汇总统计（含 Wilson 置信区间与 flaky 检测）。"""
        scores = [
            cr.eval_score
            for cr in run_result.case_results
            if cr.eval_score is not None
        ]
        if not scores:
            return {}

        decided = [s for s in scores if s.status in {"passed", "failed"}]
        overall_scores = [s.overall_score for s in decided]
        avg_score = sum(overall_scores) / len(decided) if decided else 0.0
        passed = sum(1 for s in decided if s.status == "passed")
        failed = sum(1 for s in decided if s.status == "failed")
        review = sum(1 for s in scores if s.status == "review")
        errors = sum(1 for s in scores if s.status == "error")

        # Wilson 置信区间
        _, pass_lower, pass_upper = wilson_interval(passed, len(decided))

        # Flaky 检测
        flaky_cases = [
            {"case_id": cr.case.id, "pass_rate": cr.pass_rate}
            for cr in run_result.case_results
            if cr.flaky
        ]

        # Judge 分歧率（low_confidence 标注的 llm_judge 维度占比）
        judge_total = 0
        judge_low_conf = 0
        for s in scores:
            for d in s.dimensions:
                if d.skipped:
                    continue
                if d.scorer == ScorerType.LLM_JUDGE:
                    judge_total += 1
                    if d.low_confidence:
                        judge_low_conf += 1
        judge_disagreement_rate = (
            round(judge_low_conf / judge_total, 4) if judge_total else None
        )

        dimension_stats: dict[str, dict[str, float]] = {}
        for s in scores:
            for d in s.dimensions:
                if d.skipped or d.needs_review:
                    continue
                key = d.scorer.value
                if key not in dimension_stats:
                    dimension_stats[key] = {"total": 0, "count": 0, "avg": 0.0}
                dimension_stats[key]["total"] += d.score
                dimension_stats[key]["count"] += 1
        for key, stats in dimension_stats.items():
            stats["avg"] = round(stats["total"] / max(stats["count"], 1), 4)

        return {
            "scoring_version": "evidence-v2",
            "eval_avg_score": round(avg_score, 4),
            "eval_pass_rate": round(passed / len(decided), 4) if decided else 0.0,
            "eval_pass_rate_ci": [round(pass_lower, 4), round(pass_upper, 4)],
            "eval_decided": len(decided),
            "eval_passed": passed,
            "eval_failed": failed,
            "eval_review": review,
            "eval_error": errors,
            "eval_unresolved": review + errors,
            "repeat_status_counts": {
                status: sum(
                    1 for cr in run_result.case_results for repeat in cr.repeats
                    if repeat.status == status
                )
                for status in ("passed", "failed", "review", "error")
            },
            "flaky_count": len(flaky_cases),
            "flaky_cases": flaky_cases,
            "judge_disagreement_rate": judge_disagreement_rate,
            "judge_low_confidence_count": judge_low_conf,
            "judge_total": judge_total,
            "dimension_stats": dimension_stats,
        }
