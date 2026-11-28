"""CI 质量门禁。

在评测完成后对运行结果施加一系列硬性门槛，任一门槛不满足即判定
评测失败（CI 中表现为 exit code 非 0）。门禁结果落盘为
``gate_result.json``，供 CI 系统解析。

门禁规则：
- ``min_score``       整体平均分下限
- ``min_pass_rate``   通过率下限
- ``max_regressions`` 允许的最大回归用例数（包括明确失败，需 baseline 对比）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class GateRule(BaseModel):
    """门禁规则集合。None 表示该项不参与门禁。"""

    min_score: float | None = None
    min_pass_rate: float | None = None
    max_regressions: int | None = None


class GateCheck(BaseModel):
    """单项门禁检查结果。"""

    name: str
    passed: bool
    threshold: float | int | None
    actual: float | int | None
    detail: str = ""


class GateResult(BaseModel):
    """门禁判定结果。"""

    passed: bool
    checks: list[GateCheck] = []
    violations: list[str] = []

    @property
    def violations_count(self) -> int:
        return len(self.violations)


def evaluate_gate(
    summary: dict[str, Any],
    comparison: dict[str, Any] | None,
    rule: GateRule,
) -> GateResult:
    """对评测汇总与（可选的）基线对比执行门禁判定。

    Args:
        summary: RunResult.summary，含 avg_score / pass_rate / regressions 等
        comparison: BaselineStore.compare 的返回，含 summary.regression_count
        rule: 门禁规则
    """
    checks: list[GateCheck] = []
    violations: list[str] = []

    avg_score = summary.get("eval_avg_score", 0.0)
    pass_rate = summary.get("eval_pass_rate", 0.0)

    # New summaries distinguish a business verdict from unavailable evidence.
    # Keep older integrations readable, but never certify unresolved new runs.
    if "eval_unresolved" in summary:
        unresolved = int(summary["eval_unresolved"])
        decided = int(summary.get("eval_decided", 0))
        complete = unresolved == 0 and decided > 0
        checks.append(GateCheck(
            name="quality_complete", passed=complete, threshold=0,
            actual=unresolved,
            detail=f"已判定 {decided} 条，待核验或执行异常 {unresolved} 条",
        ))
        if not complete:
            violations.append("评测证据不完整或没有已判定用例，禁止放行")

    # ── 平均分下限 ──
    if rule.min_score is not None:
        passed = avg_score >= rule.min_score
        check = GateCheck(
            name="min_score",
            passed=passed,
            threshold=rule.min_score,
            actual=round(avg_score, 4),
            detail=f"平均分 {avg_score:.4f} / 下限 {rule.min_score}",
        )
        checks.append(check)
        if not passed:
            violations.append(f"平均分 {avg_score:.4f} 低于下限 {rule.min_score}")

    # ── 通过率下限 ──
    if rule.min_pass_rate is not None:
        passed = pass_rate >= rule.min_pass_rate
        check = GateCheck(
            name="min_pass_rate",
            passed=passed,
            threshold=rule.min_pass_rate,
            actual=round(pass_rate, 4),
            detail=f"通过率 {pass_rate:.1%} / 下限 {rule.min_pass_rate:.1%}",
        )
        checks.append(check)
        if not passed:
            violations.append(
                f"通过率 {pass_rate:.1%} 低于下限 {rule.min_pass_rate:.1%}"
            )

    # ── 回归用例数上限（需 baseline 对比）──
    if rule.max_regressions is not None:
        if comparison is None or "error" in comparison:
            # Requested regression protection cannot be verified without a baseline.
            checks.append(
                GateCheck(
                    name="max_regressions",
                    passed=False,
                    threshold=rule.max_regressions,
                    actual=None,
                    detail="无有效基线，无法确认回归数",
                )
            )
            violations.append("缺少有效基线，不能验证回归数上限")
        else:
            reg_count = comparison.get("summary", {}).get(
                "regression_count", len(comparison.get("regressions", []))
            )
            passed = reg_count <= rule.max_regressions
            check = GateCheck(
                name="max_regressions",
                passed=passed,
                threshold=rule.max_regressions,
                actual=reg_count,
                detail=f"回归 {reg_count} 例 / 上限 {rule.max_regressions}",
            )
            checks.append(check)
            if not passed:
                violations.append(
                    f"回归 {reg_count} 例超过上限 {rule.max_regressions}"
                )

    return GateResult(
        passed=len(violations) == 0,
        checks=checks,
        violations=violations,
    )


def save_gate_result(result: GateResult, output_dir: Path) -> Path:
    """将门禁结果写入 gate_result.json，返回文件路径。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gate_result.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
    return path
