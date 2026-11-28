"""事实一致性评分器（通用版）。

校验 Agent 的回答是否满足用例 expectations 中声明的确定性期望：
关键词存在和正则匹配。参考答案需由独立的业务检查或 Judge 核验，
数字恰好出现不能证明实体、单位和结论正确。
不含任何领域特定的硬编码规则。
"""

from __future__ import annotations

import re

from agent_harness.models import DimensionScore, ScorerType, SSEStreamResult, TestCase


class FactScorer:
    """基于 expectations 的通用事实校验。"""

    def score(self, case: TestCase, stream: SSEStreamResult) -> DimensionScore:
        answer = stream.full_answer
        if not answer:
            return DimensionScore(
                scorer=ScorerType.FACT,
                score=0.0,
                passed=False,
                details="回答为空",
                issues=["回答为空"],
            )

        exp = case.expectations
        issues: list[str] = []
        checks_passed = 0
        checks_total = 0

        # 1. 关键词存在性校验
        if exp.expected_keywords:
            checks_total += 1
            missing_kw = [kw for kw in exp.expected_keywords if kw not in answer]
            if not missing_kw:
                checks_passed += 1
            else:
                issues.append(f"回答缺少关键词: {missing_kw[:5]}")

        # 2. 正则模式匹配校验
        if exp.expected_patterns:
            checks_total += 1
            failed_patterns: list[str] = []
            for pattern in exp.expected_patterns:
                try:
                    if not re.search(pattern, answer):
                        failed_patterns.append(pattern)
                except re.error:
                    failed_patterns.append(f"[invalid regex: {pattern}]")
            if not failed_patterns:
                checks_passed += 1
            else:
                issues.append(f"回答未匹配正则: {failed_patterns[:3]}")

        needs_review = bool(exp.reference_answer)
        review_detail = (
            "；参考答案尚需业务检查或独立 Judge 核验，未使用数字集合覆盖判定正确"
            if needs_review else ""
        )

        if checks_total == 0:
            return DimensionScore(
                scorer=ScorerType.FACT,
                score=0.0,
                passed=False,
                skipped=True,
                needs_review=needs_review,
                details="未声明可执行的关键词或正则检查，跳过" + review_detail,
            )

        score = round(checks_passed / checks_total, 4)
        return DimensionScore(
            scorer=ScorerType.FACT,
            score=score,
            passed=not issues,
            needs_review=needs_review,
            details=f"通过 {checks_passed}/{checks_total} 项确定性检查" + review_detail,
            issues=issues,
        )
