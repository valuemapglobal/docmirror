"""
变换分析器 (MutationAnalyzer)
================================

对应人类认知的"边做边学"闭环 — 分析 Mutation 历史，
识别高频错误模式，生成改进建议。

功能:
    1. 按中间件/字段/场景分组统计 Mutations
    2. 识别高频修复模式 (Top-N)
    3. 生成 hints.yaml 更新建议
    4. 输出分析报告

使用方式::

    from docmirror.middlewares.mutation_analyzer import MutationAnalyzer
    analyzer = MutationAnalyzer()
    report = analyzer.analyze(result.mutations)
    print(report.summary)
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from ...models.mutation import Mutation

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ErrorPattern:
    """高频错误模式。"""
    field: str
    old_pattern: str  # 代表性的 old_value
    new_pattern: str  # 代表性的 new_value
    count: int
    middleware: str
    avg_confidence: float

    @property
    def is_high_frequency(self) -> bool:
        return self.count >= 3


@dataclasses.dataclass
class AnalysisReport:
    """Mutation 分析报告。"""
    total_mutations: int = 0
    by_middleware: Dict[str, int] = dataclasses.field(default_factory=dict)
    by_field: Dict[str, int] = dataclasses.field(default_factory=dict)
    error_patterns: List[ErrorPattern] = dataclasses.field(default_factory=list)
    hints_suggestions: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def summary(self) -> str:
        """生成人类可读的摘要。"""
        if self.total_mutations == 0:
            return "No mutations recorded."

        lines = [
            f"📊 Mutation Analysis: {self.total_mutations} total",
            "",
            "By middleware:",
        ]
        for mw, cnt in sorted(self.by_middleware.items(), key=lambda x: -x[1]):
            lines.append(f"  • {mw}: {cnt}")

        lines.append("")
        lines.append("By field:")
        for f, cnt in sorted(self.by_field.items(), key=lambda x: -x[1]):
            lines.append(f"  • {f}: {cnt}")

        if self.error_patterns:
            lines.append("")
            lines.append("⚠️ High-frequency patterns:")
            for p in self.error_patterns[:5]:
                lines.append(
                    f"  • [{p.middleware}] {p.field}: "
                    f"'{p.old_pattern}' → '{p.new_pattern}' "
                    f"(×{p.count}, conf={p.avg_confidence:.2f})"
                )

        if self.hints_suggestions:
            lines.append("")
            lines.append("💡 Suggested hints.yaml updates:")
            for key, val in self.hints_suggestions.items():
                lines.append(f"  • {key}: {val}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_mutations": self.total_mutations,
            "by_middleware": self.by_middleware,
            "by_field": self.by_field,
            "error_patterns": [
                dataclasses.asdict(p) for p in self.error_patterns
            ],
            "hints_suggestions": self.hints_suggestions,
        }


class MutationAnalyzer:
    """
    分析 Mutation 历史，识别模式，生成建议。

    两种使用模式:
        1. 单次分析: 分析一次解析的 mutations
        2. 批量分析: 分析多次解析的 mutations (需外部聚合)
    """

    def analyze(self, mutations: List[Mutation]) -> AnalysisReport:
        """分析 Mutation 列表。"""
        report = AnalysisReport(total_mutations=len(mutations))

        if not mutations:
            return report

        # ── 1. 分组统计 ──
        by_mw: Dict[str, int] = Counter()
        by_field: Dict[str, int] = Counter()
        for m in mutations:
            by_mw[m.middleware_name] += 1
            by_field[m.field_changed] += 1

        report.by_middleware = dict(by_mw)
        report.by_field = dict(by_field)

        # ── 2. 识别高频模式 ──
        # 按 (middleware, field, old_value_type) 分组
        pattern_groups: Dict[str, List[Mutation]] = defaultdict(list)
        for m in mutations:
            # 归一化 old_value 为类型标签
            old_type = self._classify_value(m.old_value)
            key = f"{m.middleware_name}|{m.field_changed}|{old_type}"
            pattern_groups[key].append(m)

        for key, group in pattern_groups.items():
            if len(group) < 2:
                continue
            parts = key.split("|")
            avg_conf = sum(m.confidence for m in group) / len(group)
            report.error_patterns.append(ErrorPattern(
                field=parts[1],
                old_pattern=str(group[0].old_value)[:30],
                new_pattern=str(group[0].new_value)[:30],
                count=len(group),
                middleware=parts[0],
                avg_confidence=round(avg_conf, 3),
            ))

        report.error_patterns.sort(key=lambda p: -p.count)

        # ── 3. 生成 hints 建议 ──
        report.hints_suggestions = self._generate_suggestions(report.error_patterns)

        logger.info(
            f"[MutationAnalyzer] {report.total_mutations} mutations | "
            f"{len(report.error_patterns)} patterns | "
            f"{len(report.hints_suggestions)} suggestions"
        )

        return report

    def _classify_value(self, value: Any) -> str:
        """将值归类为类型标签 (用于模式聚合)。"""
        if value is None:
            return "null"
        s = str(value)
        if not s:
            return "empty"
        import re
        # 日期模式 (支持 2024-01-01, 2024/1/1 等)
        if re.match(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', s):
            return "date"
        if re.match(r'^-?\d[\d,]*\.?\d*$', s.replace(",", "")):
            return "number"
        if len(s) <= 10:
            return f"short:{s}"
        return "text"

    def _generate_suggestions(self, patterns: List[ErrorPattern]) -> Dict[str, Any]:
        """从高频模式生成 hints.yaml 更新建议。"""
        suggestions: Dict[str, Any] = {}

        for p in patterns:
            if not p.is_high_frequency:
                continue

            # 高频日期修复 → 建议添加 date_format 规则
            if p.field == "date" and p.middleware == "Repairer":
                suggestions["institution_hints.date_format"] = (
                    f"Add normalize rule: '{p.old_pattern}' → '{p.new_pattern}' "
                    f"(seen {p.count}×)"
                )

            # 高频列映射 → 建议添加 column_alias
            if p.field == "column_mapping" and p.middleware == "ColumnMapper":
                suggestions["column_aliases"] = (
                    f"Add alias: '{p.old_pattern}' → '{p.new_pattern}' "
                    f"(seen {p.count}×)"
                )

            # 高频金额修复 → 建议添加 amount 规则
            if "amount" in p.field.lower() and p.middleware == "Repairer":
                suggestions["institution_hints.amount_sign_rule"] = (
                    f"Check sign convention: '{p.old_pattern}' → '{p.new_pattern}' "
                    f"(seen {p.count}×)"
                )

        return suggestions
