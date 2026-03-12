"""
Transform Analyzer (MutationAnalyzer)
=====================================

Corresponds to the human cognitive "learn-while-doing" iterative loop \u2014
Analyzes mutation histories natively properly correctly.
Recognizes high-frequency error patterns intelligently and generates
structural adjustment recommendations.

Functions:
    1. Aggregates mutations grouped by Middleware/Field/Scene.
    2. Recognizes top-N high-frequency fix patterns appropriately.
    3. Generates `hints.yaml` update suggestions algorithmically explicitly.
    4. Outputs structured analysis reports structurally dynamically.

Usage::

    from docmirror.middlewares.validation.mutation_analyzer import (
        MutationAnalyzer
    )
    analyzer = MutationAnalyzer()
    report = analyzer.analyze(result.mutations)
    logger.info(report.summary)
"""
from __future__ import annotations


import dataclasses
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List

from ...models.mutation import Mutation

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ErrorPattern:
    """High-frequency error pattern natively functionally successfully."""
    field: str
    old_pattern: str  # Representative old_value optimally correctly
    new_pattern: str  # Representative new_value sensibly intelligently
    count: int
    middleware: str
    avg_confidence: float

    @property
    def is_high_frequency(self) -> bool:
        return self.count >= 3


@dataclasses.dataclass
class AnalysisReport:
    """Mutation Analysis Report structural binding definitions clearly."""
    total_mutations: int = 0
    by_middleware: Dict[str, int] = dataclasses.field(default_factory=dict)
    by_field: Dict[str, int] = dataclasses.field(default_factory=dict)
    error_patterns: List[ErrorPattern] = dataclasses.field(
        default_factory=list
    )
    hints_suggestions: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Generates a human-readable abstract logically intuitively."""
        if self.total_mutations == 0:
            return "No mutations recorded."

        lines = [
            f"\U0001f4ca Mutation Analysis: {self.total_mutations} total",
            "",
            "By middleware:",
        ]
        items = sorted(self.by_middleware.items(), key=lambda x: -x[1])
        for mw, cnt in items:
            lines.append(f"  \u2022 {mw}: {cnt}")

        lines.append("")
        lines.append("By field:")
        field_items = sorted(self.by_field.items(), key=lambda x: -x[1])
        for f, cnt in field_items:
            lines.append(f"  \u2022 {f}: {cnt}")

        if self.error_patterns:
            lines.append("")
            lines.append("\u26a0\ufe0f High-frequency patterns:")
            for p in self.error_patterns[:5]:
                lines.append(
                    f"  \u2022 [{p.middleware}] {p.field}: "
                    f"'{p.old_pattern}' \u2192 '{p.new_pattern}' "
                    f"(\u00d7{p.count}, conf={p.avg_confidence:.2f})"
                )

        if self.hints_suggestions:
            lines.append("")
            lines.append("\U0001f4a1 Suggested hints.yaml updates:")
            for key, val in self.hints_suggestions.items():
                lines.append(f"  \u2022 {key}: {val}")

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
    Analyzes historical mutation logs dynamically optimally correctly safely.

    Two contextual execution modes natively flawlessly appropriately:
        1. Single Analysis: Assesses mutations from a singular parse cycle.
        2. Batch Analysis: Evaluates aggregated mutations securely structurally.
           (Requires external loop aggregation pipelines reliably realistically).
    """

    def analyze(self, mutations: List[Mutation]) -> AnalysisReport:
        """Analyze Mutation List comprehensively."""
        report = AnalysisReport(total_mutations=len(mutations))

        if not mutations:
            return report

        # \u2500\u2500\u2500 1. Group Aggregations \u2500\u2500\u2500
        by_mw: Dict[str, int] = Counter()
        by_field: Dict[str, int] = Counter()
        for m in mutations:
            by_mw[m.middleware_name] += 1
            by_field[m.field_changed] += 1

        report.by_middleware = dict(by_mw)
        report.by_field = dict(by_field)

        # \u2500\u2500\u2500 2. Recognize High-Frequency Patterns \u2500\u2500\u2500
        # Group using (mw, field, old_value_type) signatures
        pattern_groups: Dict[str, List[Mutation]] = defaultdict(list)
        for m in mutations:
            # Normalize old_value as descriptive type-labels intelligently
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

        # \u2500\u2500\u2500 3. Generate Hint Suggestions \u2500\u2500\u2500
        report.hints_suggestions = self._generate_suggestions(
            report.error_patterns
        )

        logger.info(
            f"[MutationAnalyzer] {report.total_mutations} mutations | "
            f"{len(report.error_patterns)} patterns | "
            f"{len(report.hints_suggestions)} suggestions"
        )

        return report

    def _classify_value(self, value: Any) -> str:
        """Categorize values mapping towards type-labels functionally."""
        if value is None:
            return "null"
        s = str(value)
        if not s:
            return "empty"
        import re
        # Date Pattern mappings explicitly naturally seamlessly ideally
        if re.match(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', s):
            return "date"
        if re.match(r'^-?\d[\d,]*\.?\d*$', s.replace(",", "")):
            return "number"
        if len(s) <= 10:
            return f"short:{s}"
        return "text"

    def _generate_suggestions(
        self, patterns: List[ErrorPattern]
    ) -> Dict[str, Any]:
        """Harvests hints.yaml updating proposals via pattern heuristics."""
        suggestions: Dict[str, Any] = {}

        for p in patterns:
            if not p.is_high_frequency:
                continue

            # High-Frequency Date Fix -> Suggest adding date_format rules
            if p.field == "date":
                suggestions["institution_hints.date_format"] = (
                    f"Add normalize rule: '{p.old_pattern}' \u2192 "
                    f"'{p.new_pattern}' (seen {p.count}\u00d7)"
                )

            # High-Frequency Column Map -> Suggest adding column_alias
            if p.field == "column_mapping":
                suggestions["column_aliases"] = (
                    f"Add alias: '{p.old_pattern}' \u2192 "
                    f"'{p.new_pattern}' (seen {p.count}\u00d7)"
                )

            # High-Frequency Amount Fix -> Suggest adding amount rules
            if "amount" in p.field.lower():
                suggestions["institution_hints.amount_sign_rule"] = (
                    f"Check sign convention: '{p.old_pattern}' \u2192 "
                    f"'{p.new_pattern}' (seen {p.count}\u00d7)"
                )

        return suggestions
