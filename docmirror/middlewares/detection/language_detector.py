"""
LanguageDetector \u2014 Cross-Format Language Detection Middleware
============================================================

Utilizes a CJK character ratio heuristic to identify the primary language 
of the document structurally. Applicable ubiquitously across all file formats.
"""
from __future__ import annotations


import logging

from ..base import BaseMiddleware
from ...models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)


class LanguageDetector(BaseMiddleware):
    """Detects primary language dominance across document texts."""

    def process(self, result: EnhancedResult) -> EnhancedResult:
        if result.base_result is None:
            return result

        # Limit heuristic sample boundary sizing maximizing operational performance
        text = result.base_result.full_text[:3000]
        if not text.strip():
            result.enhanced_data["language"] = "unknown"
            return result

        lang = self._detect(text)
        result.enhanced_data["language"] = lang
        result.record_mutation(
            self.name, "doc", "language", "", lang,
            reason=f"Auto-detected from {len(text)} chars",
        )
        return result

    @staticmethod
    def _detect(text: str) -> str:
        """Heuristic language detection using CJK code block ratios."""
        total = max(len(text), 1)
        cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        ratio = cjk / total

        if ratio > 0.3:
            return "zh"
        elif ratio > 0.05:
            return "mixed"
        return "en"
