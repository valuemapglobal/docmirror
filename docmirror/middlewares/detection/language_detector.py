"""
LanguageDetector — 跨格式语言检测中间件
==========================================

通过 CJK 字符比例启发式检测文档主要语言。
适用于所有文件格式。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..base import BaseMiddleware
from ...models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)


class LanguageDetector(BaseMiddleware):
    """检测文档主要语言 (zh/en/mixed)。"""

    def process(self, result: EnhancedResult) -> EnhancedResult:
        if result.base_result is None:
            return result

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
        """启发式语言检测。"""
        total = max(len(text), 1)
        cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        ratio = cjk / total

        if ratio > 0.3:
            return "zh"
        elif ratio > 0.05:
            return "mixed"
        return "en"
