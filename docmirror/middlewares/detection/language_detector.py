# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
LanguageDetector — Cross-Format Language Detection Middleware
============================================================

Utilizes a CJK character ratio heuristic to identify primary language.
Writes to ParseResult.entities.domain_specific["language"].
"""

from __future__ import annotations

import logging

from ...models.entities.parse_result import ParseResult
from ..base import BaseMiddleware

logger = logging.getLogger(__name__)


class LanguageDetector(BaseMiddleware):
    """Detects primary language dominance across document texts."""

    def process(self, result: ParseResult) -> ParseResult:
        text = result.full_text[:3000]
        if not text.strip():
            result.entities.domain_specific["language"] = "unknown"
            return result

        lang = self._detect(text)
        result.entities.domain_specific["language"] = lang
        result.record_mutation(
            self.name,
            "doc",
            "language",
            "",
            lang,
            reason=f"Auto-detected from {len(text)} chars",
        )
        return result

    @staticmethod
    def _detect(text: str) -> str:
        """Heuristic language detection using CJK code block ratios."""
        total = max(len(text), 1)
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        ratio = cjk / total

        if ratio > 0.3:
            return "zh"
        elif ratio > 0.05:
            return "mixed"
        return "en"
