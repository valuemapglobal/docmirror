# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Mirror Fidelity Validator
==========================

Measures **parsing fidelity** — how faithfully DocMirror reproduced the
original document. Uses a penalty-based approach:

    mirror_fidelity = 1.0 - Σ penalties

Only **parsing artifacts** reduce the score. Document characteristics
(empty cells, short text, content distribution) do NOT.

7 Detectable Parsing Artifacts:
    1. Column Misalignment — Table rows with different column counts
    2. Encoding Errors — Mojibake, replacement characters
    3. Page Loss — Physical pages that produced zero content blocks
    4. Duplicate Rows — Consecutive identical table rows
    5. Header Repetition — Header row appearing as data row
    6. Whitespace Anomaly — Excessive inter-character spacing
    7. Text Truncation — Text ending mid-word without punctuation
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set

from ...models.entities.parse_result import ParseResult, TableBlock, TrustResult
from ..base import BaseMiddleware

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Detection Patterns
# ══════════════════════════════════════════════════════════════════

_RE_GARBLED = re.compile(
    r"[\ufffd\ufffe\uffff]|"
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]"
)

_RE_CHAR_SPACED = re.compile(r"(?:\S\s){3,}\S")

_RE_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

_RE_SENTENCE_END = re.compile(r"[.!?。！？；;:：\-—)\]》）】」』\d%‰]$")

_PENALTY_CAPS = {
    "column_misalignment": 0.25,
    "encoding_errors": 0.20,
    "page_loss": 0.15,
    "duplicate_rows": 0.15,
    "header_repetition": 0.10,
    "whitespace_anomaly": 0.10,
    "text_truncation": 0.05,
}


class Validator(BaseMiddleware):
    """
    Mirror Fidelity Assessment Middleware.

    Operates directly on ParseResult — reads tables and text from pages,
    writes result to ParseResult.trust.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        pass_threshold: float = 0.7,
    ):
        super().__init__(config)
        self.pass_threshold = pass_threshold

    def process(self, result: ParseResult) -> ParseResult:
        """Assess mirror fidelity of the parsed document."""
        # ─── Collect data directly from ParseResult ───
        tables: list[TableBlock] = result.all_tables()
        all_text_content: list[str] = []
        block_pages: set[int] = set()
        total_page_count = result.page_count

        for page in result.pages:
            has_content = bool(page.tables or page.texts or page.key_values)
            if has_content:
                block_pages.add(page.page_number)

            # Collect text from text blocks
            for text in page.texts:
                if text.content:
                    all_text_content.append(text.content)

            # Collect text from table cells
            for table in page.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text:
                            all_text_content.append(cell.text)

            # Collect text from key-value pairs
            for kv in page.key_values:
                if kv.value:
                    all_text_content.append(kv.value)

        # ─── Convert TableBlock to 2D arrays for existing checks ───
        table_arrays = self._tables_to_arrays(tables)

        # ─── Detect all 7 artifact categories ───
        details: dict[str, float] = {}
        penalties: dict[str, float] = {}

        # 1. Column Misalignment
        score = self._check_column_alignment(table_arrays)
        details["column_alignment"] = score
        if score < 1.0:
            penalties["column_misalignment"] = round((1.0 - score) * _PENALTY_CAPS["column_misalignment"], 4)

        # 2. Encoding Errors
        score = self._check_encoding_fidelity(all_text_content)
        details["encoding_fidelity"] = score
        if score < 1.0:
            penalties["encoding_errors"] = round((1.0 - score) * _PENALTY_CAPS["encoding_errors"], 4)

        # 3. Page Loss
        score = self._check_page_coverage(block_pages, total_page_count)
        details["page_coverage"] = score
        if score < 1.0:
            penalties["page_loss"] = round((1.0 - score) * _PENALTY_CAPS["page_loss"], 4)

        # 4. Duplicate Rows
        score = self._check_duplicate_rows(table_arrays)
        details["row_uniqueness"] = score
        if score < 1.0:
            penalties["duplicate_rows"] = round((1.0 - score) * _PENALTY_CAPS["duplicate_rows"], 4)

        # 5. Header Repetition
        score = self._check_header_repetition(table_arrays)
        details["header_uniqueness"] = score
        if score < 1.0:
            penalties["header_repetition"] = round((1.0 - score) * _PENALTY_CAPS["header_repetition"], 4)

        # 6. Whitespace Anomaly
        score = self._check_whitespace_anomaly(all_text_content)
        details["whitespace_fidelity"] = score
        if score < 1.0:
            penalties["whitespace_anomaly"] = round((1.0 - score) * _PENALTY_CAPS["whitespace_anomaly"], 4)

        # 7. Text Truncation
        score = self._check_text_truncation(all_text_content)
        details["text_completeness"] = score
        if score < 1.0:
            penalties["text_truncation"] = round((1.0 - score) * _PENALTY_CAPS["text_truncation"], 4)

        # ─── Composite mirror fidelity ───
        total_penalty = min(1.0, sum(penalties.values()))
        total_score = round(1.0 - total_penalty, 4)
        passed = total_score >= self.pass_threshold

        # ─── Write directly to ParseResult.trust ───
        result.trust = TrustResult(
            validation_score=total_score,
            validation_passed=passed,
            trust_score=total_score,
            details={
                **details,
                "penalties": penalties,
                "threshold": self.pass_threshold,
                "row_count": sum(max(0, len(t) - 1) for t in table_arrays) if table_arrays else 0,
            },
        )

        result.record_mutation(
            middleware_name=self.name,
            target_block_id="document",
            field_changed="validation",
            old_value=None,
            new_value=f"fidelity={total_score:.3f} passed={passed}",
            confidence=1.0,
            reason=f"mirror fidelity: penalties={penalties}",
        )

        logger.info(f"[Validator] fidelity={total_score:.3f} | passed={passed} | penalties={penalties}")

        return result

    # ════════════════════════════════════════════════════════════════
    # Helper: Convert typed TableBlock to 2D arrays for legacy checks
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _tables_to_arrays(tables: list[TableBlock]) -> list[list[list[str]]]:
        """Convert typed TableBlocks to 2D string arrays for checks."""
        arrays = []
        for table in tables:
            if not table.headers and not table.rows:
                continue
            array: list[list[str]] = []
            if table.headers:
                array.append(table.headers)
            for row in table.rows:
                array.append([c.text for c in row.cells])
            arrays.append(array)
        return arrays

    # ════════════════════════════════════════════════════════════════
    # 7 Artifact Detection Dimensions
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _check_column_alignment(tables: list[list[list[str]]]) -> float:
        total = 0
        aligned = 0
        for table in tables:
            if len(table) < 2:
                continue
            expected = len(table[0])
            for row in table[1:]:
                total += 1
                if len(row) == expected:
                    aligned += 1
        return aligned / total if total > 0 else 1.0

    @staticmethod
    def _check_encoding_fidelity(texts: list[str]) -> float:
        total_chars = 0
        garbled_chars = 0
        for text in texts:
            if not text:
                continue
            total_chars += len(text)
            garbled_chars += len(_RE_GARBLED.findall(text))
            for ch in text:
                if unicodedata.category(ch) == "Cn":
                    garbled_chars += 1
        if total_chars == 0:
            return 1.0
        garbled_ratio = garbled_chars / total_chars
        return max(0.0, 1.0 - garbled_ratio * 20)

    @staticmethod
    def _check_page_coverage(block_pages: set[int], total: int) -> float:
        if total <= 1:
            return 1.0
        if not block_pages:
            return 0.0
        return len(block_pages) / total

    @staticmethod
    def _check_duplicate_rows(tables: list[list[list[str]]]) -> float:
        total = 0
        duplicates = 0
        for table in tables:
            rows = table[1:] if len(table) > 1 else []
            for i, row in enumerate(rows):
                total += 1
                if i > 0 and row == rows[i - 1]:
                    duplicates += 1
        return (total - duplicates) / total if total > 0 else 1.0

    @staticmethod
    def _check_header_repetition(tables: list[list[list[str]]]) -> float:
        total = 0
        repeated = 0
        for table in tables:
            if len(table) < 2:
                continue
            header = table[0]
            header_norm = [c.strip().lower() for c in header]
            for row in table[1:]:
                total += 1
                row_norm = [c.strip().lower() for c in row]
                if row_norm == header_norm:
                    repeated += 1
        return (total - repeated) / total if total > 0 else 1.0

    @staticmethod
    def _check_whitespace_anomaly(texts: list[str]) -> float:
        total_segments = 0
        anomalous_segments = 0
        for text in texts:
            if not text or len(text) < 5:
                continue
            total_segments += 1
            if _RE_CHAR_SPACED.search(text):
                anomalous_segments += 1
        return (total_segments - anomalous_segments) / total_segments if total_segments > 0 else 1.0

    @staticmethod
    def _check_text_truncation(texts: list[str]) -> float:
        checked = 0
        truncated = 0
        for text in texts:
            if not text:
                continue
            stripped = text.strip()
            if len(stripped) < 10 or " " not in stripped:
                continue
            checked += 1
            if not _RE_SENTENCE_END.search(stripped):
                if stripped and _RE_CJK.match(stripped[-1]):
                    continue
                if stripped and stripped[-1].isdigit():
                    continue
                truncated += 1
        return (checked - truncated) / checked if checked > 0 else 1.0
