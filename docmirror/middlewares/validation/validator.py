"""
Mirror Fidelity Validator
==========================

Measures **parsing fidelity** — how faithfully DocMirror reproduced
the original document. Uses a penalty-based approach:

    mirror_fidelity = 1.0 - Σ penalties

Only **parsing artifacts** reduce the score. Document characteristics
(empty cells, short text, content distribution) do NOT.

7 Detectable Parsing Artifacts:

    1. Column Misalignment — Table rows with different column counts
       than the header → cell boundary detection failure.
    2. Encoding Errors — Mojibake, replacement characters (U+FFFD),
       control characters → text extraction/decoding failure.
    3. Page Loss — Physical pages that produced zero content blocks
       → entire page silently dropped or unreadable.
    4. Duplicate Rows — Consecutive identical table rows → parser
       extracted same row twice at page boundaries.
    5. Header Repetition — Header row appearing again as a data row
       → multi-page table header de-duplication failure.
    6. Whitespace Anomaly — Excessive inter-character spacing within
       words → OCR/extraction spacing reconstruction failure.
    7. Text Truncation — Text ending mid-word without punctuation
       → content cut off at extraction boundary.

A perfectly parsed document scores 1.0 regardless of how much empty
space, blank cells, or sparse text it contains.
"""
from __future__ import annotations


import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set

from ..base import BaseMiddleware
from ...models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Detection Patterns
# ══════════════════════════════════════════════════════════════════

# Mojibake / garbled character indicators
_RE_GARBLED = re.compile(
    r'[\ufffd\ufffe\uffff]|'          # Unicode replacement / invalid
    r'[\x00-\x08\x0b\x0c\x0e-\x1f]'  # Control chars (excluding \t \n \r)
)

# Inter-character spacing anomaly: 3+ single-char-then-space sequences
# Matches patterns like "H e l l o" or "账 户 名 称"
_RE_CHAR_SPACED = re.compile(
    r'(?:\S\s){3,}\S'
)

# CJK character range for language-aware truncation detection
_RE_CJK = re.compile(
    r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]'
)

# Natural sentence endings (multilingual)
_RE_SENTENCE_END = re.compile(
    r'[.!?。！？；;:：\-—)\]》）】」』\d%‰]$'
)

# Penalty weight caps — total pool sums to 100%
_PENALTY_CAPS = {
    "column_misalignment":  0.25,  # Up to 25%
    "encoding_errors":      0.20,  # Up to 20%
    "page_loss":            0.15,  # Up to 15%
    "duplicate_rows":       0.15,  # Up to 15%
    "header_repetition":    0.10,  # Up to 10%
    "whitespace_anomaly":   0.10,  # Up to 10%
    "text_truncation":      0.05,  # Up to 5%
}


class Validator(BaseMiddleware):
    """
    Mirror Fidelity Assessment Middleware.

    Penalty-based scoring: starts at 1.0 and deducts only for
    detected parsing artifacts. Document characteristics (empty cells,
    short text) are explicitly NOT penalized.

    Injects results into ``EnhancedResult.enhanced_data["validation"]``.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        pass_threshold: float = 0.7,
    ):
        super().__init__(config)
        self.pass_threshold = pass_threshold

    def process(self, result: EnhancedResult) -> EnhancedResult:
        """Assess mirror fidelity of the parsed document."""
        # ─── Collect data from base_result ───
        tables: List[List[List[str]]] = []
        all_text_content: List[str] = []
        block_pages: Set[int] = set()
        total_page_count: int = 0

        if result.base_result:
            total_page_count = result.base_result.page_count
            for block in result.base_result.all_blocks:
                block_pages.add(block.page)
                if block.block_type == "table" and isinstance(
                    block.raw_content, list
                ):
                    tables.append(block.raw_content)
                    for row in block.raw_content:
                        if isinstance(row, list):
                            all_text_content.extend(
                                c for c in row if isinstance(c, str)
                            )
                elif isinstance(block.raw_content, str):
                    all_text_content.append(block.raw_content)

        # ─── Detect all 7 artifact categories ───
        details: Dict[str, float] = {}
        penalties: Dict[str, float] = {}

        # 1. Column Misalignment
        score = self._check_column_alignment(tables)
        details["column_alignment"] = score
        if score < 1.0:
            penalties["column_misalignment"] = round(
                (1.0 - score) * _PENALTY_CAPS["column_misalignment"], 4
            )

        # 2. Encoding Errors
        score = self._check_encoding_fidelity(all_text_content)
        details["encoding_fidelity"] = score
        if score < 1.0:
            penalties["encoding_errors"] = round(
                (1.0 - score) * _PENALTY_CAPS["encoding_errors"], 4
            )

        # 3. Page Loss
        score = self._check_page_coverage(block_pages, total_page_count)
        details["page_coverage"] = score
        if score < 1.0:
            penalties["page_loss"] = round(
                (1.0 - score) * _PENALTY_CAPS["page_loss"], 4
            )

        # 4. Duplicate Rows
        score = self._check_duplicate_rows(tables)
        details["row_uniqueness"] = score
        if score < 1.0:
            penalties["duplicate_rows"] = round(
                (1.0 - score) * _PENALTY_CAPS["duplicate_rows"], 4
            )

        # 5. Header Repetition
        score = self._check_header_repetition(tables)
        details["header_uniqueness"] = score
        if score < 1.0:
            penalties["header_repetition"] = round(
                (1.0 - score) * _PENALTY_CAPS["header_repetition"], 4
            )

        # 6. Whitespace Anomaly
        score = self._check_whitespace_anomaly(all_text_content)
        details["whitespace_fidelity"] = score
        if score < 1.0:
            penalties["whitespace_anomaly"] = round(
                (1.0 - score) * _PENALTY_CAPS["whitespace_anomaly"], 4
            )

        # 7. Text Truncation
        score = self._check_text_truncation(all_text_content)
        details["text_completeness"] = score
        if score < 1.0:
            penalties["text_truncation"] = round(
                (1.0 - score) * _PENALTY_CAPS["text_truncation"], 4
            )

        # ─── Composite mirror fidelity ───
        total_penalty = min(1.0, sum(penalties.values()))
        total_score = round(1.0 - total_penalty, 4)
        passed = total_score >= self.pass_threshold

        # ─── Image quality assessment (from PreAnalyzer) ───
        image_quality = self._assess_image_quality_from_metadata(
            result, total_score,
        )

        validation = {
            "passed": passed,
            "total_score": total_score,
            "details": details,
            "penalties": penalties,
            "image_quality": image_quality,
            "threshold": self.pass_threshold,
            "row_count": sum(
                max(0, len(t) - 1) for t in tables
            ) if tables else 0,
        }

        result.enhanced_data["validation"] = validation

        result.record_mutation(
            middleware_name=self.name,
            target_block_id="document",
            field_changed="validation",
            old_value=None,
            new_value=f"fidelity={total_score:.3f} passed={passed}",
            confidence=1.0,
            reason=f"mirror fidelity: penalties={penalties}",
        )

        logger.info(
            f"[Validator] fidelity={total_score:.3f} | "
            f"passed={passed} | penalties={penalties}"
        )

        return result

    # ════════════════════════════════════════════════════════════════
    # 7 Artifact Detection Dimensions
    # ════════════════════════════════════════════════════════════════

    # ── 1. Column Misalignment ──────────────────────────────────────

    @staticmethod
    def _check_column_alignment(
        tables: List[List[List[str]]],
    ) -> float:
        """
        Fraction of data rows whose column count matches the header.

        Mismatches indicate the table parser failed at cell boundary
        detection. Empty cells within correctly-structured rows are
        NOT penalized.
        """
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

    # ── 2. Encoding Errors ──────────────────────────────────────────

    @staticmethod
    def _check_encoding_fidelity(texts: List[str]) -> float:
        """
        Absence of garbled characters across all extracted text.

        Replacement chars (U+FFFD), control characters, and unassigned
        Unicode codepoints are signs of byte-to-text decoding failure.
        """
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

    # ── 3. Page Loss ────────────────────────────────────────────────

    @staticmethod
    def _check_page_coverage(
        block_pages: Set[int],
        total_page_count: int,
    ) -> float:
        """
        Fraction of physical pages that produced at least one block.

        Missing pages indicate entire pages were silently dropped.
        """
        if total_page_count <= 1:
            return 1.0
        if not block_pages:
            return 0.0
        return len(block_pages) / total_page_count

    # ── 4. Duplicate Rows ──────────────────────────────────────────

    @staticmethod
    def _check_duplicate_rows(
        tables: List[List[List[str]]],
    ) -> float:
        """
        Fraction of table rows that are NOT consecutive duplicates.

        Consecutive identical rows strongly suggest the parser extracted
        the same row twice (e.g., at page boundaries). Non-consecutive
        duplicates may be legitimate (repeated transactions, etc.) and
        are NOT penalized.
        """
        total = 0
        duplicates = 0
        for table in tables:
            rows = table[1:] if len(table) > 1 else []
            for i, row in enumerate(rows):
                total += 1
                if i > 0 and row == rows[i - 1]:
                    duplicates += 1
        return (total - duplicates) / total if total > 0 else 1.0

    # ── 5. Header Repetition ───────────────────────────────────────

    @staticmethod
    def _check_header_repetition(
        tables: List[List[List[str]]],
    ) -> float:
        """
        Fraction of data rows that are NOT duplicates of the header.

        When headers re-appear as data rows, it indicates the parser
        failed to strip repeated headers from multi-page tables.
        """
        total = 0
        repeated = 0
        for table in tables:
            if len(table) < 2:
                continue
            header = table[0]
            # Normalize for comparison: strip whitespace, lowercase
            header_norm = [c.strip().lower() for c in header]
            for row in table[1:]:
                total += 1
                row_norm = [c.strip().lower() for c in row]
                if row_norm == header_norm:
                    repeated += 1
        return (total - repeated) / total if total > 0 else 1.0

    # ── 6. Whitespace Anomaly ──────────────────────────────────────

    @staticmethod
    def _check_whitespace_anomaly(texts: List[str]) -> float:
        """
        Absence of inter-character spacing artifacts.

        Patterns like "H e l l o" or "账 户 名 称" indicate the OCR
        or text extraction incorrectly inserted spaces between every
        character. Normal word spacing is NOT penalized.
        """
        total_segments = 0
        anomalous_segments = 0
        for text in texts:
            if not text or len(text) < 5:
                continue
            total_segments += 1
            if _RE_CHAR_SPACED.search(text):
                anomalous_segments += 1
        return (
            (total_segments - anomalous_segments) / total_segments
            if total_segments > 0 else 1.0
        )

    # ── 7. Text Truncation ─────────────────────────────────────────

    @staticmethod
    def _check_text_truncation(texts: List[str]) -> float:
        """
        Fraction of substantial text blocks that end naturally.

        Text ending mid-word without punctuation suggests the content
        was cut off at an extraction boundary. Short texts (< 10 chars),
        single-word values, and table cell fragments are excluded from
        this check since they naturally lack terminal punctuation.
        """
        checked = 0
        truncated = 0
        for text in texts:
            if not text:
                continue
            stripped = text.strip()
            # Only check substantial text blocks (multi-word, >10 chars)
            if len(stripped) < 10 or ' ' not in stripped:
                continue
            checked += 1
            # Check if it ends with a natural terminator
            if not _RE_SENTENCE_END.search(stripped):
                # For CJK text, ending on a CJK char is natural
                if stripped and _RE_CJK.match(stripped[-1]):
                    continue
                # Ending with a digit is natural (IDs, dates, amounts, page numbers)
                if stripped and stripped[-1].isdigit():
                    continue
                truncated += 1
        return (checked - truncated) / checked if checked > 0 else 1.0

    # ── 8. Image Quality Assessment (from PreAnalyzer) ────────────

    @staticmethod
    def _assess_image_quality_from_metadata(
        result: EnhancedResult,
        fidelity_score: float,
    ) -> Dict[str, Any]:
        """
        Extract image quality metrics from PreAnalyzer data and produce
        a VLM enhancement recommendation.

        The PreAnalyzer already computes per-page image quality via
        Laplacian variance (sharpness detection). This method surfaces
        those metrics and decides whether a VLM API call would improve
        recognition accuracy.

        VLM recommendation triggers:
            - Scanned pages with avg_image_quality < 60
            - Mirror fidelity score < 0.8 (parsing artifacts detected)
            - Entirely scanned document (no native text layer)
            - Whitespace anomalies or encoding errors present
        """
        pre = {}
        if result.base_result:
            pre = result.base_result.metadata.get("pre_analysis", {})

        avg_quality = pre.get("avg_image_quality", 100)
        scanned_pages = pre.get("estimated_scanned_pages", 0)
        total_pages = pre.get("num_pages", 0)
        content_type = pre.get("content_type", "unknown")
        has_text_layer = pre.get("has_text_layer", True)
        quality_score = pre.get("quality_score", 1.0)

        # Determine VLM recommendation
        vlm_reasons: List[str] = []

        if scanned_pages > 0 and avg_quality < 60:
            vlm_reasons.append(
                f"low_image_quality: avg={avg_quality}/100 "
                f"on {scanned_pages} scanned pages"
            )

        if content_type == "scanned":
            vlm_reasons.append(
                "fully_scanned: no native text layer detected"
            )

        if fidelity_score < 0.8:
            vlm_reasons.append(
                f"low_fidelity: mirror_score={fidelity_score:.3f} "
                f"suggests parsing artifacts"
            )

        if not has_text_layer and total_pages > 0:
            vlm_reasons.append(
                "no_text_layer: document requires full visual recognition"
            )

        if quality_score < 0.5:
            vlm_reasons.append(
                f"poor_document_quality: score={quality_score:.2f}"
            )

        return {
            "avg_image_quality": avg_quality,
            "scanned_pages": scanned_pages,
            "total_pages": total_pages,
            "content_type": content_type,
            "has_text_layer": has_text_layer,
            "document_quality_score": round(quality_score, 2),
            "vlm_recommended": len(vlm_reasons) > 0,
            "vlm_reasons": vlm_reasons,
        }

