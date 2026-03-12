"""
Table pre-classification, confidence scoring, and validation gates.

Split from ``table_extraction.py``.
"""
from __future__ import annotations


import contextvars
import logging
import math
import re
from typing import Dict, List

from ...utils.text_utils import _is_cjk_char, _smart_join
from ...utils.vocabulary import PIPE_CHARS, _ALL_BORDER_CHARS, _is_header_row, _normalize_for_vocab, _score_header_by_vocabulary, _RE_IS_DATE, _RE_IS_AMOUNT

logger = logging.getLogger(__name__)

TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "lines",
    "snap_x_tolerance": 5,
    "join_x_tolerance": 5,
    "snap_y_tolerance": 3,
}

TABLE_SETTINGS_LINES = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
}

# ── contextvars: thread-safe / async-safe layer timings ──
_layer_timings_var: contextvars.ContextVar[Dict[str, float]] = contextvars.ContextVar(
    'layer_timings', default={}
)


def get_last_layer_timings() -> Dict[str, float]:
    """Return per-layer timing (ms) from the most recent ``extract_tables_layered``
    call in the current context."""
    return dict(_layer_timings_var.get({}))


def _quick_classify(work_page) -> str:
    """Table pre-classification: suggest a starting layer based on quick
    feature heuristics, allowing later layers to be skipped.

    Returns a label for the suggested starting layer:
      - ``'pipe'``  : pipe characters >= 10 → start at L0.5 (default)
      - ``'lines'`` : >= 6 PDF drawing lines → start at L1
      - ``'text'``  : some lines + dispersed x-coordinates → start at L1b
      - ``'char'``  : no lines / no pipes → jump straight to L2 char-level
    """
    chars = work_page.chars or []
    lines = work_page.lines or []

    # Feature 1: pipe character count
    pipe_count = sum(1 for c in chars if c.get("text") in PIPE_CHARS)
    if pipe_count >= 10:
        return "pipe"

    # Feature 2: line count (horizontal + vertical)
    h_lines = [l for l in lines if abs(l.get("top", 0) - l.get("bottom", 0)) < 1]
    v_lines = [l for l in lines if abs(l.get("x0", 0) - l.get("x1", 0)) < 1]
    total_lines = len(h_lines) + len(v_lines)

    if total_lines >= 6:  # Enough lines → take the line-based path
        return "lines"

    # Feature 3: x-coordinate dispersion (text-aligned table vs plain text)
    if chars:
        x_positions = set(round(c["x0"] / 10) * 10 for c in chars)
        if len(x_positions) >= 5:  # Dispersed x → possibly a borderless table
            if total_lines >= 3:    # Some horizontal lines → text strategy may work better
                return "text"
            return "char"          # No lines → go directly to char-level

    return "pipe"  # Default: start from the beginning


def _compute_table_confidence(
    tables: List[List[List[str]]],
    layer: str,
) -> float:
    """Compute an extraction confidence score (0.0–1.0) for a table result.

    Factors considered:
      - ``vocab_score``: header vocabulary hit count (highest weight).
      - ``row_count``: more rows → higher confidence.
      - ``col_consistency``: ratio of rows with consistent column count.
      - ``layer_bonus``: earlier layers are inherently more trustworthy.
    """
    if not tables or not tables[0]:
        return 0.0

    tbl = tables[0]  # Primary table
    if len(tbl) < 1:
        return 0.0

    # 1. vocab_score (0–1, linear mapping: 0→0, 3→0.6, 5+→1.0)
    header = tbl[0]
    vocab = _score_header_by_vocabulary(header)
    vocab_norm = min(1.0, vocab / 5.0)

    # 2. row_count (0–1, logarithmic mapping: 2→0.3, 10→0.7, 50+→1.0)
    row_count = len(tbl)
    row_norm = min(1.0, math.log2(max(2, row_count)) / math.log2(50))

    # 3. col_consistency (0–1, ratio of rows matching the header column count)
    if len(tbl) >= 2:
        expected_cols = len(tbl[0])
        consistent = sum(1 for row in tbl if len(row) == expected_cols)
        col_norm = consistent / len(tbl)
    else:
        col_norm = 0.5

    # 4. layer_bonus (earlier layers are inherently more trustworthy)
    _LAYER_BONUS = {
        "pipe_delimited": 0.15, "lines": 0.15, "hline_columns": 0.10,
        "rect_columns": 0.10, "text": 0.10, "docling_tableformer": 0.10,
        "rapid_table": 0.12,  # Vision model: ranked above char-level strategies
        "header_anchors": 0.05, "word_anchors": 0.05,
        "data_voting": 0.05, "whitespace_projection": 0.05,
        "x_clustering": 0.0, "fallback": -0.10,
    }
    bonus = _LAYER_BONUS.get(layer, 0.0)

    # Weighted sum
    confidence = (vocab_norm * 0.40 + row_norm * 0.20 + col_norm * 0.25) + bonus
    return round(max(0.0, min(1.0, confidence)), 3)

def _cell_is_stuffed(cell: str) -> bool:
    """Detect whether a cell has had multiple rows of data stuffed into it
    (a symptom of row mis-merging).

    A normal cell should not simultaneously contain multiple dates or
    multiple amounts.  If any of the following are detected, it indicates
    the previous layer collapsed several records into a single cell:

      - >= 2 date patterns in one cell (e.g. '2025-09-21...2025-10-27...')
      - >= 4 amount patterns in one cell (e.g. '3000000.00 600000.00 ...')

    Notes:
      - Dates use ``(?:19|20)\\d{6}`` instead of ``\\d{8}`` to avoid
        treating phone numbers (e.g. 13883435811) as 8-digit dates.
      - Amounts use word-boundary regex to prevent substring double-counting
        (e.g. '3000000.00' containing '000.00').
      - No ``\\n`` check: hline_columns joins cell text with spaces, not newlines.
    """
    if not cell or len(cell) < 10:
        return False
        
    # Ignore preamble common text containing dates
    if "至" in cell and re.search(r"时间|期限|Period", cell):
        return False

    # Condition 1: >= 2 dates in a single cell
    # Note: (?:19|20)\d{6} matches only 8-digit dates starting with 19/20
    dates = re.findall(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?<!\d)(?:19|20)\d{6}(?!\d)', cell)
    if len(dates) >= 2:
        return True
    # Condition 2: >= 4 amounts in a single cell (word-boundary match)
    amounts = re.findall(r'(?<!\d)\d[\d,]*\.\d{2}(?!\d)', cell)
    if len(amounts) >= 4:
        return True
    return False


def _tables_look_valid(tables: list, min_rows: int = 2, has_borders: bool = False) -> bool:
    """Check whether extracted tables pass quality validation (including
    row density and stuffed-cell detection).

    Args:
        has_borders: If ``True``, the page has vertical border lines.
                     Skip stuffed-cell detection because borders guarantee
                     correct column structure and multi-line content is valid.
    """
    if not tables:
        return False
    for tbl in tables:
        if tbl and len(tbl) >= min_rows:
            col_count = len(tbl[0])
            if 2 <= col_count <= 30:
                # ── Check 1: abnormal average row character count ──
                # (all lines merged into one row causes character count to spike)
                total_chars = sum(
                    len(str(c or "")) for row in tbl for c in row
                )
                avg_chars_per_row = total_chars / len(tbl)
                if avg_chars_per_row > 500:
                    logger.warning(
                        f"table rejected: avg {avg_chars_per_row:.0f} "
                        f"chars/row > 500 → fallback to char-level"
                    )
                    return False
                # ── F-1: enhanced sampling check (first 4 + middle 2 + last 2) ──
                # Skip stuffed-cell detection for bordered tables
                if not has_borders:
                    n = len(tbl)
                    sample_indices = list(range(min(4, n)))
                    if n > 8:
                        mid = n // 2
                        sample_indices.extend([mid - 1, mid])
                    if n > 4:
                        sample_indices.extend([n - 2, n - 1])
                    sample_indices = sorted(set(i for i in sample_indices if 0 <= i < n))

                    for idx in sample_indices:
                        for cell in tbl[idx]:
                            if _cell_is_stuffed(str(cell or "")):
                                logger.warning(
                                    f"table rejected: stuffed cell at row {idx} "
                                    f"(cell={str(cell or '')[:40]!r}…) → fallback"
                                )
                                return False
                return True
    return False
