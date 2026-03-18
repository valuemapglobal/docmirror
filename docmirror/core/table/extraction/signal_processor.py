# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Dual-Axis Signal Processor for Table Extraction
==================================================

Replaces 4 parallel char-level extraction methods with a single-pass
1D signal processing approach:

  1. **X-axis projection**: Character density histogram → column boundaries
  2. **Y-axis projection**: Character density histogram → row boundaries
  3. **Grid assembly**: bisect-based cell binning → 2D table

All algorithms are O(n) in character count, CPU-only, no external
dependencies beyond the Python standard library.
"""

from __future__ import annotations

import bisect
import logging
from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Optional, Tuple

from ...utils.text_utils import _is_cjk_char, _smart_join
from ...utils.vocabulary import (
    _RE_IS_AMOUNT,
    _RE_IS_DATE,
    _is_header_cell,
    _score_header_by_vocabulary,
)
from ...utils.watermark import is_watermark_char

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# 1. Dual-Axis Projection
# ═════════════════════════════════════════════════════════════════════


def project_chars_to_axes(
    chars: list[dict],
    page_w: float,
    page_h: float,
    resolution: float = 1.0,
    global_tensor_x: list[float] | None = None,
) -> tuple[list[int], list[int]]:
    """Project characters onto X and Y axes as 1D density signals.

    Each character's bounding box ``[x0, x1]`` increments the X-axis
    signal, and ``[top, bottom]`` increments the Y-axis signal.
    Density = number of rows (for X) or columns (for Y) that have
    content at that coordinate.

    Args:
        chars: Character dicts with x0, x1, top, bottom keys.
        page_w: Page width in points.
        page_h: Page height in points.
        resolution: Signal resolution (points per bin). Default 1.0.
        global_tensor_x: Document-level alignment matrix to blend into D_x.

    Returns:
        ``(D_x, D_y)`` — X-axis and Y-axis density signals.
    """
    w_bins = int(page_w / resolution) + 2
    h_bins = int(page_h / resolution) + 2
    D_x = [0] * w_bins
    D_y = [0] * h_bins

    # Group chars into rows first (for X-projection: count distinct rows)
    # and into columns (for Y-projection: count distinct columns)
    # Simple approach: each char contributes 1 to both projections
    # S5: Use itemgetter for C-speed multi-key extraction
    _get_bbox = itemgetter("x0", "x1", "top", "bottom")
    for c in chars:
        raw = _get_bbox(c)
        cx0 = max(0, int(raw[0] / resolution))
        cx1 = min(w_bins - 1, int((raw[1] if raw[1] else raw[0]) / resolution))
        cy0 = max(0, int(raw[2] / resolution))
        cy1 = min(h_bins - 1, int((raw[3] if raw[3] else raw[2]) / resolution))

        for x in range(cx0, cx1 + 1):
            D_x[x] += 1
        for y in range(cy0, cy1 + 1):
            D_y[y] += 1

    # Apply Magnetic Signal Blending: Enforce document-level column boundaries
    # by raising the density where strong structural anchors exist.
    if global_tensor_x:
        for x in range(min(len(D_x), len(global_tensor_x))):
            D_x[x] += global_tensor_x[x]

    return D_x, D_y


# ═════════════════════════════════════════════════════════════════════
# 2. Valley Detection
# ═════════════════════════════════════════════════════════════════════


def detect_valleys(
    signal: list[int],
    min_width: float = 2.0,
    threshold_ratio: float = 0.10,
    resolution: float = 1.0,
) -> list[tuple[float, float, float]]:
    """Detect valleys (low-density bands) in a 1D signal.

    A valley is a contiguous region where the signal value is below
    ``max_signal * threshold_ratio``.

    Args:
        signal: 1D density signal.
        min_width: Minimum valley width in points to qualify as boundary.
        threshold_ratio: Valley threshold as fraction of peak signal value.
        resolution: Coordinate resolution (points per bin).

    Returns:
        List of ``(center_pt, width_pt, avg_depth)`` tuples, sorted by
        center position.
    """
    if not signal:
        return []

    peak = max(signal)
    if peak == 0:
        return []

    threshold = peak * threshold_ratio
    min_bins = max(1, int(min_width / resolution))

    valleys = []
    in_valley = False
    valley_start = 0

    for i in range(len(signal)):
        if signal[i] <= threshold:
            if not in_valley:
                valley_start = i
                in_valley = True
        else:
            if in_valley:
                width = i - valley_start
                if width >= min_bins:
                    center = (valley_start + i - 1) / 2 * resolution
                    width_pt = width * resolution
                    avg_depth = sum(signal[valley_start:i]) / width
                    valleys.append((center, width_pt, avg_depth))
                in_valley = False

    # Handle trailing valley
    if in_valley:
        width = len(signal) - valley_start
        if width >= min_bins:
            center = (valley_start + len(signal) - 1) / 2 * resolution
            width_pt = width * resolution
            avg_depth = sum(signal[valley_start:]) / width
            valleys.append((center, width_pt, avg_depth))

    return valleys


# ═════════════════════════════════════════════════════════════════════
# 3. Grid Assembly
# ═════════════════════════════════════════════════════════════════════


def build_grid(
    row_boundaries: list[float],
    col_boundaries: list[float],
    chars: list[dict],
) -> list[list[str]]:
    """Assemble characters into a 2D grid using row/column boundaries.

    Uses ``bisect.bisect_right`` for O(log c) column lookup per character,
    and O(log r) row lookup per character.

    Args:
        row_boundaries: Y-coordinates separating rows.
        col_boundaries: X-coordinates separating columns.
        chars: Character dicts with x0, x1, top, bottom, text keys.

    Returns:
        2D table as list of lists of strings.
    """
    n_rows = len(row_boundaries) + 1
    n_cols = len(col_boundaries) + 1

    if n_rows < 2 or n_cols < 2:
        return []

    # Initialize grid
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    # Pre-sort chars for word merging
    sorted_chars = sorted(chars, key=lambda c: (c.get("top", 0), c.get("x0", 0)))

    # Group chars into rows first using bisect
    row_bins: dict[int, list[dict]] = defaultdict(list)
    for c in sorted_chars:
        cy = (c.get("top", 0) + c.get("bottom", 0)) / 2
        ri = bisect.bisect_right(row_boundaries, cy)
        ri = max(0, min(ri, n_rows - 1))
        row_bins[ri].append(c)

    # For each row, merge chars into words, then assign to columns
    for ri in range(n_rows):
        row_chars = row_bins.get(ri, [])
        if not row_chars:
            continue

        # Merge adjacent chars into words
        words = _merge_chars_to_words(sorted(row_chars, key=lambda c: c.get("x0", 0)))

        # Assign words to columns via bisect
        for w in words:
            wx = (w["x0"] + w["x1"]) / 2
            ci = bisect.bisect_right(col_boundaries, wx)
            ci = max(0, min(ci, n_cols - 1))

            if grid[ri][ci]:
                grid[ri][ci] = _smart_join(grid[ri][ci].strip(), w["text"].strip())
            else:
                grid[ri][ci] = w["text"]

    # Strip all cells
    return [[cell.strip() for cell in row] for row in grid]


def _merge_chars_to_words(sorted_chars: list[dict]) -> list[dict]:
    """Merge adjacent characters into words based on gap distance."""
    if not sorted_chars:
        return []

    words = []
    curr = None

    for c in sorted_chars:
        text = str(c.get("text", "")).strip()
        if not text:
            continue
        if curr is None:
            curr = {"x0": c["x0"], "x1": c.get("x1", c["x0"]), "text": text}
        else:
            gap = c["x0"] - curr["x1"]
            is_prev_cjk = _is_cjk_char(curr["text"][-1]) if curr["text"] else False
            is_curr_cjk = _is_cjk_char(text[0]) if text else False
            threshold = 5.0 if (is_prev_cjk or is_curr_cjk) else 2.5

            if gap < threshold:
                curr["x1"] = max(curr["x1"], c.get("x1", c["x0"]))
                curr["text"] += text
            else:
                words.append(curr)
                curr = {"x0": c["x0"], "x1": c.get("x1", c["x0"]), "text": text}

    if curr:
        words.append(curr)

    return words


# ═════════════════════════════════════════════════════════════════════
# 4. Unified Entry Point
# ═════════════════════════════════════════════════════════════════════


def extract_table_by_signal(
    page_plum,
    min_cols: int = 3,
    min_rows: int = 3,
    global_tensor_x: list[float] | None = None,
    pid_resample: bool = False,
) -> list[list[str]] | None:
    """Extract a table using dual-axis 1D signal processing.

    Single-pass algorithm:
      1. Filter watermark chars and collect text characters.
      2. Project onto X/Y axes to get density signals.
      3. Detect valleys (column gaps and row gaps).
      4. Assemble into a 2D grid using bisect binning.
      5. Validate header and trim empty columns.

    Args:
        page_plum: pdfplumber page object.
        min_cols: Minimum columns to qualify as a table.
        min_rows: Minimum rows to qualify as a table.
        global_tensor_x: Document-level grid alignment tensor.

    Returns:
        2D table (list of list of str), or None if no table detected.
    """
    chars = page_plum.chars
    if not chars or len(chars) < 20:
        return None

    page_w = page_plum.width or 600
    page_h = page_plum.height or 800

    # ── Step 0: Filter watermark and non-text characters ──
    text_chars = [c for c in chars if c.get("text", "").strip() and not is_watermark_char(c)]
    if len(text_chars) < 15:
        return None

    # ── Step 1: Dual-axis projection ──
    D_x, D_y = project_chars_to_axes(text_chars, page_w, page_h, global_tensor_x=global_tensor_x)

    # ── Step 2: Column detection (X-axis valleys) ──
    # Use adaptive gap width based on average character width
    char_widths = [c.get("x1", 0) - c.get("x0", 0) for c in text_chars if c.get("x1", 0) > c.get("x0", 0)]
    avg_char_w = sum(char_widths) / len(char_widths) if char_widths else 6.0
    min_col_gap = max(2.0, avg_char_w * 0.5)
    threshold_ratio = 0.10

    # ── PID Loop Resampling Degradation ──
    # If this is a retry due to low confidence, tighten the threshold
    # and lower the gap requirement to forcefully expose swallowed columns
    if pid_resample:
        threshold_ratio = 0.05
        min_col_gap = max(1.0, min_col_gap / 1.5)
        import logging

        logging.getLogger(__name__).debug(
            f"PID Resample Triggered: tightening threshold_ratio to {threshold_ratio} and min_col_gap to {min_col_gap:.1f}"
        )

    x_valleys = detect_valleys(D_x, min_width=min_col_gap, threshold_ratio=threshold_ratio)

    if len(x_valleys) < min_cols - 1:
        # Not enough column gaps → not a table
        return None

    col_boundaries = [v[0] for v in x_valleys]

    # ── Step 3: Row detection (Y-axis valleys) ──
    # Row gaps are typically narrower than column gaps
    char_heights = [c.get("bottom", 0) - c.get("top", 0) for c in text_chars if c.get("bottom", 0) > c.get("top", 0)]
    if not char_heights:
        return None
    sorted_h = sorted(char_heights)
    median_h = sorted_h[len(sorted_h) // 2]
    min_row_gap = max(1.0, median_h * 0.3)

    y_valleys = detect_valleys(D_y, min_width=min_row_gap, threshold_ratio=0.05)

    if len(y_valleys) < min_rows - 1:
        return None

    row_boundaries = [v[0] for v in y_valleys]

    # ── Step 4: Grid assembly ──
    table = build_grid(row_boundaries, col_boundaries, text_chars)

    if not table or len(table) < min_rows:
        return None

    # Count actual columns (non-empty in at least 1 row)
    n_cols = len(table[0]) if table else 0
    if n_cols < min_cols:
        return None

    # ── Step 5: Trim fully empty columns ──
    non_empty_cols = [ci for ci in range(n_cols) if any(row[ci].strip() for row in table if ci < len(row))]
    if len(non_empty_cols) < min_cols:
        return None

    if len(non_empty_cols) < n_cols:
        table = [[row[ci] for ci in non_empty_cols] for row in table]

    # ── Step 6: Trim fully empty rows ──
    table = [row for row in table if any(cell.strip() for cell in row)]

    if len(table) < min_rows:
        return None

    # ── Step 7: Validate header quality ──
    # Find best header row in first 5 rows
    best_header_idx = 0
    best_score = _score_header_by_vocabulary(table[0]) if table else 0
    for ri in range(1, min(5, len(table))):
        score = _score_header_by_vocabulary(table[ri])
        if score > best_score:
            best_score = score
            best_header_idx = ri

    # Skip preamble rows if header is not at position 0
    if best_header_idx > 0 and best_score >= 3:
        table = table[best_header_idx:]

    if len(table) < min_rows:
        return None

    logger.debug(
        f"signal_processor: {len(table)} rows × {len(table[0])} cols, "
        f"x_valleys={len(x_valleys)}, y_valleys={len(y_valleys)}, "
        f"header_vocab={best_score}"
    )

    return table
