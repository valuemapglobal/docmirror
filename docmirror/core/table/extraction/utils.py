# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utility functions split from table_extraction.py."""

from __future__ import annotations

import bisect
import logging
from typing import Dict, List, Tuple

from ...utils.text_utils import _is_cjk_char, _smart_join

logger = logging.getLogger(__name__)


# ── Shared utility functions ──


def _adaptive_row_tolerance(chars: list[dict]) -> float:
    """F-1: Calculate adaptive y-tolerance for row grouping.

    Based on the median character height, dynamically adjusts the tolerance
    to prevent:
      - Small font-size PDFs: fixed 3 pt tolerance merging multiple lines.
      - Large font-size PDFs: 3 pt tolerance splitting same-row characters.

    Returns:
        Adaptive tolerance value (typically 1.5–5.0 pt).
    """
    if not chars or len(chars) < 5:
        return 3.0

    heights = [
        c["bottom"] - c["top"] for c in chars if c.get("bottom", 0) > c.get("top", 0) and c["bottom"] - c["top"] < 30
    ]
    if not heights:
        return 3.0

    heights.sort()
    median_h = heights[len(heights) // 2]
    # Tolerance = median character height × 0.6, clamped to [1.5, 5.0]
    tol = max(1.5, min(5.0, median_h * 0.6))
    return tol


def _group_chars_into_rows(chars: list[dict], y_tolerance: float = 3.0) -> list[tuple[float, list[dict]]]:
    """Group characters into rows by y-coordinate proximity.

    F-1 enhancement: when ``y_tolerance <= 0``, automatically uses
    ``_adaptive_row_tolerance``.
    """
    if not chars:
        return []

    # F-1: adaptive tolerance
    if y_tolerance <= 0:
        y_tolerance = _adaptive_row_tolerance(chars)

    sorted_chars = sorted(chars, key=lambda c: c["top"])
    rows: list[tuple[float, list[dict]]] = []
    current_row: list[dict] = [sorted_chars[0]]
    current_y = sorted_chars[0]["top"]

    for c in sorted_chars[1:]:
        if abs(c["top"] - current_y) <= y_tolerance:
            current_row.append(c)
        else:
            y_mid = sum(ch["top"] for ch in current_row) / len(current_row)
            rows.append((y_mid, sorted(current_row, key=lambda x: x["x0"])))
            current_row = [c]
            current_y = c["top"]

    if current_row:
        y_mid = sum(ch["top"] for ch in current_row) / len(current_row)
        rows.append((y_mid, sorted(current_row, key=lambda x: x["x0"])))

    return rows


def _cluster_x_positions(
    x_coords: list[float], gap_multiplier: float = 2.0, min_col_width: float = 10.0
) -> list[tuple[float, float]]:
    """X-coordinate clustering: find column boundaries.

    Optimisation 3: uses an IQR-inspired adaptive threshold (natural-break)
    instead of ``median × multiplier``, making it more robust for narrow
    inter-column gaps.  Falls back to ``median × multiplier`` when there
    are too few gap samples (< 4).
    """
    if not x_coords:
        return []

    sorted_x = sorted(set(round(x, 1) for x in x_coords))
    if len(sorted_x) < 2:
        return [(sorted_x[0], sorted_x[0] + 50)]

    gaps = [sorted_x[i + 1] - sorted_x[i] for i in range(len(sorted_x) - 1)]
    non_zero_gaps = sorted(g for g in gaps if g > 0.5)

    if not non_zero_gaps:
        return [(sorted_x[0], sorted_x[-1])]

    # ── Adaptive threshold (natural break) ──
    # Column gaps typically follow a bimodal distribution:
    #   small gaps = intra-column character spacing
    #   large gaps = inter-column spacing
    # Find the largest jump in the sorted gaps to set the threshold
    median_gap = non_zero_gaps[len(non_zero_gaps) // 2]

    if len(non_zero_gaps) >= 4:
        # Find the largest adjacent jump in sorted gaps
        max_jump = 0
        jump_idx = -1
        for j in range(len(non_zero_gaps) - 1):
            jump = non_zero_gaps[j + 1] - non_zero_gaps[j]
            if jump > max_jump:
                max_jump = jump
                jump_idx = j

        if max_jump > median_gap * 2 and jump_idx >= 0:
            # Clear bimodal distribution: threshold = midpoint of the jump
            threshold = (non_zero_gaps[jump_idx] + non_zero_gaps[jump_idx + 1]) / 2
        else:
            # Continuous distribution: fall back to median × multiplier
            threshold = median_gap * gap_multiplier
    else:
        # Too few data points: fall back to original logic
        threshold = median_gap * gap_multiplier

    col_bounds: list[tuple[float, float]] = []
    col_start = sorted_x[0]

    for i, gap in enumerate(gaps):
        if gap > threshold:
            col_end = sorted_x[i]
            if col_end - col_start >= min_col_width:
                col_bounds.append((col_start, col_end))
            col_start = sorted_x[i + 1]

    col_end = sorted_x[-1]
    if col_end - col_start >= min_col_width:
        col_bounds.append((col_start, col_end))

    return col_bounds


def _assign_chars_to_columns(row_chars: list[dict], col_bounds: list[tuple[float, float]]) -> list[str]:
    """Assign a row's characters to column bins using divider midpoints.

    Divider lines are placed at the midpoint between adjacent column
    boundaries.  This is more precise than fixed-tolerance binning
    (unaffected by uneven column widths) and more stable than nearest-
    centre (unaffected by wide/narrow column asymmetry).
    """
    if not col_bounds:
        return []

    cells = ["" for _ in col_bounds]

    # Compute divider lines between adjacent columns
    dividers = [col_bounds[0][0] - 10]  # Left boundary
    for i in range(len(col_bounds) - 1):
        mid = (col_bounds[i][1] + col_bounds[i + 1][0]) / 2
        dividers.append(mid)
    dividers.append(col_bounds[-1][1] + 10)  # Right boundary

    # Merge adjacent characters into words (prevents word-boundary splitting)
    sorted_chars = sorted(row_chars, key=lambda x: x["x0"])
    words = []
    curr_word = None

    def is_cjk(ch):
        return _is_cjk_char(ch)

    for c in sorted_chars:
        if not str(c.get("text", "")).strip():
            continue
        if not curr_word:
            curr_word = {"x0": c["x0"], "x1": c.get("x1", c["x0"]), "text": c["text"]}
        else:
            gap = c["x0"] - curr_word["x1"]
            is_prev_cjk = _is_cjk_char(curr_word["text"][-1]) if curr_word["text"] else False
            is_curr_cjk = _is_cjk_char(c["text"])
            threshold = 5.0 if (is_prev_cjk or is_curr_cjk) else 2.5

            if gap < threshold:
                curr_word["x1"] = max(curr_word["x1"], c.get("x1", c["x0"]))
                curr_word["text"] += c["text"]
            else:
                words.append(curr_word)
                curr_word = {"x0": c["x0"], "x1": c.get("x1", c["x0"]), "text": c["text"]}
    if curr_word:
        words.append(curr_word)

    for w in words:
        char_x = (w["x0"] + w["x1"]) / 2
        # O(log c) binary search for the containing column
        col_idx = bisect.bisect_right(dividers, char_x) - 1
        col_idx = max(0, min(col_idx, len(cells) - 1))
        if 0 <= col_idx < len(cells):
            if cells[col_idx]:
                existing = cells[col_idx].strip()
                val = w.get("text", "").strip()
                if existing and val:
                    cells[col_idx] = _smart_join(existing, val)
                else:
                    cells[col_idx] += w.get("text", "")
            else:
                cells[col_idx] += w.get("text", "")

    return [cell.strip() for cell in cells]


def _chars_to_text(chars: list[dict]) -> str:
    """Merge a list of character dicts into a single text string."""
    if not chars:
        return ""
    sorted_c = sorted(chars, key=lambda c: c["x0"])
    parts = [sorted_c[0].get("text", "")]
    for i in range(1, len(sorted_c)):
        gap = sorted_c[i]["x0"] - sorted_c[i - 1].get("x1", sorted_c[i - 1]["x0"])
        if gap > 3:
            parts.append(" ")
        parts.append(sorted_c[i].get("text", ""))
    return "".join(parts)
