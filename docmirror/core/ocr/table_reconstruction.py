# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Table Reconstruction from OCR Characters
==========================================

Converts raw OCR character outputs into 2D table grids using spatial clustering.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def group_chars_into_rows(chars: list[dict], y_tolerance: float = 8.0) -> list[tuple[float, list[dict]]]:
    """Group OCR character dicts into rows by y-coordinate proximity."""
    if not chars:
        return []

    sorted_chars = sorted(chars, key=lambda c: c.get("top", 0))

    rows: list[tuple[float, list[dict]]] = []
    current_y = sorted_chars[0].get("top", 0)
    current_row: list[dict] = [sorted_chars[0]]

    for ch in sorted_chars[1:]:
        ch_y = ch.get("top", 0)
        if abs(ch_y - current_y) <= y_tolerance:
            current_row.append(ch)
        else:
            current_row.sort(key=lambda c: c.get("x0", 0))
            rows.append((current_y, current_row))
            current_y = ch_y
            current_row = [ch]

    if current_row:
        current_row.sort(key=lambda c: c.get("x0", 0))
        rows.append((current_y, current_row))

    return rows


def chars_to_text(chars: list[dict]) -> str:
    """Merge a list of character dicts into a single text string."""
    return " ".join(c.get("text", "") for c in chars).strip()


def cluster_x_positions(x_positions: list[float], gap_multiplier: float = 2.0) -> list[tuple[float, float]]:
    """Detect column boundaries by clustering x-coordinates."""
    if not x_positions:
        return []

    sorted_x = sorted(set(x_positions))
    if len(sorted_x) < 2:
        return [(sorted_x[0], sorted_x[0] + 100)]

    gaps = [sorted_x[i + 1] - sorted_x[i] for i in range(len(sorted_x) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 10

    col_starts = [sorted_x[0]]
    for i, gap in enumerate(gaps):
        if gap > median_gap * gap_multiplier:
            col_starts.append(sorted_x[i + 1])

    bounds = []
    for i, start in enumerate(col_starts):
        if i + 1 < len(col_starts):
            end = col_starts[i + 1]
        else:
            end = max(x_positions) + 10
        bounds.append((start, end))

    return bounds


def assign_chars_to_columns(chars: list[dict], col_bounds: list[tuple[float, float]]) -> list[str]:
    """Assign a row's characters to column bins."""
    cols: list[list[dict]] = [[] for _ in col_bounds]

    for ch in chars:
        cx = (ch.get("x0", 0) + ch.get("x1", 0)) / 2
        assigned = False
        for i, (start, end) in enumerate(col_bounds):
            if start <= cx < end:
                cols[i].append(ch)
                assigned = True
                break
        if not assigned and cols:
            min_dist = float("inf")
            min_idx = 0
            for i, (start, end) in enumerate(col_bounds):
                mid = (start + end) / 2
                dist = abs(cx - mid)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            cols[min_idx].append(ch)

    return [chars_to_text(col) for col in cols]


def split_tables_by_y_gap(
    rows_by_y: list[tuple[float, list[dict]]], page_h: float
) -> list[list[tuple[float, list[dict]]]]:
    """Split grouped rows into multiple tables based on vertical gaps."""
    if len(rows_by_y) < 4:
        return [rows_by_y]

    gap_threshold = page_h * 0.05
    tables: list[list[tuple[float, list[dict]]]] = []
    current: list[tuple[float, list[dict]]] = [rows_by_y[0]]

    for i in range(1, len(rows_by_y)):
        if rows_by_y[i][0] - rows_by_y[i - 1][0] > gap_threshold:
            tables.append(current)
            current = []
        current.append(rows_by_y[i])
    tables.append(current)

    return [t for t in tables if len(t) >= 2]


def reconstruct_table_grid_2d(
    chars: list[dict], hough_lines: list[tuple[float, float]] | None = None
) -> list[list[str]]:
    """Robust 2D Table Grid Reconstruction (Virtual Grid Alignment).

    Algorithm:
        1. Base Row Clustering: Group chars by y-overlap (IoU).
        2. Base Col Clustering: Group chars by x-overlap (IoU) or Hough lines.
        3. Grid Snapping: Assign each char to a (row_idx, col_idx) bucket.
        4. Output Generation: Build a dense 2D list of strings.
    """
    if not chars:
        return []

    # 1. Base Row Clustering
    sorted_chars = sorted(chars, key=lambda c: c["top"])
    rows_y = []

    for c in sorted_chars:
        c_min_y, c_max_y = c["top"], c["bottom"]
        matched = False
        for i in range(len(rows_y) - 1, max(-1, len(rows_y) - 4), -1):
            r_min_y, r_max_y, r_chars = rows_y[i]
            overlap = max(0, min(c_max_y, r_max_y) - max(c_min_y, r_min_y))
            c_height = c_max_y - c_min_y
            if overlap > 0.4 * c_height or (c_min_y >= r_min_y and c_max_y <= r_max_y):
                rows_y[i] = (min(r_min_y, c_min_y), max(r_max_y, c_max_y), r_chars + [c])
                matched = True
                break
        if not matched:
            rows_y.append((c_min_y, c_max_y, [c]))

    rows_y.sort(key=lambda x: x[0])
    row_chars_list = [r[2] for r in rows_y]

    # 2. Base Col Clustering
    col_bounds = []
    if hough_lines and len(hough_lines) >= 2:
        col_bounds = hough_lines
    else:
        x_spans = [(c["x0"], c["x1"]) for c in chars]
        x_spans.sort(key=lambda x: x[0])
        merged_cols = []
        for span in x_spans:
            if not merged_cols:
                merged_cols.append([span[0], span[1]])
                continue
            last_col = merged_cols[-1]
            if span[0] <= last_col[1] + 10:
                last_col[1] = max(last_col[1], span[1])
            else:
                merged_cols.append([span[0], span[1]])

        if len(merged_cols) < 2:
            all_x0 = [c["x0"] for c in chars]
            col_bounds = cluster_x_positions(all_x0, gap_multiplier=2.0)
        else:
            col_bounds = [(c[0], c[1]) for c in merged_cols]

    if not col_bounds:
        col_bounds = [(0, 9999)]

    # 3. Grid Snapping
    num_rows = len(row_chars_list)
    num_cols = len(col_bounds)
    table_grid: list[list[list[dict]]] = [[[] for _ in range(num_cols)] for _ in range(num_rows)]

    for r_idx, r_chars in enumerate(row_chars_list):
        for c in r_chars:
            cx = (c["x0"] + c["x1"]) / 2
            best_c_idx = 0
            min_dist = float("inf")
            for c_idx, (start, end) in enumerate(col_bounds):
                if start <= cx <= end:
                    best_c_idx = c_idx
                    break
                mid = (start + end) / 2
                dist = abs(cx - mid)
                if dist < min_dist:
                    min_dist = dist
                    best_c_idx = c_idx
            table_grid[r_idx][best_c_idx].append(c)

    # 4. Output Generation
    final_table = []
    for r_idx in range(num_rows):
        row_str = []
        for c_idx in range(num_cols):
            cell_chars = table_grid[r_idx][c_idx]
            cell_chars.sort(key=lambda c: c["x0"])
            row_str.append(chars_to_text(cell_chars))
        final_table.append(row_str)

    return final_table


def detect_table_lines_hough(img_bgr, page_h: int, page_w: int) -> list[tuple[float, float]] | None:
    """Detect vertical table lines using Hough transform.

    Returns column boundary intervals or None if too few lines found.
    """
    import cv2

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_line_len = int(page_h * 0.15)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=3.14159 / 180,
        threshold=80,
        minLineLength=min_line_len,
        maxLineGap=10,
    )
    if lines is None:
        return None

    vertical_x = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 5:
            vertical_x.append((x1 + x2) / 2)

    if len(vertical_x) < 2:
        return None

    vertical_x.sort()
    clusters = [vertical_x[0]]
    for x in vertical_x[1:]:
        if x - clusters[-1] > 10:
            clusters.append(x)
        else:
            clusters[-1] = (clusters[-1] + x) / 2

    if len(clusters) < 2:
        return None

    col_bounds = []
    for i in range(len(clusters) - 1):
        col_bounds.append((clusters[i], clusters[i + 1]))

    col_bounds = [(a, b) for a, b in col_bounds if b - a > 20]

    return col_bounds if len(col_bounds) >= 2 else None


def detect_has_table(img, page_h: int) -> bool:
    """Check whether the page image has genuine table line structure."""
    col_bounds = detect_table_lines_hough(img, page_h, img.shape[1] if img is not None else 0)
    if not col_bounds or len(col_bounds) < 3:
        return False

    widths = sorted(b - a for a, b in col_bounds)
    median_w = widths[len(widths) // 2]
    max_w = widths[-1]
    if median_w > 0 and max_w / median_w > 5:
        return False

    return True


def group_words_into_lines(words: list[tuple], y_tolerance: float = 12.0) -> list[dict]:
    """Group OCR words into text lines by y-proximity.

    Returns list of line dicts: {"text": str, "bbox": (x0, y0, x1, y1)}
    """
    if not words:
        return []

    sorted_w = sorted(words, key=lambda w: (w[1], w[0]))

    lines: list[dict] = []
    cur_words = [sorted_w[0]]
    cur_y = sorted_w[0][1]

    for w in sorted_w[1:]:
        if abs(w[1] - cur_y) <= y_tolerance:
            cur_words.append(w)
        else:
            cur_words.sort(key=lambda ww: ww[0])
            text = " ".join(ww[4] for ww in cur_words)
            x0 = min(ww[0] for ww in cur_words)
            y0 = min(ww[1] for ww in cur_words)
            x1 = max(ww[2] for ww in cur_words)
            y1 = max(ww[3] for ww in cur_words)
            lines.append({"text": text, "bbox": (x0, y0, x1, y1)})
            cur_words = [w]
            cur_y = w[1]

    if cur_words:
        cur_words.sort(key=lambda ww: ww[0])
        text = " ".join(ww[4] for ww in cur_words)
        x0 = min(ww[0] for ww in cur_words)
        y0 = min(ww[1] for ww in cur_words)
        x1 = max(ww[2] for ww in cur_words)
        y1 = max(ww[3] for ww in cur_words)
        lines.append({"text": text, "bbox": (x0, y0, x1, y1)})

    return lines
