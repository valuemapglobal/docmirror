"""
Pipe-delimited table extraction — Layer 0.5: Grid Consistency algorithm.

Split from ``table_extraction.py``.

Designed for mainframe-generated ASCII-art PDFs:
  - Vertical separators: ``|``, ``│`` and other ``PIPE_CHARS``
  - Horizontal separators: ``─``, ``━`` and other ``HLINE_CHARS``
  - PDF contains no graphical primitives (lines / rects = 0)
"""
from __future__ import annotations


import logging
from collections import defaultdict
from typing import Dict, List, Optional

from ...utils.text_utils import _is_cjk_char, _smart_join
from ...utils.vocabulary import PIPE_CHARS, _ALL_BORDER_CHARS, _is_header_row, _normalize_for_vocab, _score_header_by_vocabulary, _RE_IS_DATE, _RE_IS_AMOUNT

logger = logging.getLogger(__name__)

def _extract_by_pipe_delimited(
    page_plum,
) -> Optional[List[List[str]]]:
    """Pipe-delimited table extraction (Layer 0.5) — Grid Consistency algorithm.

    Specifically designed for mainframe-generated ASCII-art PDFs:
      - Vertical separators: pipe characters (``|``, ``│``, etc.)
      - Horizontal separators: horizontal line characters (``─``, ``━``, etc.)
      - PDF contains zero graphical primitives (lines / rects = 0)

    Safety gates:
      G1: ``pdfplumber.lines == 0`` and ``rects == 0``
          (only enabled when there are no drawing primitives).
      G2: >= 3 consistent vertical grid lines (present in >= 70 % of data rows).
      G3: >= 3 data rows (after excluding horizontal separator rows).
      G4: x-coordinate standard deviation of each grid line <= 3 pt.
    """
    # ── G1: only enabled when no PDF drawing primitives ──
    pdf_lines = page_plum.lines or []
    pdf_rects = page_plum.rects or []
    if pdf_lines or pdf_rects:
        return None

    chars = page_plum.chars
    if not chars:
        return None

    # ── Step 1: group characters into rows by y-coordinate ──
    y_groups: Dict[int, List[dict]] = defaultdict(list)
    for c in chars:
        y_key = round(c["top"] / 3) * 3
        y_groups[y_key].append(c)

    if len(y_groups) < 3:
        return None

    # ── Step 2: classify rows — data rows vs horizontal separator rows ──
    data_rows_ys: List[int] = []      # y_key values of pipe-containing data rows
    hline_rows_ys: List[int] = []     # y_key values of pure horizontal-line rows
    all_pipe_x_by_row: Dict[int, List[float]] = {}  # y_key → list of pipe x-coordinates

    for y_key in sorted(y_groups.keys()):
        row_chars = y_groups[y_key]
        row_text = "".join(c["text"] for c in sorted(row_chars, key=lambda c: c["x0"]))

        # Detect horizontal separator rows: majority of chars are border characters
        non_space = [c for c in row_text if c.strip()]
        if non_space:
            border_ratio = sum(1 for c in non_space if c in _ALL_BORDER_CHARS) / len(non_space)
            if border_ratio >= 0.8:
                hline_rows_ys.append(y_key)
                continue

        # Collect pipe character x-coordinates
        pipe_xs = [
            round(c["x0"], 1)
            for c in row_chars
            if c.get("text") in PIPE_CHARS
        ]
        if len(pipe_xs) >= 2:
            data_rows_ys.append(y_key)
            all_pipe_x_by_row[y_key] = sorted(pipe_xs)

    # ── G3: at least 3 data rows ──
    if len(data_rows_ys) < 3:
        return None

    # ── Step 3: cluster pipe x-coordinates to find vertical grid lines ──
    # Collect all pipe x-coordinates
    all_pipe_xs: List[float] = []
    for xs in all_pipe_x_by_row.values():
        all_pipe_xs.extend(xs)

    if not all_pipe_xs:
        return None

    # Cluster: snap to a 5-pt grid
    SNAP = 5.0
    x_clusters: Dict[float, List[float]] = defaultdict(list)
    for x in sorted(all_pipe_xs):
        snapped = round(x / SNAP) * SNAP
        x_clusters[snapped].append(x)

    # Merge nearby clusters (distance < 8 pt)
    sorted_centers = sorted(x_clusters.keys())
    merged_clusters: List[List[float]] = []
    for center in sorted_centers:
        if merged_clusters and center - sum(merged_clusters[-1]) / len(merged_clusters[-1]) < 8:
            merged_clusters[-1].extend(x_clusters[center])
        else:
            merged_clusters.append(list(x_clusters[center]))

    # ── G2 + G4: check grid consistency ──
    n_data_rows = len(data_rows_ys)
    consistent_grid_lines: List[float] = []  # Grid line x-centres that pass consistency check

    for cluster in merged_clusters:
        # Presence ratio: fraction of data rows containing this x-cluster
        rows_with_this_x = set()
        for y_key, pipe_xs in all_pipe_x_by_row.items():
            if any(abs(px - sum(cluster) / len(cluster)) < 8 for px in pipe_xs):
                rows_with_this_x.add(y_key)

        presence_ratio = len(rows_with_this_x) / n_data_rows
        if presence_ratio < 0.7:
            continue

        # G4: x-coordinate standard deviation
        mean_x = sum(cluster) / len(cluster)
        variance = sum((x - mean_x) ** 2 for x in cluster) / len(cluster)
        std_x = variance ** 0.5
        if std_x > 3.0:
            continue

        consistent_grid_lines.append(mean_x)

    # G2: at least 3 consistent vertical grid lines
    if len(consistent_grid_lines) < 3:
        return None

    consistent_grid_lines.sort()
    n_cols = len(consistent_grid_lines) - 1  # Number of columns between pipes
    if n_cols < 2:
        return None

    logger.info(
        f"pipe_delimited: detected {len(consistent_grid_lines)} grid lines, "
        f"{n_cols} cols, {n_data_rows} data rows, "
        f"{len(hline_rows_ys)} hline rows"
    )

    # ── Step 4: split characters into columns using grid lines ──
    # Column intervals: (left_pipe_x, right_pipe_x)
    col_intervals = [
        (consistent_grid_lines[i], consistent_grid_lines[i + 1])
        for i in range(n_cols)
    ]

    table: List[List[str]] = []
    for y_key in sorted(data_rows_ys):
        row_chars = sorted(y_groups[y_key], key=lambda c: c["x0"])
        # Filter out the pipe characters themselves
        content_chars = [c for c in row_chars if c.get("text") not in PIPE_CHARS]

        cells = [""] * n_cols
        for c in content_chars:
            cx = c["x0"]
            # Find the containing column
            assigned = False
            for col_idx, (left, right) in enumerate(col_intervals):
                if left - 3 <= cx < right + 3:
                    cells[col_idx] += c["text"]
                    assigned = True
                    break
            if not assigned:
                # Outside the grid: assign to the nearest column
                distances = [abs(cx - (l + r) / 2) for l, r in col_intervals]
                nearest = distances.index(min(distances))
                cells[nearest] += c["text"]

        table.append([cell.strip() for cell in cells])

    if len(table) < 3:
        return None

    # ── Step 5: merge continuation rows (mainframe records may span multiple lines) ──
    table = _merge_pipe_continuation_rows(table)

    logger.info(
        f"pipe_delimited: extracted {len(table)} rows x {n_cols} cols"
    )
    return table


def _merge_pipe_continuation_rows(table: List[List[str]]) -> List[List[str]]:
    """Merge continuation rows in a pipe-delimited table.

    In mainframe output, a single record may be split across multiple lines:
      Row N:   | 1  |251209|251209|...transfer...|   894.34|         |   9,143.21|...
      Row N+1: |    |      |      |              |         |         |           |...

    Rule: if the first column (sequence number) is empty, the row is treated
    as a continuation of the previous row and its content is appended.
    """
    if not table or len(table) < 2:
        return table

    merged: List[List[str]] = [table[0]]
    for row in table[1:]:
        first_cell = row[0].strip() if row else ""
        # Continuation row: first column (sequence number) is empty with other content
        has_content = any(c.strip() for c in row[1:])
        if not first_cell and has_content and merged:
            # Append to the previous row
            prev = merged[-1]
            for i in range(len(row)):
                if i < len(prev):
                    cell_text = row[i].strip()
                    if cell_text:
                        if prev[i].strip():
                            prev[i] = prev[i].strip() + cell_text
                        else:
                            prev[i] = cell_text
        else:
            merged.append(row)

    return merged
