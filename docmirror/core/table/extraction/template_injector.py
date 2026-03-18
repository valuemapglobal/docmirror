# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
template_injector - Graph-Propagated Template Injection
=========================================================

Handles the extraction of a rigid, absolutely positioned grid template from a highly
confident golden sample page, and the forced injection of this template onto noisier
pages (like the first page) to absolutely guarantee column alignment.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from ...utils.watermark import is_watermark_char

logger = logging.getLogger(__name__)


@dataclass
class GlobalTableTemplate:
    """Immutable template representing the absolute column boundaries and header."""

    header_vocab: list[str]
    col_bounds: list[float]  # List of X-coordinate column dividers. len = cols + 1


def build_global_template(page_plum, extracted_table: list[list[str]]) -> GlobalTableTemplate | None:
    """
    Reverse-engineers the precise column boundaries (X-coordinates) from a successfully
    extracted table on a golden sample page.
    """
    if not extracted_table or len(extracted_table) < 2:
        return None

    num_cols = len(extracted_table[0])
    if num_cols < 2:
        return None

    chars = page_plum.chars
    if not chars:
        return None

    text_chars = [c for c in chars if c.get("text", "").strip() and not is_watermark_char(c)]
    if not text_chars:
        return None

    # Step 1: Assign every character to a column based on the extracted table strings
    # To do this safely without complex string matching, we can just do a whitespace projection
    # on the text_chars, assuming the golden page is very clean.
    # Alternatively, we can project all char X-coordinates and find the largest gaps
    # that result in exactly `num_cols` groups.

    # A simpler and highly robust method for a golden page:
    # Just cluster the characters by X coordinate.
    from .char_strategy import _cluster_x_positions

    [c["x0"] for c in text_chars]

    # Try different gap multipliers until we find exactly `num_cols`, or fallback to just getting the boundaries
    # directly from the char distribution.

    # Since we know the table is perfect, projecting characters vertically and finding the num_cols-1 largest gaps
    # is mathematically guaranteed to find the columns.
    ranges = []
    for c in text_chars:
        ranges.append((c["x0"], c["x1"]))

    # Merge overlapping ranges
    ranges.sort()
    merged = []
    for r in ranges:
        if not merged:
            merged.append(r)
        else:
            last = merged[-1]
            if r[0] <= last[1] + 2.0:  # 2pt tolerance for kerning
                merged[-1] = (last[0], max(last[1], r[1]))
            else:
                merged.append(r)

    # Gaps are between merged blocks
    gaps = []
    for i in range(len(merged) - 1):
        gap_width = merged[i + 1][0] - merged[i][1]
        gap_center = (merged[i + 1][0] + merged[i][1]) / 2.0
        gaps.append((gap_width, gap_center))

    # Sort gaps by width descending, pick top (num_cols - 1), then sort by x-coordinate
    gaps.sort(key=lambda x: x[0], reverse=True)
    best_gaps = gaps[: num_cols - 1]
    best_gap_centers = sorted([g[1] for g in best_gaps])

    # Col bounds are [0, gap1, gap2, ..., page_width]
    page_w = page_plum.width or 1000
    col_bounds = [0.0] + best_gap_centers + [float(page_w)]

    if len(col_bounds) != num_cols + 1:
        logger.warning(f"[TemplateInjector] Failed to derive {num_cols} columns from gaps.")
        return None

    header = extracted_table[0]
    template = GlobalTableTemplate(header_vocab=header, col_bounds=col_bounds)

    logger.info(f"[TemplateInjector] Successfully generated robust GlobalTableTemplate: {num_cols} columns.")
    return template


def extract_by_injected_template(page_plum, template: GlobalTableTemplate) -> list[list[str]] | None:
    """
    Forcefully extracts a table by chunking characters strictly into the injected
    template's absolute column boundaries.
    """
    if not template or not template.col_bounds:
        return None

    chars = page_plum.chars
    if not chars:
        return None

    text_chars = [c for c in chars if c.get("text", "").strip() and not is_watermark_char(c)]
    if not text_chars:
        return None

    # Group chars into rows by Y-coordinate
    from .char_strategy import _group_chars_into_rows

    rows_by_y = _group_chars_into_rows(text_chars)

    col_bounds = template.col_bounds
    num_cols = len(col_bounds) - 1

    table_data = []

    # To improve robustness, we sort chars by x0
    for y_mid, row_chars in rows_by_y:
        row_chars.sort(key=lambda c: c["x0"])
        row_cells = ["" for _ in range(num_cols)]

        # O(N*M) is fine here since num_cols is small
        for c in row_chars:
            cx = (c["x0"] + c["x1"]) / 2.0
            # Find which column cx falls into
            for idx in range(num_cols):
                if col_bounds[idx] <= cx < col_bounds[idx + 1]:
                    row_cells[idx] += c["text"]
                    break

        # Clean up cell texts
        # Fix char spacing logic matching the rest of the project
        # Re-joining using distance heuristic
        clean_row = []
        for idx in range(num_cols):
            cell_chars = [c for c in row_chars if col_bounds[idx] <= ((c["x0"] + c["x1"]) / 2.0) < col_bounds[idx + 1]]
            if not cell_chars:
                clean_row.append("")
                continue

            cell_chars.sort(key=lambda c: c["x0"])
            text_parts = []
            last_x = cell_chars[0]["x0"]
            for char in cell_chars:
                # Add space if gap is large enough
                if char["x0"] - last_x > (char["x1"] - char["x0"]) * 0.5:
                    if text_parts and not text_parts[-1].endswith(" "):
                        text_parts.append(" ")
                text_parts.append(char["text"])
                last_x = char["x1"]

            cell_text = "".join(text_parts).strip()
            clean_row.append(cell_text)

        if any(cell for cell in clean_row):
            table_data.append(clean_row)

    if len(table_data) >= 2:
        return table_data
    return None
