# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
table_merger — Cross-page table merging
========================================

Independent module extracted from ``CoreExtractor._merge_cross_page_tables``.
Responsible for detecting content-continuous cross-page tables and merging
them into a single Block.

Merge strategy:
  1. Next page's first row is a header (matching the previous page's header)
     → skip the duplicate header and merge data rows directly.
  2. First row is not a header (continuation page) → ``_strip_preamble``
     strips summary rows / duplicate headers before merging.
  3. Completely different table (header mismatch) → treat as an independent table.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from ...models.entities.domain import Block, PageLayout
from ..utils.text_utils import headers_match
from ..utils.vocabulary import _is_header_row
from .postprocess import _strip_preamble

logger = logging.getLogger(__name__)


def _median_col_count(rows: list) -> int:
    """P3-6: Compute the median column count of a table's rows."""
    if not rows:
        return 0
    counts = sorted(len(r) for r in rows if isinstance(r, (list, tuple)))
    if not counts:
        return 0
    return counts[len(counts) // 2]


def merge_cross_page_tables(pages: list[PageLayout]) -> list[PageLayout]:
    """Cross-page table merging — operates at the Block level.

    Args:
        pages: List of PageLayout objects for all pages.

    Returns:
        PageLayout list with cross-page tables merged.
    """
    if len(pages) <= 1:
        return pages

    all_blocks = []
    for page in pages:
        for block in page.blocks:
            all_blocks.append(
                {
                    "block": block,
                    "page_number": page.page_number,
                }
            )

    merged_table_data: list[dict] = []
    non_table_blocks: list[dict] = []

    for entry in all_blocks:
        block = entry["block"]
        if block.block_type != "table" or not isinstance(block.raw_content, list):
            non_table_blocks.append(entry)
            continue

        curr_rows = block.raw_content

        if not merged_table_data:
            merged_table_data.append(
                {
                    "rows": list(curr_rows),
                    "pages": [entry["page_number"]],
                    "block": block,
                }
            )
            continue

        prev = merged_table_data[-1]
        prev_rows = prev["rows"]

        first_row = curr_rows[0] if curr_rows else []
        is_header = _is_header_row(first_row)

        # P3-6: column count validation — prevent merging different tables
        prev_col_count = _median_col_count(prev_rows)
        curr_col_count = _median_col_count(curr_rows)
        col_count_mismatch = abs(prev_col_count - curr_col_count) > 1

        if col_count_mismatch:
            # Column count ratio check: ratio < 0.5 = extraction failure (skip, don't break chain)
            max_cc = max(prev_col_count, curr_col_count, 1)
            min_cc = min(prev_col_count, curr_col_count)
            ratio = min_cc / max_cc
            if ratio < 0.5:
                logger.warning(
                    f"[TableMerger] skipped extraction-failed table on page "
                    f"{entry['page_number']} ({curr_col_count} cols vs "
                    f"expected {prev_col_count})"
                )
                continue  # Skip, don't break the merge chain
            # Similar but different column counts → treat as independent table
            merged_table_data.append(
                {
                    "rows": list(curr_rows),
                    "pages": [entry["page_number"]],
                    "block": block,
                }
            )
        elif is_header and prev_rows:
            prev_header = prev_rows[0] if prev_rows else []
            if headers_match(prev_header, first_row):
                prev["rows"].extend(curr_rows[1:])
                prev["pages"].append(entry["page_number"])
            else:
                merged_table_data.append(
                    {
                        "rows": list(curr_rows),
                        "pages": [entry["page_number"]],
                        "block": block,
                    }
                )
        elif not is_header and prev_rows:
            # Continuation page: strip summary / duplicate header rows, then merge
            confirmed_hdr = prev_rows[0] if prev_rows else []
            stripped = _strip_preamble(list(curr_rows), confirmed_hdr)
            stripped = [r for r in stripped if any((c or "").strip() for c in r)]
            if stripped:
                prev["rows"].extend(stripped)
                prev["pages"].append(entry["page_number"])
        else:
            merged_table_data.append(
                {
                    "rows": list(curr_rows),
                    "pages": [entry["page_number"]],
                    "block": block,
                }
            )

    # F-7: post-merge row count audit
    for mdata in merged_table_data:
        if len(mdata["pages"]) > 1:
            merged_rows = len(mdata["rows"])
            logger.info(
                f"[TableMerger] F-7 audit: merged {len(mdata['pages'])} pages → "
                f"{merged_rows} rows (table starts page {mdata['pages'][0]})"
            )

    new_pages = []
    for page in pages:
        page_blocks: list[Block] = []
        for entry in non_table_blocks:
            if entry["page_number"] == page.page_number:
                page_blocks.append(entry["block"])

        for mdata in merged_table_data:
            if mdata["pages"][0] == page.page_number:
                original = mdata["block"]
                merged_block = Block(
                    block_id=original.block_id,
                    block_type="table",
                    bbox=original.bbox,
                    reading_order=original.reading_order,
                    page=original.page,
                    raw_content=mdata["rows"],
                )
                page_blocks.append(merged_block)

        page_blocks.sort(key=lambda b: b.reading_order)

        new_page = PageLayout(
            page_number=page.page_number,
            width=page.width,
            height=page.height,
            blocks=tuple(page_blocks),
            semantic_zones=page.semantic_zones,
            is_scanned=page.is_scanned,
        )
        new_pages.append(new_page)

    return new_pages
