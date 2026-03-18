# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Table Postprocessing Utilities
"""

from __future__ import annotations

import logging
from typing import List

from docmirror.core.layout.layout_analysis import post_process_table
from docmirror.models.entities.domain import Block, PageLayout

logger = logging.getLogger(__name__)


def process_page_tables(pages: list[PageLayout]) -> list[PageLayout]:
    """Table post-processing -- header detection + preamble KV extraction + data row cleaning.

    For each table Block, execute the following steps:
      1. Call post_process_table:
         - VOCAB_BY_CATEGORY scans first 10 rows, finds the row with highest vocab_score as header
         - When header_row_idx > 0: call _extract_preamble_kv for summary rows before header
           Results stored as preamble KV (e.g. total amount/count/start date)
         - Call _strip_preamble on data_rows to strip mixed summary rows and duplicate headers
         - Per-row cleanup: filter junk rows, append non-data rows to previous, align column count
      2. Call get_and_clear_preamble_kv() to retrieve KV cache:
         - If non-empty, create key_value Block inserted before the table Block
         - Block ID is "{block_id}_kv", reading_order same as table

    Side effect: if table Block has < 2 rows after processing, retain original Block unchanged.
    """
    new_pages = []
    for page in pages:
        new_blocks = []
        for block in page.blocks:
            if block.block_type == "table" and isinstance(block.raw_content, list):
                # Skip entirely empty tables
                if not any((cell or "").strip() for row in block.raw_content for cell in row):
                    logger.debug("[DocMirror] skipped empty table")
                    continue
                try:
                    processed, preamble_kv = post_process_table(block.raw_content)

                    # NOTE: fix_table_structure (column removal, line merging) is
                    # intentionally NOT called here. It runs AFTER cross-page merge
                    # in the extractor pipeline to avoid column count mismatches
                    # that would prevent correct cross-page table merging.

                    # Preamble KV obtained directly from return value, no global state
                    if preamble_kv:
                        kv_block = Block(
                            block_id=f"{block.block_id}_kv",
                            block_type="key_value",
                            bbox=block.bbox,
                            reading_order=block.reading_order,
                            page=block.page,
                            raw_content=preamble_kv,
                        )
                        new_blocks.append(kv_block)
                    if processed and len(processed) >= 2:
                        new_block = Block(
                            block_id=block.block_id,
                            block_type="table",
                            bbox=block.bbox,
                            reading_order=block.reading_order,
                            page=block.page,
                            raw_content=processed,
                        )
                        new_blocks.append(new_block)
                    else:
                        new_blocks.append(block)
                except Exception as e:
                    logger.debug(f"[DocMirror] post_process error: {e}")
                    new_blocks.append(block)
            else:
                new_blocks.append(block)

        new_page = PageLayout(
            page_number=page.page_number,
            width=page.width,
            height=page.height,
            blocks=tuple(new_blocks),
            semantic_zones=page.semantic_zones,
            is_scanned=page.is_scanned,
        )
        new_pages.append(new_page)

    return new_pages
