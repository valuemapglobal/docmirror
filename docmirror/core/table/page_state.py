# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
PageState — Cross-page table extraction state machine.
========================================================

Carries confirmed extraction metadata across pages:
  - ``winning_layer``: the extraction layer that succeeded on page N.
  - ``confirmed_header``: the validated header row.
  - ``col_count``: column count of the current table.
  - ``confidence``: extraction confidence from the winning layer.

Used by ``extract_tables_layered()`` (via ``layer_hint``) and
``post_process_table()`` (via ``confirmed_header``) to skip
redundant exploration on subsequent pages.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PageState:
    """Cross-page table extraction state.

    Created once per document, updated after each page's successful
    table extraction, and passed forward to the next page.
    """

    confirmed_header: list[str] | None = None
    winning_layer: str | None = None
    col_count: int = 0
    page_count: int = 0
    confidence: float = 0.0

    def should_use_hint(self) -> bool:
        """Return True if enough context has been accumulated to
        provide a meaningful layer hint to the next page.

        Requires:
          - At least one page already processed.
          - A winning layer has been identified.
          - Confidence is above a minimum threshold (0.3).
        """
        return self.page_count > 0 and self.winning_layer is not None and self.confidence >= 0.3

    def update(
        self,
        header: list[str] | None,
        layer: str | None,
        confidence: float,
    ) -> None:
        """Update state after a successful page extraction.

        If the new header differs significantly from the existing one
        (different column count or <40% overlap), the state is reset
        rather than updated — this handles documents with multiple
        distinct tables.
        """
        if header is None or layer is None:
            return

        new_col_count = len(header)

        # Check if this is the same table or a new one
        if self.confirmed_header is not None:
            old_set = {c.strip() for c in self.confirmed_header if c and c.strip()}
            new_set = {c.strip() for c in header if c and c.strip()}
            col_count_ok = abs(self.col_count - new_col_count) <= 1
            overlap = len(old_set & new_set) / max(len(old_set), 1) if old_set else 0

            if not col_count_ok or overlap < 0.4:
                # Different table detected → reset state
                logger.debug(
                    f"[PageState] Table change detected "
                    f"(cols {self.col_count}→{new_col_count}, "
                    f"overlap={overlap:.2f}). Resetting state."
                )
                self.confirmed_header = header
                self.winning_layer = layer
                self.col_count = new_col_count
                self.confidence = confidence
                self.page_count = 1
                return

        self.confirmed_header = header
        self.winning_layer = layer
        self.col_count = new_col_count
        self.confidence = confidence
        self.page_count += 1

    def reset(self) -> None:
        """Full reset for a new document."""
        self.confirmed_header = None
        self.winning_layer = None
        self.col_count = 0
        self.page_count = 0
        self.confidence = 0.0
