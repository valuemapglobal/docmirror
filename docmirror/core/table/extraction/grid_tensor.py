# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Global Invisible Grid Tensor
=============================
Builds a document-level alignment matrix (1D projection) based on invariant
structural anchors (like aligned decimal points in numerical columns).
This globally protects column boundaries from drifting in borderless, sparse tables
where a single page might lack sufficient data to form robust local peaks.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


def build_global_grid_tensor(all_pages_chars: list[list[dict]], page_w: float, resolution: float = 1.0) -> list[int]:
    """
    Scans all characters across the document to build a global alignment tensor.

    Anchors:
      - Decimal points '.' surrounded by digits on the same baseline.
      - These indicate numerical columns (amounts, dates, coordinates).
      - In random text, their X-positions are dispersed (noise).
      - In tables, their X-positions stack perfectly (signal).

    Args:
        all_pages_chars: List of character lists for each page.
        page_w: Maximum page width.
        resolution: Points per bin (must match signal_processor resolution).

    Returns:
        1D list of amplified densities (G_x tensor).
    """
    w_bins = int(page_w / resolution) + 2
    G_x = [0] * w_bins

    x_dots = []

    # ── Step 1: Fast O(N) scan for decimal anchors ──
    for chars in all_pages_chars:
        # Avoid empty pages
        if not chars or len(chars) < 10:
            continue

        for i, c in enumerate(chars):
            if c.get("text") == ".":
                if 0 < i < len(chars) - 1:
                    prev_c = chars[i - 1]
                    next_c = chars[i + 1]

                    # Validate: Surrounded by digits and vertically aligned
                    if prev_c.get("text", "").isdigit() and next_c.get("text", "").isdigit():
                        y_diff = abs(prev_c.get("top", 0) - c.get("top", 0))
                        if y_diff < 3.0:  # Same line
                            # Use exact X coordinate of the decimal point
                            x_pos = (c.get("x0", 0) + c.get("x1", c.get("x0", 0))) / 2.0
                            x_dots.append(x_pos)

    # ── Step 2: Build Tensor with Kernel Smoothing ──
    # Apply a triangle kernel [5, 10, 5] to enforce a strong protected peak
    # around the decimal point.
    for x in x_dots:
        bin_idx = int(x / resolution)
        if 0 <= bin_idx < w_bins:
            G_x[bin_idx] += 10
            if bin_idx > 0:
                G_x[bin_idx - 1] += 5
            if bin_idx < w_bins - 1:
                G_x[bin_idx + 1] += 5

    # Scale down global tensor slightly relative to total pages to avoid
    # overshadowing strong local signals on massive documents, but keep it
    # strong enough to act as an anchor.
    # We cap the maximum height of the tensor to simulate a "normalized"
    # prior distribution.
    max_peak = max(G_x) if G_x else 0
    if max_peak > 100:
        scale_factor = 100.0 / max_peak
        G_x = [int(v * scale_factor) for v in G_x]

    logger.debug(f"[GridTensor] mapped {len(x_dots)} decimal anchors globally. max_peak={max_peak}")

    return G_x
