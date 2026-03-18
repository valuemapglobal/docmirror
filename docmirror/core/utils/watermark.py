# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Preprocessing & Watermark Filter
=================================

PDF preprocessing helpers extracted from layout_analysis.py.

Contents:
    - ``preprocess_document`` — Layer-0 physical clean-up: strips annotation
      layers using pikepdf.
    - ``is_watermark_char`` — triple-check heuristic (rotation / matrix /
      colour) to identify watermark characters in pdfplumber output.
    - ``filter_watermark_page`` — applies the watermark filter to a
      pdfplumber page.
    - ``_dedup_overlapping_chars`` — removes pseudo-bold duplicate
      characters (same text at nearly the same position).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def preprocess_document(file_path: Path) -> Path:
    """Layer-0 physical clean-up: use pikepdf to strip annotation layers.

    If annotations are found and removed, the cleaned PDF is saved to a
    new file (``<name>_cleaned.pdf``) and its path is returned.  Otherwise
    the original path is returned unchanged.
    """
    try:
        import pikepdf

        pdf = pikepdf.open(str(file_path))
        modified = False
        for page in pdf.pages:
            if "/Annots" in page:
                del page["/Annots"]
                modified = True
        if modified:
            temp_path = file_path.parent / f"{file_path.stem}_cleaned.pdf"
            pdf.save(str(temp_path))
            logger.info(f"preprocess: removed annotations → {temp_path.name}")
            return temp_path
    except Exception as e:
        logger.debug(f"preprocess: pikepdf skip ({e})")
    return file_path


def is_watermark_char(obj: dict) -> bool:
    """Determine whether a pdfplumber character object is a watermark.

    Uses a triple-check heuristic:
      1. **Rotation**: ``upright`` flag is ``False``.
      2. **Transformation matrix**: significant skew (``|m[1]| > 0.1`` or
         ``|m[2]| > 0.1``).
      3. **Colour**: non-stroking colour components are all > 0.5
         (very light / washed-out text).
    """
    if not obj.get("upright", True):
        return True
    m = obj.get("matrix")
    if m and (abs(m[1]) > 0.1 or abs(m[2]) > 0.1):
        return True
    nsc = obj.get("non_stroking_color")
    if isinstance(nsc, (list, tuple)) and all(c > 0.5 for c in nsc):
        return True
    return False


def filter_watermark_page(page):
    """Return a filtered pdfplumber page with watermark characters removed."""
    return page.filter(lambda obj: obj.get("object_type") != "char" or not is_watermark_char(obj))


def separate_watermark_layer(page) -> object:
    """Deep watermark/content separation using statistical pattern detection.

    Goes beyond ``filter_watermark_page`` by analysing the *distribution*
    of character rotation angles and opacity values across the entire page.
    Watermarks typically share a single consistent rotation and light colour
    that differ from the body text cluster.

    Algorithm:
        1. Cluster characters by quantised rotation angle (from the
           transformation matrix).
        2. The dominant rotation cluster (largest group) is treated as
           content; minority rotation clusters as watermark candidates.
        3. Within the content cluster, characters whose non-stroking colour
           components are all > 0.6 (very light) are also reclassified
           as watermark.
        4. Return the filtered page with watermark characters removed.

    Falls back to ``filter_watermark_page`` when statistical separation
    is not conclusive (e.g., all characters share the same rotation).

    Returns:
        Filtered pdfplumber page.
    """
    chars = page.chars
    if not chars or len(chars) < 10:
        return filter_watermark_page(page)

    import math

    # ── Step 1: Cluster by rotation angle ──
    rotation_buckets: dict[int, list] = {}
    for i, c in enumerate(chars):
        m = c.get("matrix")
        if m and len(m) >= 4:
            # atan2(m[1], m[0]) gives the rotation angle in radians
            angle_deg = round(math.degrees(math.atan2(m[1], m[0])))
        else:
            angle_deg = 0
        bucket = rotation_buckets.setdefault(angle_deg, [])
        bucket.append(i)

    if len(rotation_buckets) <= 1:
        # All characters share the same rotation — statistical separation
        # not conclusive, fall back to basic filter.
        return filter_watermark_page(page)

    # ── Step 2: Dominant cluster = content ──
    dominant_angle = max(rotation_buckets, key=lambda a: len(rotation_buckets[a]))
    watermark_indices: set = set()

    for angle, indices in rotation_buckets.items():
        if angle != dominant_angle:
            # Minority rotation cluster → watermark
            watermark_indices.update(indices)

    # ── Step 3: Opacity check within content cluster ──
    for idx in rotation_buckets.get(dominant_angle, []):
        c = chars[idx]
        nsc = c.get("non_stroking_color")
        if isinstance(nsc, (list, tuple)) and len(nsc) >= 3:
            if all(comp > 0.6 for comp in nsc[:3]):
                watermark_indices.add(idx)

    if not watermark_indices:
        return page  # nothing to remove

    pct = len(watermark_indices) / len(chars) * 100
    logger.debug(
        f"[watermark] deep separation: removed {len(watermark_indices)} "
        f"chars ({pct:.1f}%) across {len(rotation_buckets)} rotation clusters"
    )

    keep_set = {id(chars[i]) for i in range(len(chars)) if i not in watermark_indices}
    return page.filter(lambda obj: obj.get("object_type") != "char" or id(obj) in keep_set)


def _dedup_overlapping_chars(page):
    """Remove pseudo-bold duplicate characters.

    Some PDFs render bold text by placing the same character multiple times
    at nearly the same position.  This function detects and removes the
    duplicates using a spatial bucket (3 pt resolution) keyed on
    ``(bucket_x, bucket_y, character_text)``.
    """
    seen = set()
    dedup_ids = set()
    bucket = 3
    for i, c in enumerate(page.chars):
        bx = int(c["x0"] / bucket)
        by = int(c["top"] / bucket)
        key = (bx, by, c["text"])
        if key in seen:
            dedup_ids.add(i)
        else:
            seen.add(key)

    if not dedup_ids:
        return page

    keep_chars = {id(page.chars[i]) for i in range(len(page.chars)) if i not in dedup_ids}
    return page.filter(lambda obj: obj.get("object_type") != "char" or id(obj) in keep_chars)


def fused_filter_and_dedup(page) -> tuple:
    """S1: Single-pass watermark removal + pseudo-bold dedup.

    Combines ``is_watermark_char`` check and spatial-bucket dedup
    into one O(N) scan instead of two separate O(N) passes.

    Returns:
        ``(filtered_page, watermark_found)`` tuple.
    """
    seen = set()
    remove_ids = set()
    bucket = 3

    for i, c in enumerate(page.chars):
        # ── Watermark check (inline from is_watermark_char) ──
        if not c.get("upright", True):
            remove_ids.add(i)
            continue
        m = c.get("matrix")
        if m and (abs(m[1]) > 0.1 or abs(m[2]) > 0.1):
            remove_ids.add(i)
            continue
        nsc = c.get("non_stroking_color")
        if isinstance(nsc, (list, tuple)) and all(v > 0.5 for v in nsc):
            remove_ids.add(i)
            continue

        # ── Dedup check (spatial bucket) ──
        key = (int(c["x0"] / bucket), int(c["top"] / bucket), c["text"])
        if key in seen:
            remove_ids.add(i)
        else:
            seen.add(key)

    watermark_found = bool(remove_ids)
    if not remove_ids:
        return page, False

    keep = {id(page.chars[i]) for i in range(len(page.chars)) if i not in remove_ids}
    filtered = page.filter(lambda obj: obj.get("object_type") != "char" or id(obj) in keep)
    return filtered, watermark_found
