# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Header-Data Alignment Verification
====================================

Handles content-type-based inference
and structural alignment verification.

Core Capabilities:
  - ``infer_column_type()``: Samples data columns to infer expected datatype distributions (date/amount/seq/text).
  - ``verify_header_data_alignment()``: Detects and automatically corrects systematic alignment offsets occurring between headers and data rows.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List

logger = logging.getLogger(__name__)

# Date Regex Patterns (Lenient, capturing YYYYMMDD / YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD etc.)
_RE_COL_DATE = re.compile(
    r"^\d{8}(\s*\d{1,2}:\d{2}(:\d{2})?)?$|"
    r"^\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?"
    r"(\s*\d{1,2}:\d{2}(:\d{2})?)?$|"
    r"^\d{2}[-/]\d{2}[-/]\d{4}$"
)
# Numeric Amount Regex (Supports comma-separated inputs)
_RE_COL_AMOUNT = re.compile(r"^[+-]?\d[\d,]*\.\d{1,4}$")
# Sequence/Index Number Regex (Pure integers, 1~8 digits)
_RE_COL_SEQ = re.compile(r"^\d{1,8}$")


def infer_column_type(
    data_rows: list[list[str]],
    col_idx: int,
    sample_size: int = 30,
) -> dict[str, float]:
    """Infers the dominant data type distribution of a target column.

    Returns:
        Probability map, e.g., {"date": 0.9, "amount": 0.05, "seq": 0.0, "text": 0.05}
    """
    counts = {"date": 0, "amount": 0, "seq": 0, "text": 0}
    total = 0
    for row in data_rows[:sample_size]:
        if col_idx >= len(row):
            continue
        val = (row[col_idx] or "").strip()
        if not val:
            continue
        total += 1
        clean = val.replace(",", "").replace("，", "").replace("¥", "")
        if _RE_COL_DATE.match(val):
            counts["date"] += 1
        elif _RE_COL_AMOUNT.match(clean):
            counts["amount"] += 1
        elif _RE_COL_SEQ.match(val):
            counts["seq"] += 1
        else:
            counts["text"] += 1

    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def verify_header_data_alignment(
    headers: list[str],
    data_rows: list[list[str]],
    header_type_expectations: dict[str, str],
    mutation_recorder=None,
    middleware_name: str = "HeaderAlignment",
) -> list[str]:
    """Validates alignment between headers and corresponding data columns,
    detecting and repairing systematic offsets.

    Args:
        headers: Original list of table headers.
        data_rows: Extraction data rows matrix.
        header_type_expectations: Mapping of Header Name \u2192 Expected Datatype ("date"/"amount"/"seq").
        mutation_recorder: Optional `EnhancedResult` dependency for mutation tracking logging.
        middleware_name: Name of the invoking middleware for telemetry.

    Returns:
        A corrected list of table headers, or the original payload if unchanged.
    """
    n_cols = len(headers)
    if len(data_rows) < 5 or n_cols < 3:
        return headers

    # \u2500\u2500\u2500 Step 1: Collect structural anchor columns \u2500\u2500\u2500
    anchors: list[dict] = []
    for i, h in enumerate(headers):
        h_clean = h.strip()
        if not h_clean:
            continue
        expected = header_type_expectations.get(h_clean)
        if not expected:
            for kw, typ in header_type_expectations.items():
                if len(kw) >= 2 and kw in h_clean:
                    expected = typ
                    break
        if expected:
            anchors.append(
                {
                    "header_idx": i,
                    "expected_type": expected,
                    "header_name": h_clean,
                }
            )

    if len(anchors) < 2:
        return headers

    # \u2500\u2500\u2500 Step 2: Validate column-level alignment against anchors \u2500\u2500\u2500
    offsets: list[dict] = []
    for anchor in anchors:
        hi = anchor["header_idx"]
        et = anchor["expected_type"]
        current_profile = infer_column_type(data_rows, hi)
        current_match = current_profile.get(et, 0.0)

        # Early exit matching threshold
        if current_match >= 0.5:
            offsets.append({"anchor": anchor, "offset": 0, "confidence": current_match})
            continue

        best_offset = 0
        best_match = current_match
        # Sweep surrounding columns (±2 window) resolving potential OCR shift artifacts
        for delta in [1, -1, 2, -2]:
            check_idx = hi + delta
            if 0 <= check_idx < n_cols:
                profile = infer_column_type(data_rows, check_idx)
                match = profile.get(et, 0.0)
                if match > best_match:
                    best_match = match
                    best_offset = delta

        # Register shifted positive hits
        if best_match >= 0.5 and best_offset != 0:
            offsets.append({"anchor": anchor, "offset": best_offset, "confidence": best_match})
        else:
            offsets.append({"anchor": anchor, "offset": 0, "confidence": current_match})

    # \u2500\u2500\u2500 Step 3: Global systemic pattern evaluation \u2500\u2500\u2500
    non_zero = [o for o in offsets if o["offset"] != 0]
    if len(non_zero) < 2:
        return headers

    offset_counts = Counter(o["offset"] for o in non_zero)
    dominant_offset, dominant_count = offset_counts.most_common(1)[0]
    if dominant_count < 2:
        return headers

    zero_count = sum(1 for o in offsets if o["offset"] == 0)
    # Require systematic skew to overpower static (valid) alignment observations
    if zero_count > dominant_count:
        return headers

    # \u2500\u2500\u2500 Step 4: Execute Alignment Correction \u2500\u2500\u2500
    logger.info(
        f"[{middleware_name}] alignment fix: detected systematic offset={dominant_offset}, "
        f"anchors={[(o['anchor']['header_name'], o['offset']) for o in offsets]}"
    )

    new_headers = list(headers)
    if dominant_offset > 0:
        # Padded insertion shifting headers rightward natively
        for _ in range(abs(dominant_offset)):
            first_shift_idx = min(o["anchor"]["header_idx"] for o in non_zero if o["offset"] == dominant_offset)
            new_headers.insert(first_shift_idx, "")
        new_headers = new_headers[:n_cols]
    elif dominant_offset < 0:
        # Deletion strategy shifting headers incrementally structural leftward
        for _ in range(abs(dominant_offset)):
            first_shift_idx = min(o["anchor"]["header_idx"] for o in non_zero if o["offset"] == dominant_offset)
            remove_idx = None
            for ri in range(first_shift_idx, -1, -1):
                if not new_headers[ri].strip():
                    remove_idx = ri
                    break
            if remove_idx is not None:
                new_headers.pop(remove_idx)
                new_headers.append("")

    if mutation_recorder:
        mutation_recorder.record_mutation(
            middleware_name=middleware_name,
            target_block_id="document",
            field_changed="header_alignment",
            old_value=str(headers[:5]),
            new_value=str(new_headers[:5]),
            confidence=0.85,
            reason=f"systematic offset={dominant_offset}, {dominant_count} anchors confirmed",
        )

    logger.info(f"[{middleware_name}] alignment fix applied: {headers[:6]} \u2192 {new_headers[:6]}")
    return new_headers
