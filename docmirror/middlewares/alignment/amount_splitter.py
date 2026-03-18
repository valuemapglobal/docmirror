# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Debit/Credit Split Column Detection (Amount Split Detector)
===========================================================

Extracted from ``column_mapper.py``: Detects income/expense split-column
patterns in tables.

Supports 3 modes:
  1. Explicit Split: Table headers explicitly contain income/expense keywords.
  2. Implicit Split: Amount column adjacent to an empty header column
     (e.g., Shanghai Pudong Development Bank 'Transaction amount' + Empty column).
  3. Merged Split: Concatenated column names containing embedded debit/credit keywords.

F-6 Enhancements:
  - Data Row Validation: Checks if debit/credit columns both simultaneously contain
    values (>30% conflict ratio \u2192 rejected, implies they are not mutually exclusive splits).
  - Debit/Credit Flag Detection: If a dedicated "Debit/Credit Flag" column exists,
    amount splitting is aggressively skipped.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# F-6: Data validation \u2014 Generic number regex
_RE_HAS_NUMBER = re.compile(r"\d")


def _validate_split_by_data(
    data_rows: list[list[str]],
    inc_idx: int | None,
    exp_idx: int | None,
    max_sample: int = 20,
) -> bool:
    """F-6: Samples data rows to validate if column splitting is logical.

    If >30% of data rows have numeric values in BOTH the supposed income
    and expense columns simultaneously, it indicates this is not a true
    debit/credit split (but rather two independent amount columns).
    """
    if inc_idx is None or exp_idx is None or not data_rows:
        return True  # Unable to validate, default to trusting the headers

    sample = data_rows[:max_sample]
    both_count = 0
    valid_count = 0

    for row in sample:
        if inc_idx >= len(row) or exp_idx >= len(row):
            continue
        inc_val = (row[inc_idx] or "").strip()
        exp_val = (row[exp_idx] or "").strip()
        inc_has = bool(inc_val and _RE_HAS_NUMBER.search(inc_val))
        exp_has = bool(exp_val and _RE_HAS_NUMBER.search(exp_val))

        if inc_has or exp_has:
            valid_count += 1
            if inc_has and exp_has:
                both_count += 1

    if valid_count < 3:
        return True  # Data sample too small, default to trusting headers

    conflict_ratio = both_count / valid_count
    if conflict_ratio > 0.3:
        logger.info(
            f"[AmountSplit] F-6: split rejected by data validation "
            f"(conflict={both_count}/{valid_count}={conflict_ratio:.0%})"
        )
        return False
    return True


def detect_split_amount(
    headers: list[str],
    mapping: dict[str, str | None],
    income_keywords: set[str],
    expense_keywords: set[str],
    amount_like_keywords: set[str],
    data_rows: list[list[str]] | None = None,
) -> tuple[bool, int | None, int | None]:
    """Detects whether an income/expense split column layout exists.

    Args:
        headers: Original list of table headers.
        mapping: Column mapping result {raw_header: standard_name or None}.
        income_keywords: Set of keywords identifying an income column.
        expense_keywords: Set of keywords identifying an expense column.
        amount_like_keywords: Set of generic amount-related keywords.
        data_rows: (F-6) Optional data rows for validating split logic viability.

    Returns:
        A tuple: (has_split_amount, income_col_idx, expense_col_idx)
    """
    # F-6: Detect Debit/Credit Flag Column \u2014 skip split if flag column exists
    _DEBIT_CREDIT_FLAGS = {
        "借贷标志",
        "借贷",
        "借/贷",
        "收支",
        "支/收",
        "借贷Status",
        "收支标志",
        "DC标志",
        "Debit/Credit Flag",
    }
    header_set = {h.strip() for h in headers if h}
    if header_set & _DEBIT_CREDIT_FLAGS:
        logger.info("[AmountSplit] F-6: skipped \u2014 debit/credit flag column found")
        return False, None, None

    has_income = bool(header_set & income_keywords)
    has_expense = bool(header_set & expense_keywords)

    if has_income and has_expense:
        # Mode 1: Explicit split (Headers clearly indicate Income and Expense)
        inc_idx = exp_idx = None
        for i, h in enumerate(headers):
            h_clean = h.strip()
            if h_clean in income_keywords and inc_idx is None:
                inc_idx = i
            elif h_clean in expense_keywords and exp_idx is None:
                exp_idx = i
        # F-6: Data validation phase
        if data_rows and not _validate_split_by_data(data_rows, inc_idx, exp_idx):
            return False, None, None
        return True, inc_idx, exp_idx

    # Mode 2: Amount column + Adjacent empty header column \u2192 Implicit split
    for i, h in enumerate(headers):
        h_clean = h.strip()
        if h_clean in amount_like_keywords and i + 1 < len(headers):
            next_h = headers[i + 1].strip()
            if not next_h:
                # F-6: Data validation phase
                # Assume column `i` is expense and empty column `i+1` is income
                if data_rows and not _validate_split_by_data(data_rows, i + 1, i):
                    continue
                logger.info(
                    f"[AmountSplit] detected implicit split: '{h_clean}'(idx={i})=expense + empty(idx={i + 1})=income"
                )
                return True, i + 1, i

    # Mode 3: Merged split \u2014 concatenations of column names embedding debit/credit keywords
    merged_inc_idx = merged_exp_idx = None
    for i, h in enumerate(headers):
        h_clean = h.strip()
        if not h_clean:
            continue
        for kw in income_keywords:
            if kw in h_clean:
                if h_clean == kw:
                    merged_inc_idx = i
                elif merged_inc_idx is None:
                    merged_inc_idx = i
                break
        for kw in expense_keywords:
            if kw in h_clean:
                if h_clean == kw:
                    merged_exp_idx = i
                elif merged_exp_idx is None:
                    merged_exp_idx = i
                break

    if merged_inc_idx is not None and merged_exp_idx is not None and merged_inc_idx != merged_exp_idx:
        # F-6: Data validation phase
        if data_rows and not _validate_split_by_data(data_rows, merged_inc_idx, merged_exp_idx):
            return False, None, None
        logger.info(
            f"[AmountSplit] detected merged split: "
            f"income(idx={merged_inc_idx})='{headers[merged_inc_idx].strip()[:20]}' + "
            f"expense(idx={merged_exp_idx})='{headers[merged_exp_idx].strip()[:20]}'"
        )
        return True, merged_inc_idx, merged_exp_idx

    return False, None, None
