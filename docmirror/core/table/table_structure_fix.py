# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Table Structure Fix Engine
===========================

A general-purpose, domain-agnostic module for post-processing table
structure defects.  Runs after OCR / extraction and before final output.

10 independent fix functions + 1 unified entry point:
    1. ``merge_split_rows``             — Merge records split across multiple rows
    2. ``clean_cell_text``              — Remove excess whitespace / newlines in cells
    3. ``split_concatenated_cells``     — Split concatenated cells (e.g. balance + account number)
    4. ``align_row_columns``            — Align row column counts to the header
    5. ``strip_underline_footer``       — Remove underline-delimited footer statistics
    6. ``trim_trailing_empty_columns``  — Trim all-empty trailing columns
    7. ``merge_digit_spaces``           — Remove spaces inside pure-digit cells
    8. ``strip_header_labels_from_cells`` — Remove bilingual column-title suffixes from data cells
    9. ``remove_empty_tables``          — Remove fully empty tables
   10. ``split_account_from_name``      — Split account numbers prefixed to account names
   11. ``strip_currency_prefix``        — Strip currency code prefixes from amounts
   12. ``remove_empty_interior_columns``— Remove all-empty interior columns

Design principles:
    - Pure functions, no state, no side effects.
    - Every fix performs a safety check; uncertain cases are left unmodified.
    - Empty / single-row tables are returned as-is.
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 1: Merge records split across multiple rows
# ═══════════════════════════════════════════════════════════════════════════════

# Pure time pattern (HH:MM:SS or HH:MM)
_TIME_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
# Date pattern (YYYY-MM-DD or YYYY.MM.DD or YYYY/MM/DD)
_DATE_RE = re.compile(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$")
_TIME_ONLY_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def _smart_join(s1: str, s2: str) -> str:
    """Joins two strings with a space, handling empty strings gracefully."""
    if not s1:
        return s2
    if not s2:
        return s1
    return f"{s1} {s2}"


def merge_split_rows(table: list[list[str]]) -> list[list[str]]:
    """Merge records that were split across multiple rows using Unsupervised Anchor Folding (Phase 3)."""
    try:
        if len(table) < 2:
            return table

        header = table[0]
        data = table[1:]
        col_count = len(header)

        if not data:
            return table

        # ── Step 1: Detect Semantic Anchor Columns ──
        # An anchor column is a strong indicator of a new transaction.
        # Primary anchors are Date columns and Amount columns.
        date_col = -1
        amount_col = -1

        # Scan header to find potential anchor columns
        for c, h_text in enumerate(header):
            h_norm = (h_text or "").strip().lower()
            if any(kw in h_norm for kw in ["日期", "时间", "date"]):
                date_col = c
            if any(kw in h_norm for kw in ["发生额", "金额", "支出", "存入", "余额", "amount", "balance"]):
                amount_col = c

        # If header search failed, scan the first 5 data rows
        if date_col == -1 or amount_col == -1:
            for c in range(min(10, col_count)):
                date_score = 0
                amount_score = 0
                for row in data[:5]:
                    if c < len(row):
                        cell = (row[c] or "").strip()
                        if _DATE_RE.match(cell) or _TIME_ONLY_RE.match(cell):
                            date_score += 1
                        # Remove commas and currency symbols for amount check
                        clean_cell = cell.replace(",", "").replace("¥", "").replace("$", "")
                        if re.match(r"^-?\d+\.\d{2}$", clean_cell):
                            amount_score += 1

                if date_score > 0 and date_col == -1:
                    date_col = c
                if amount_score > 0 and amount_col == -1:
                    amount_col = c

        # ── Step 2: Global Semantic Closure Engine ──
        result = [header]
        current_record = None

        for row in data:
            # Pad or truncate row to match header
            if len(row) < col_count:
                row = row + [""] * (col_count - len(row))
            elif len(row) > col_count:
                row = row[:col_count]

            # Check if row has any content at all
            has_data = any((c or "").strip() for c in row)
            if not has_data:
                continue

            # Semantic Anchor Detection for the current row
            is_anchor_row = False

            # Condition 1: Has a valid date in the date column
            if date_col != -1 and date_col < len(row):
                cell = (row[date_col] or "").strip()
                if _DATE_RE.match(cell):
                    is_anchor_row = True

            # Condition 2: Has a valid amount in the amount column
            if not is_anchor_row and amount_col != -1 and amount_col < len(row):
                cell = (row[amount_col] or "").strip().replace(",", "").replace("¥", "")
                if re.match(r"^-?\d+\.\d{2}$", cell):
                    is_anchor_row = True

            # Fallback Pattern 1: Any sequence number at column 0 (e.g. "123")
            if not is_anchor_row and len(row) > 0:
                cell = (row[0] or "").strip()
                if re.match(r"^\d{1,6}$", cell):
                    is_anchor_row = True

            # Fallback Pattern 2: Time-Only continuation
            # (Special case: time is split from date onto the next line)
            first_cell = (row[0] or "").strip()
            if current_record and _TIME_ONLY_RE.match(first_cell) and _DATE_RE.match((current_record[0] or "").strip()):
                current_record[0] = f"{current_record[0].strip()} {first_cell}"
                for j in range(1, col_count):
                    v = (row[j] or "").strip()
                    if v:
                        existing = (current_record[j] or "").strip()
                        current_record[j] = _smart_join(existing, v) if existing else v
                continue

            if is_anchor_row:
                # ── Anchor Row: Seal the previous record and start a new one ──
                if current_record:
                    result.append(current_record)
                current_record = list(row)
            else:
                # ── Fragment Row: Merge into the current open record ──
                if current_record:
                    logger.info(
                        "[TableFix] Merging split fragment row into anchor record (Unsupervised Anchor Folding)"
                    )
                    # Semantic Closure: Append to existing record
                    for i in range(col_count):
                        v = (row[i] or "").strip()
                        if v:
                            existing = (current_record[i] or "").strip()
                            # Use newline for structural merges, smart_join for others
                            if existing:
                                current_record[i] = existing + "\n" + v
                            else:
                                current_record[i] = v
                else:
                    # Rare edge case: Fragment appears before any Anchor row
                    # (Usually happens due to preamble strip failure)
                    # We have to treat it as a new record
                    current_record = list(row)

        # Seal the final record
        if current_record:
            result.append(current_record)

        return result
    except Exception as e:
        import sys
        import traceback

        print(f"CRASH IN MERGE_SPLIT_ROWS: {e}", file=sys.stderr)
        traceback.print_exc()
        return table


def _is_summary_row(row: list[str]) -> bool:
    """Detect summary rows (should not be merged)."""
    text = "".join(str(c) for c in row)
    summary_keywords = [
        "\u603b\u6536\u5165",
        "\u603b\u652f\u51fa",
        "\u5408\u8ba1",
        "\u603b\u8ba1",
        "\u5c0f\u8ba1",
        "\u672c\u9875",
        "\u7d2f\u8ba1",
    ]
    return any(kw in text for kw in summary_keywords)


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 2: Clean cell text
# ═══════════════════════════════════════════════════════════════════════════════

# Spaces between CJK characters (should be removed)
_CJK_SPACE_RE = re.compile(r"([\u4e00-\u9fff\u3400-\u4dbf])\s+([\u4e00-\u9fff\u3400-\u4dbf])")


def clean_cell_text(text: str) -> str:
    """Remove excess whitespace and newlines from a cell.

    Rules:
      - Spaces between CJK characters \u2192 removed (artefact of PDF multi-line text reassembly).
      - Preserved: spaces between English words, digits, or CJK-English boundaries.
      - Leading / trailing whitespace is trimmed.
    """
    if not text or not text.strip():
        return text.strip()

    # Replace newlines with spaces
    text = text.replace("\n", " ").replace("\r", "")

    # Iteratively remove spaces between CJK characters (handles "A B C" \u2192 "ABC")
    prev = ""
    while prev != text:
        prev = text
        text = _CJK_SPACE_RE.sub(r"\1\2", text)

    return text.strip()


def clean_table_cells(table: list[list[str]]) -> list[list[str]]:
    """Apply text cleanup to all cells in a table."""
    return [[clean_cell_text(cell) if isinstance(cell, str) else cell for cell in row] for row in table]


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 3: Split concatenated cells
# ═══════════════════════════════════════════════════════════════════════════════

# Digit\u2192letter boundary (e.g. "110.9731080243CNYFC")
_NUM_ALPHA_BOUNDARY_RE = re.compile(
    r"(\d{1,3}\.\d{2})"  # Amount portion (e.g. 110.97)
    r"(\d{5,}[A-Z]*\d*)"  # Account number portion (e.g. 31080243CNYFC0445)
)


def split_concatenated_cells(
    table: list[list[str]],
) -> list[list[str]]:
    """Split concatenated cells \u2014 triggered when a row has fewer columns
    than the header.

    Rules:
      - Only triggered when row column count < header column count.
      - Detects digit.digit + digit-alpha concatenation patterns.
      - After splitting, column count should equal the header count.
    """
    if not table or len(table) < 2:
        return table

    header = table[0]
    header_col_count = len(header)
    result = [header]

    for row in table[1:]:
        if len(row) >= header_col_count:
            result.append(row)
            continue

        # Attempt to split concatenated cells
        deficit = header_col_count - len(row)
        if deficit <= 0:
            result.append(row)
            continue

        new_row = []
        splits_done = 0
        for cell in row:
            if splits_done >= deficit and len(new_row) + (len(row) - len(new_row)) <= header_col_count:
                new_row.append(cell)
                continue

            # Detect amount + account-number concatenation
            m = _NUM_ALPHA_BOUNDARY_RE.match(str(cell))
            if m and splits_done < deficit:
                new_row.append(m.group(1))  # Amount
                new_row.append(m.group(2))  # Account number
                splits_done += 1
            else:
                new_row.append(cell)

        # Use the new row if column count matches; otherwise keep the original
        if len(new_row) == header_col_count:
            result.append(new_row)
        else:
            result.append(row)  # Split failed — keep the original row

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 4: Align row column counts
# ═══════════════════════════════════════════════════════════════════════════════


def align_row_columns(table: list[list[str]]) -> list[list[str]]:
    """Align all rows' column counts to the header.

    Rules:
      - Fewer columns than header \u2192 pad with empty strings at the end.
      - More columns than header \u2192 merge excess trailing columns into the last column.
    """
    if not table:
        return table

    header = table[0]
    target = len(header)
    result = [header]

    for row in table[1:]:
        if len(row) == target:
            result.append(row)
        elif len(row) < target:
            result.append(row + [""] * (target - len(row)))
        else:
            # Merge excess columns into the last column
            merged = row[: target - 1] + [" ".join(str(c) for c in row[target - 1 :] if c)]
            result.append(merged)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 5: Strip underline footer
# ═══════════════════════════════════════════════════════════════════════════════

# Match >= 3 consecutive underscores (footer separator line)
_UNDERLINE_RE = re.compile(r"_{3,}")


def strip_underline_footer(table: list[list[str]]) -> list[list[str]]:
    """Remove underline-delimited footer statistics from cells.

    Pattern: "4085.26___Total debits:...__Total credits:...__Count:..."
    Rule: truncate at the first ``___`` (keep the actual data value before it).

    Generalised: not dependent on specific column names; any cell containing
    ``___`` is processed.
    """
    for row in table:
        for ci in range(len(row)):
            cell = row[ci] or ""
            m = _UNDERLINE_RE.search(cell)
            if m:
                row[ci] = cell[: m.start()].rstrip()
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 6: Trim trailing empty columns
# ═══════════════════════════════════════════════════════════════════════════════


def trim_trailing_empty_columns(table: list[list[str]]) -> list[list[str]]:
    """Trim all-empty trailing columns.

    Generalised: only trims from the end; does not affect interior empty columns.
    """
    if not table or not table[0]:
        return table

    col_count = max(len(row) for row in table)
    trim_to = col_count
    for ci in range(col_count - 1, -1, -1):
        all_empty = all(not (row[ci] if ci < len(row) else "").strip() for row in table)
        if all_empty:
            trim_to = ci
        else:
            break

    if trim_to < col_count:
        table = [row[:trim_to] for row in table]
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 7: Merge digit spaces
# ═══════════════════════════════════════════════════════════════════════════════

# Match pure "digits + spaces" pattern (no letters / CJK)
_DIGIT_SPACE_RE = re.compile(r"^[\d\s]+$")


def merge_digit_spaces(table: list[list[str]]) -> list[list[str]]:
    """Remove spaces inside pure-digit cells.

    Pattern: "6216911304 963684" \u2192 "6216911304963684"
    Rule: only applies to cells containing digits and spaces only;
    cells with letters or CJK are not affected.

    Generalised: automatic detection, no column-name dependency.
    """
    for row in table[1:]:  # Skip header
        for ci in range(len(row)):
            cell = (row[ci] or "").strip()
            if cell and _DIGIT_SPACE_RE.match(cell) and " " in cell:
                row[ci] = cell.replace(" ", "")
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 8: Strip bilingual header labels from data cells
# ═══════════════════════════════════════════════════════════════════════════════

# Common bilingual column-title suffixes (English portion) — matched longest-first
_BILINGUAL_SUFFIXES = [
    "Counterparty Institution",
    "Counterparty Name",
    "Transaction Amount",
    "Transaction Date",
    "Account Balance",
    "Abstract Code",
    "Serial Number",
    "Description",
    "Debit",
    "Credit",
]
# Regex: match "CJK...English suffix" pattern
_BILINGUAL_SUFFIX_RE = re.compile(
    r"([\u4e00-\u9fff][\u4e00-\u9fff\s]*)\s*(" + "|".join(re.escape(s) for s in _BILINGUAL_SUFFIXES) + r")\s*$"
)


def strip_header_labels_from_cells(table: list[list[str]]) -> list[list[str]]:
    """Remove bilingual column-title suffixes concatenated to data cells.

    Patterns:
      - "0.90DebitDebit" \u2192 "0.90"
      - "\u6d66\u53d1\u94f6\u884c\u91cd\u5e86\u5206\u884c\u8425\u4e1a\u90e8 Counterparty Institution" \u2192 "\u6d66\u53d1\u94f6\u884c\u91cd\u5e86\u5206\u884c\u8425\u4e1a\u90e8"

    Rule: detect "CJK + English" column-title composition at cell end,
    truncate before the CJK portion.
    Generalised: based on common bilingual column-title keywords, not specific columns.
    """
    for row in table[1:]:  # Skip header
        for ci in range(len(row)):
            cell = (row[ci] or "").strip()
            if not cell or len(cell) < 5:
                continue
            m = _BILINGUAL_SUFFIX_RE.search(cell)
            if m:
                # Truncate before the CJK column-title begins
                row[ci] = cell[: m.start()].rstrip()
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 9: Remove fully empty tables
# ═══════════════════════════════════════════════════════════════════════════════


def remove_empty_tables(tables: list[list[list[str]]]) -> list[list[list[str]]]:
    """Remove tables where all cells are empty.

    Generalised: only removes fully empty tables; tables with any data are kept.
    """
    result = []
    for table in tables:
        has_content = any((cell or "").strip() for row in table for cell in row)
        if has_content:
            result.append(table)
        else:
            logger.debug("[DocMirror] removed empty table")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 11: Split account numbers prefixed to account names
# ═══════════════════════════════════════════════════════════════════════════════

# >= 10 consecutive digits + CJK (account number concatenated with account name)
_ACCT_PREFIX_RE = re.compile(r"^(\d{10,})([\u4e00-\u9fff].*)$")


def split_account_from_name(table: list[list[str]]) -> list[list[str]]:
    """Split long digit prefixes (account numbers) from account-name cells.

    Pattern: "7065018800015\u9547\u6c5f\u4e00\u751f\u4e00\u4e16\u597d\u6e38\u620f\u6709\u9650\u516c\u53f8" \u2192
             counterparty account = "7065018800015",
             counterparty name = "\u9547\u6c5f\u4e00\u751f\u4e00\u4e16\u597d\u6e38\u620f\u6709\u9650\u516c\u53f8"

    Rules:
      - If a cell starts with >= 10 consecutive digits followed by CJK \u2192 split.
      - The digit portion is merged into the preceding column (if its header
        contains "account" / "account number").
      - Generalised: based on header content matching, not column-name hard-coding.
    """
    if not table or len(table) < 2 or len(table[0]) < 2:
        return table

    header = table[0]

    # Find the "counterparty account" column and its right neighbour
    acct_col = None
    for ci, h in enumerate(header):
        h_text = (h or "").strip()
        if ("\u8d26\u6237" in h_text or "\u8d26\u53f7" in h_text) and "\u5bf9\u65b9" in h_text:
            if ci + 1 < len(header):
                acct_col = ci
                break

    if acct_col is None:
        return table

    name_col = acct_col + 1

    for row in table[1:]:
        if name_col >= len(row):
            continue
        cell = (row[name_col] or "").strip()
        m = _ACCT_PREFIX_RE.match(cell)
        if m:
            digits, name = m.group(1), m.group(2)
            # Merge digits into the account column (prepend, space-separated)
            existing = (row[acct_col] or "").strip()
            row[acct_col] = (digits + " " + existing).strip() if existing else digits
            row[name_col] = name.strip()

    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 12: Strip currency prefix
# ═══════════════════════════════════════════════════════════════════════════════

# Match "RMB 352.10" or "CNY352.10" or "USD 1,000.00"
_CURRENCY_PREFIX_RE = re.compile(
    r"^(RMB|CNY|USD|EUR|JPY|HKD|GBP)\s*"
    r"([\-\d,]+\.?\d*)\s*$"
)


def strip_currency_prefix(table: list[list[str]]) -> list[list[str]]:
    """Strip currency code prefixes from amount cells.

    Patterns: "RMB 352.10" \u2192 "352.10", "RMB7.77" \u2192 "7.77"
    Rule: only processes pure "currency code + number" cells; cells with
    other text are not affected.
    Supports: RMB / CNY / USD / EUR / JPY / HKD / GBP.
    """
    for row in table[1:]:  # Skip header
        for ci in range(len(row)):
            cell = (row[ci] or "").strip()
            if not cell:
                continue
            m = _CURRENCY_PREFIX_RE.match(cell)
            if m:
                row[ci] = m.group(2)
    return table


# ═══════════════════════════════════════════════════════════════════════════════
# Unified entry point
# ═══════════════════════════════════════════════════════════════════════════════


def fix_table_structure(table: list[list[str]]) -> list[list[str]]:
    """Unified entry point for table structure fixes.

    Executes in order:
      1. Row merging (date + time, multi-line cells)
      2. Concatenated cell splitting (balance + account number)
      3. Column count alignment
      4. Cell text cleanup
      5. Underline footer removal
      6. Trailing empty column trimming
      7. Pure-digit space merging
      8. Bilingual header-label removal
      9. Account number / name splitting
     10. Currency prefix stripping
     11. Empty interior column removal

    Args:
        table: Raw table (2-D string list; row 0 is the header).

    Returns:
        Fixed table.
    """
    if not table or len(table) < 2:
        return table

    original_rows = len(table)

    table = merge_split_rows(table)  # Fix 1
    table = split_concatenated_cells(table)  # Fix 3
    table = align_row_columns(table)  # Fix 4
    table = clean_table_cells(table)  # Fix 2
    table = strip_underline_footer(table)  # Fix 5
    table = trim_trailing_empty_columns(table)  # Fix 6
    table = merge_digit_spaces(table)  # Fix 7
    table = strip_header_labels_from_cells(table)  # Fix 8
    table = split_account_from_name(table)  # Fix 11
    table = strip_currency_prefix(table)  # Fix 12
    table = remove_empty_interior_columns(table)  # Fix 13

    fixed_rows = len(table)
    if fixed_rows != original_rows:
        logger.info(
            f"[DocMirror] table_structure_fix: "
            f"{original_rows} \u2192 {fixed_rows} rows "
            f"(merged {original_rows - fixed_rows})"
        )

    return table


def remove_empty_interior_columns(table: list[list[str]]) -> list[list[str]]:
    """Remove all-empty interior columns (including those with empty or
    duplicate headers).

    In dual-row header scenarios (e.g. Bank of Communications), debit / credit
    amount columns may produce empty columns + duplicate header names after
    post-processing.  This function only removes columns where **all** data
    rows are empty.

    Args:
        table: Fixed table (row 0 is the header).

    Returns:
        Table with empty interior columns removed.
    """
    if not table or len(table) < 2:
        return table

    n_cols = len(table[0])
    if n_cols <= 1:
        return table

    # Identify columns where all data rows are empty
    empty_cols: set = set()
    for ci in range(n_cols):
        if all(
            not ((row[ci] if ci < len(row) else "") or "").strip()
            for row in table[1:]  # skip header
        ):
            empty_cols.add(ci)

    if not empty_cols:
        return table

    # Build header-value occurrence counts for duplicate detection
    header_vals = [(table[0][ci] if ci < len(table[0]) else "").strip() for ci in range(n_cols)]
    header_counts: dict = {}
    for h in header_vals:
        header_counts[h] = header_counts.get(h, 0) + 1

    # C3 FIX: Removal condition — only remove columns where:
    #   - All data rows are empty, AND
    #   - Header is empty OR header is a duplicate
    # This prevents cross-page column count mismatch when post-process
    # runs before merge (Step 4↔5 swap): a column may be empty on page N
    # but have data on page N+1.  Unique-named header columns are ALWAYS
    # preserved to keep column counts consistent across pages.
    cols_to_remove: set = set()
    for ci in empty_cols:
        hv = header_vals[ci]
        if not hv or header_counts.get(hv, 0) > 1:
            cols_to_remove.add(ci)

    if not cols_to_remove:
        return table

    keep = [ci for ci in range(n_cols) if ci not in cols_to_remove]
    result = []
    for row in table:
        result.append([row[ci] if ci < len(row) else "" for ci in keep])

    logger.info(
        f"[TableFix] Dropping {len(cols_to_remove)} empty interior columns: "
        f"indexes {sorted(cols_to_remove)} (often caused by dual-row headers)"
    )
    return result
