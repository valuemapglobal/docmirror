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


def merge_split_rows(table: List[List[str]]) -> List[List[str]]:
    """Merge records that were split across multiple rows.

    Rules (by priority):
      R1: Row starts with a pure time (HH:MM:SS) and the previous row's
          first cell is a date → merge time into date.
      R2: Row is mostly empty (> 60 % columns) and is not a summary row
          → merge non-empty columns into the previous row.

    Generalised: no dependency on specific column names or bank formats.
    """
    if not table or len(table) < 3:
        return table

    header = table[0]
    col_count = len(header)
    result = [header]
    i = 1

    while i < len(table):
        row = table[i]

        # Ensure consistent column count (defensive)
        if len(row) < col_count:
            row = row + [""] * (col_count - len(row))

        # ── R1: pure time row → merge into previous row's date ──
        first_cell = row[0].strip() if row else ""
        if (
            result  # previous row exists
            and len(result) > 1  # not the header
            and _TIME_ONLY_RE.match(first_cell)
        ):
            prev_row = list(result[-1])
            prev_first = prev_row[0].strip()

            if _DATE_RE.match(prev_first):
                # Merge: "2025-12-24" + "01:21:34" → "2025-12-24 01:21:34"
                prev_row[0] = f"{prev_first} {first_cell}"
                # Merge other non-empty columns
                for j in range(1, min(len(row), len(prev_row))):
                    if row[j].strip() and not prev_row[j].strip():
                        prev_row[j] = row[j]
                    elif row[j].strip() and prev_row[j].strip():
                        prev_row[j] = prev_row[j] + " " + row[j]
                result[-1] = prev_row
                i += 1
                continue

        # ── R2: mostly empty row → merge into previous row ──
        non_empty = sum(1 for c in row if c.strip())
        empty_ratio = 1 - (non_empty / col_count) if col_count > 0 else 0

        if (
            empty_ratio > 0.6
            and len(result) > 1  # not the header
            and non_empty > 0  # not a fully empty row
            and not _is_summary_row(row)  # not a summary row
        ):
            prev_row = list(result[-1])
            for j in range(min(len(row), len(prev_row))):
                if row[j].strip() and not prev_row[j].strip():
                    prev_row[j] = row[j]
                elif row[j].strip() and prev_row[j].strip():
                    # Append (multi-line text such as counterparty names)
                    prev_row[j] = prev_row[j] + row[j]
            result[-1] = prev_row
            i += 1
            continue

        result.append(row)
        i += 1

    return result


def _is_summary_row(row: List[str]) -> bool:
    """Detect summary rows (should not be merged)."""
    text = "".join(str(c) for c in row)
    summary_keywords = ["\u603b\u6536\u5165", "\u603b\u652f\u51fa", "\u5408\u8ba1", "\u603b\u8ba1", "\u5c0f\u8ba1", "\u672c\u9875", "\u7d2f\u8ba1"]
    return any(kw in text for kw in summary_keywords)


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 2: Clean cell text
# ═══════════════════════════════════════════════════════════════════════════════

# Spaces between CJK characters (should be removed)
_CJK_SPACE_RE = re.compile(
    r"([\u4e00-\u9fff\u3400-\u4dbf])\s+([\u4e00-\u9fff\u3400-\u4dbf])"
)


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


def clean_table_cells(table: List[List[str]]) -> List[List[str]]:
    """Apply text cleanup to all cells in a table."""
    return [
        [clean_cell_text(cell) if isinstance(cell, str) else cell for cell in row]
        for row in table
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 3: Split concatenated cells
# ═══════════════════════════════════════════════════════════════════════════════

# Digit\u2192letter boundary (e.g. "110.9731080243CNYFC")
_NUM_ALPHA_BOUNDARY_RE = re.compile(
    r"(\d{1,3}\.\d{2})"           # Amount portion (e.g. 110.97)
    r"(\d{5,}[A-Z]*\d*)"         # Account number portion (e.g. 31080243CNYFC0445)
)


def split_concatenated_cells(
    table: List[List[str]],
) -> List[List[str]]:
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

def align_row_columns(table: List[List[str]]) -> List[List[str]]:
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
            merged = row[:target - 1] + [" ".join(str(c) for c in row[target - 1:] if c)]
            result.append(merged)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 5: Strip underline footer
# ═══════════════════════════════════════════════════════════════════════════════

# Match >= 3 consecutive underscores (footer separator line)
_UNDERLINE_RE = re.compile(r"_{3,}")


def strip_underline_footer(table: List[List[str]]) -> List[List[str]]:
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


def trim_trailing_empty_columns(table: List[List[str]]) -> List[List[str]]:
    """Trim all-empty trailing columns.

    Generalised: only trims from the end; does not affect interior empty columns.
    """
    if not table or not table[0]:
        return table

    col_count = max(len(row) for row in table)
    trim_to = col_count
    for ci in range(col_count - 1, -1, -1):
        all_empty = all(
            not (row[ci] if ci < len(row) else "").strip()
            for row in table
        )
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


def merge_digit_spaces(table: List[List[str]]) -> List[List[str]]:
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
    "Counterparty Institution", "Counterparty Name",
    "Transaction Amount", "Transaction Date",
    "Account Balance", "Abstract Code",
    "Serial Number", "Description",
    "Debit", "Credit",
]
# Regex: match "CJK...English suffix" pattern
_BILINGUAL_SUFFIX_RE = re.compile(
    r"([\u4e00-\u9fff][\u4e00-\u9fff\s]*)\s*("
    + "|".join(re.escape(s) for s in _BILINGUAL_SUFFIXES)
    + r")\s*$"
)


def strip_header_labels_from_cells(table: List[List[str]]) -> List[List[str]]:
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


def remove_empty_tables(tables: List[List[List[str]]]) -> List[List[List[str]]]:
    """Remove tables where all cells are empty.

    Generalised: only removes fully empty tables; tables with any data are kept.
    """
    result = []
    for table in tables:
        has_content = any(
            (cell or "").strip()
            for row in table
            for cell in row
        )
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


def split_account_from_name(table: List[List[str]]) -> List[List[str]]:
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


def strip_currency_prefix(table: List[List[str]]) -> List[List[str]]:
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

def fix_table_structure(table: List[List[str]]) -> List[List[str]]:
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

    table = merge_split_rows(table)              # Fix 1
    table = split_concatenated_cells(table)       # Fix 3
    table = align_row_columns(table)              # Fix 4
    table = clean_table_cells(table)              # Fix 2
    table = strip_underline_footer(table)         # Fix 5
    table = trim_trailing_empty_columns(table)    # Fix 6
    table = merge_digit_spaces(table)             # Fix 7
    table = strip_header_labels_from_cells(table) # Fix 8
    table = split_account_from_name(table)        # Fix 11
    table = strip_currency_prefix(table)          # Fix 12
    table = remove_empty_interior_columns(table)  # Fix 13

    fixed_rows = len(table)
    if fixed_rows != original_rows:
        logger.info(
            f"[DocMirror] table_structure_fix: "
            f"{original_rows} \u2192 {fixed_rows} rows "
            f"(merged {original_rows - fixed_rows})"
        )

    return table


def remove_empty_interior_columns(table: List[List[str]]) -> List[List[str]]:
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
            not (row[ci] if ci < len(row) else "").strip()
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

    # Removal condition: data all-empty AND (header is empty OR header is a duplicate)
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

    logger.debug(
        f"[fix] removed {len(cols_to_remove)} empty interior columns: "
        f"{sorted(cols_to_remove)}"
    )
    return result
