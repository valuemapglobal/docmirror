"""
pdfplumber Header Recovery — Layer 1 header recovery strategy.

Split from ``table_extraction.py``.

When pdfplumber's ``lines`` / ``text`` strategies discard the table
header row, this module recovers it from the zone's character data.
"""
from __future__ import annotations


import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ...utils.text_utils import _is_cjk_char, _smart_join
from ...utils.vocabulary import KNOWN_HEADER_WORDS, _ALL_BORDER_CHARS, _is_header_row, _normalize_for_vocab, _score_header_by_vocabulary, _RE_IS_DATE, _RE_IS_AMOUNT

logger = logging.getLogger(__name__)


# Forward import for TABLE_SETTINGS
from .classifier import TABLE_SETTINGS

def _recover_header_from_zone(
    tables: List[List[List[str]]],
    work_page,
    table_zone_bbox: Optional[Tuple[float, float, float, float]],
    original_page,
) -> List[List[List[str]]]:
    """Recover a table header row when pdfplumber's lines/text strategy
    discards it.

    Root cause: pdfplumber's ``horizontal_strategy="lines"`` starts
    extraction from the first horizontal line.  If the header row sits
    above that line but still within the table zone, it gets dropped.

    This function detects the missing header case and reinserts the header
    at the front of the table.

    Uses **x-coordinate alignment**: header words are mapped to data columns
    by x-position, solving column-count mismatches (e.g. data column
    "RMB 2936.78" is split into two columns by pdfplumber, corresponding
    to a single header "Account Balance").
    """
    if not tables or not table_zone_bbox:
        return tables

    main_table = tables[0]
    if not main_table or len(main_table) < 1:
        return tables

    # If a header already exists in the first 10 rows, no recovery needed
    # (post_process_table's _score scan will find it correctly)
    if any(_score_header_by_vocabulary(row) >= 3 for row in main_table[:10]):
        return tables

    # Extract words from the zone region to find header candidates
    try:
        x0, y0, x1, y1 = table_zone_bbox
        zone_page = original_page.crop((x0, y0, x1, y1))
        words = zone_page.extract_words(keep_blank_chars=True, x_tolerance=2)
        if not words:
            return tables
    except Exception as exc:
        logger.debug(f"operation: suppressed {exc}")
        return tables

    # Group words into rows by y-coordinate
    from collections import defaultdict
    y_rows: Dict[int, list] = defaultdict(list)
    for w in words:
        yk = round(w["top"] / 3) * 3
        y_rows[yk].append(w)

    sorted_yks = sorted(y_rows.keys())
    if len(sorted_yks) < 2:
        return tables

    # Find the row with the best vocabulary match in the first few rows
    best_yk = -1
    best_score = 0
    for yk in sorted_yks[:5]:
        texts = [w["text"].strip() for w in y_rows[yk] if w["text"].strip()]
        score = sum(1 for t in texts if t in KNOWN_HEADER_WORDS)
        if score > best_score:
            best_score = score
            best_yk = yk

    if best_score < 3 or best_yk < 0:
        return tables

    # Check: is the header already present in the first row? (no recovery needed)
    header_words = sorted(y_rows[best_yk], key=lambda w: w["x0"])
    header_texts = [w["text"].strip() for w in header_words if w["text"].strip()]
    first_row_text = set(c.strip() for c in main_table[0] if (c or "").strip())
    header_text_set = set(header_texts)
    if len(first_row_text & header_text_set) >= 2:
        return tables

    # ── x-coordinate alignment: map header words to data columns ──
    n_cols = len(main_table[0])

    # Get pdfplumber table column boundaries (vertical edges)
    col_midpoints = None
    try:
        tf = work_page.debug_tablefinder(table_settings=TABLE_SETTINGS)
        v_edges = sorted(set(
            round(e['x0'], 1) for e in tf.edges
            if abs(e['x0'] - e['x1']) < 1  # Vertical lines
        ))
        if len(v_edges) >= 2:
            col_midpoints = [
                (v_edges[i] + v_edges[i + 1]) / 2
                for i in range(len(v_edges) - 1)
            ]
    except Exception as exc:
        logger.debug(f"operation: suppressed {exc}")

    if col_midpoints and len(col_midpoints) == n_cols:
        # For each header word, find the nearest data column by x-centre
        header_row = [""] * n_cols
        for hw in header_words:
            text = hw["text"].strip()
            if not text:
                continue
            hx_mid = (hw["x0"] + hw["x1"]) / 2
            best_col = min(range(len(col_midpoints)),
                           key=lambda ci: abs(col_midpoints[ci] - hx_mid))
            if header_row[best_col]:
                header_row[best_col] += text
            else:
                header_row[best_col] = text

        logger.info(
            f"header recovery (x-aligned): vocab_score={best_score}, "
            f"header={header_row[:4]}..."
        )
        # Remove duplicate header rows from main_table (vocab_score >= 3),
        # keep preamble key-value rows
        clean_body = [
            row for row in main_table
            if _score_header_by_vocabulary(row) < 3
        ]
        new_table = [header_row] + clean_body
        return [new_table] + tables[1:]

    # Fallback: simple alignment (when data-row words are unavailable)
    header_row = list(header_texts)
    if len(header_row) > n_cols:
        header_row = header_row[:n_cols]
    elif len(header_row) < n_cols:
        header_row = header_row + [""] * (n_cols - len(header_row))

    logger.info(
        f"header recovery (fallback): vocab_score={best_score}, "
        f"header={header_row[:4]}..."
    )
    # Remove duplicate header rows, keep preamble key-value rows
    clean_body = [
        row for row in main_table
        if _score_header_by_vocabulary(row) < 3
    ]
    new_table = [header_row] + clean_body
    return [new_table] + tables[1:]
