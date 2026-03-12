"""Character-level table extraction strategies — Layer 2.

Split from ``table_extraction.py``.
"""
from __future__ import annotations


import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ...utils.text_utils import _is_cjk_char, _smart_join
from ...utils.vocabulary import _ALL_BORDER_CHARS, _is_header_row, _is_header_cell, _normalize_for_vocab, _score_header_by_vocabulary, _RE_IS_DATE, _RE_IS_AMOUNT
from ...utils.watermark import is_watermark_char
from ..postprocess import _find_vocab_words_in_string

logger = logging.getLogger(__name__)


from .utils import _group_chars_into_rows, _cluster_x_positions, _assign_chars_to_columns, _chars_to_text

def _extract_by_hline_columns(page_plum) -> Optional[List[List[str]]]:
    """Horizontal-line column boundary method.

    For PDFs with horizontal dividers but no vertical lines
    (e.g. China Merchants Bank transaction statements).
    Horizontal-line x-endpoints define column boundaries;
    data rows are clustered by word y-coordinates.

    Trigger conditions: >= 3 horizontal lines, 0 vertical lines.
    """
    lines = page_plum.lines or []
    if not lines:
        return None

    # Classify lines: horizontal vs vertical
    h_lines = [l for l in lines if abs(l["top"] - l["bottom"]) < 1]
    v_lines = [l for l in lines if abs(l["x0"] - l["x1"]) < 1]

    # Trigger: enough horizontal lines, no vertical lines
    if len(h_lines) < 3 or len(v_lines) > 0:
        return None

    # ── Extract column boundaries from horizontal-line x-coordinates ──
    raw_x = sorted(set(
        round(v, 1)
        for l in h_lines
        for v in [l["x0"], l["x1"]]
    ))
    # Merge nearby x values (snap with 10 pt threshold — avoid tiny gaps creating empty columns)
    x_positions = [raw_x[0]]
    for x in raw_x[1:]:
        if x - x_positions[-1] > 10:
            x_positions.append(x)

    if len(x_positions) < 3:
        return None  # Too few columns — unlikely to be a table

    # ── Determine column intervals ──
    col_count = len(x_positions) - 1
    intervals = [(x_positions[i], x_positions[i + 1]) for i in range(col_count)]

    # ── Determine header region (y-range of horizontal lines) ──
    h_y_values = sorted(set(round(l["top"], 1) for l in h_lines))
    # Header sits between the top two lines; data starts from the second line
    if len(h_y_values) < 2:
        return None
    header_top = h_y_values[0]
    data_start_y = h_y_values[1]

    # ── Extract words from the page ──
    try:
        words = page_plum.extract_words(keep_blank_chars=True)
    except Exception as exc:
        logger.debug(f"operation: suppressed {exc}")
        return None
    if not words:
        return None

    # ── Cluster words into rows by y-coordinate ──
    ROW_TOLERANCE = 5  # Words within 5 pt y-distance belong to the same row
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    rows_words = []  # list of (y, [words])
    current_y = -999
    current_row = []
    for w in sorted_words:
        if w["top"] - current_y > ROW_TOLERANCE:
            if current_row:
                rows_words.append((current_y, current_row))
            current_y = w["top"]
            current_row = [w]
        else:
            current_row.append(w)
    if current_row:
        rows_words.append((current_y, current_row))

    # ── Keep only header rows (between header_top and data_start_y) + data rows (below) ──
    # Validate with _is_header_row: after cropping, the header interval may
    # contain data rows (e.g. if the first h-line was cropped away)
    header_rows = []
    data_rows = []
    for y, rw in rows_words:
        if header_top - 2 <= y < data_start_y:
            # Vocabulary validation: rows with dates / amounts / long numbers → data, not header
            texts = [w["text"].strip() for w in rw if w["text"].strip()]
            if _is_header_row(texts):
                header_rows.append(rw)
            else:
                data_rows.append(rw)
        elif y >= data_start_y:
            data_rows.append(rw)

    if not data_rows:
        return None

    # ── Assign words to columns ──
    def _words_to_row(row_words):
        cells = [""] * col_count
        for w in sorted(row_words, key=lambda w: w["x0"]):
            wx = w["x0"]
            assigned = False
            for ci, (x0, x1) in enumerate(intervals):
                if x0 - 5 <= wx < x1 + 5:
                    if cells[ci]:
                        cells[ci] += " " + w["text"]
                    else:
                        cells[ci] = w["text"]
                    assigned = True
                    break
            if not assigned and col_count > 0:
                # Words beyond the right boundary → last column
                if wx >= x_positions[-1] - 5:
                    if cells[-1]:
                        cells[-1] += " " + w["text"]
                    else:
                        cells[-1] = w["text"]
        return cells

    # Build header row
    header_cells = [""] * col_count
    for rw in header_rows:
        merged = _words_to_row(rw)
        for ci in range(col_count):
            if merged[ci]:
                if header_cells[ci]:
                    header_cells[ci] += " " + merged[ci]
                else:
                    header_cells[ci] = merged[ci]

    # ── Compute header anchors (crop-immune) ──
    # From all words above data_start_y, find the nearest word centre for
    # each column interval.  Independent of whether header_rows were correctly
    # recognised — completely unaffected by engine cropping.
    pre_data_words = [w for w in words if w["top"] < data_start_y]
    header_anchors = []
    for ci in range(col_count):
        interval_mid = (intervals[ci][0] + intervals[ci][1]) / 2
        if pre_data_words:
            # Find the pre-data word whose x-centre is closest to the interval midpoint
            best_w = min(
                pre_data_words,
                key=lambda w: abs((w["x0"] + w.get("x1", w["x0"] + 10)) / 2 - interval_mid)
            )
            anchor = (best_w["x0"] + best_w.get("x1", best_w["x0"] + 10)) / 2
            # Only accept if within half the interval width (prevent mismatch)
            interval_half = (intervals[ci][1] - intervals[ci][0]) / 2
            if abs(anchor - interval_mid) < interval_half:
                header_anchors.append(anchor)
                continue
        # Fallback: use the interval midpoint
        header_anchors.append(interval_mid)

    logger.debug(
        f"hline-columns: anchors={[f'{a:.1f}' for a in header_anchors]}"
    )

    # ── Data rows: nearest-neighbour anchor assignment ──
    def _words_to_row_nn(row_words):
        cells = [""] * col_count
        for w in sorted(row_words, key=lambda w: w["x0"]):
            w_center = (w["x0"] + w.get("x1", w["x0"] + 5)) / 2
            best_ci = min(
                range(col_count),
                key=lambda ci: abs(w_center - header_anchors[ci])
            )
            if cells[best_ci]:
                cells[best_ci] += " " + w["text"]
            else:
                cells[best_ci] = w["text"]
        return cells

    # Build data rows using nearest-neighbour assignment
    table = [header_cells]
    for rw in data_rows:
        table.append(_words_to_row_nn(rw))

    # Validate: too few data rows or columns → not a valid table
    if len(table) < 2 or col_count < 2:
        return None

    logger.info(
        f"hline-columns: {len(table)-1} data rows, "
        f"{col_count} cols from {len(h_lines)} h-lines"
    )
    return table


def _extract_by_rect_columns(page_plum) -> Optional[List[List[str]]]:
    """Rectangle column boundary method."""
    rects = page_plum.rects
    if not rects or len(rects) < 3:
        return None

    y_groups = defaultdict(list)
    for r in rects:
        y_key = round(r["top"] / 3) * 3
        y_groups[y_key].append(r)

    best_group = max(y_groups.values(), key=len)
    if len(best_group) < 3:
        return None

    raw_x = sorted(set(
        round(v, 1) for r in best_group
        for v in [r["x0"], r["x1"]]
    ))
    x_positions = [0.0]
    for x in raw_x:
        if x - x_positions[-1] > 2:
            x_positions.append(x)
    x_positions.append(page_plum.width)

    if len(x_positions) < 4:
        return None

    header_top = min(r["top"] for r in best_group) - 2
    header_bottom = max(r["bottom"] for r in best_group) + 1

    try:
        cropped = page_plum.crop((
            0, header_top,
            page_plum.width, page_plum.height,
        ))
        chars = cropped.chars
        if not chars:
            return None

        col_count = len(x_positions) - 1
        intervals = [(x_positions[i], x_positions[i + 1]) for i in range(col_count)]

        def _chars_to_row(row_chars):
            cells = [""] * col_count
            for c in sorted(row_chars, key=lambda c: c["x0"]):
                for ci, (x0, x1) in enumerate(intervals):
                    if x0 - 2 <= c["x0"] < x1 + 2:
                        cells[ci] += c["text"]
                        break
            return [cell.strip() for cell in cells]

        header_chars = [c for c in chars if c["top"] < header_bottom]
        data_chars = [c for c in chars if c["top"] >= header_bottom]

        table = []
        if header_chars:
            table.append(_chars_to_row(header_chars))

        row_groups = defaultdict(list)
        for c in data_chars:
            y_key = round(c["top"] / 3) * 3
            row_groups[y_key].append(c)

        for y_key in sorted(row_groups.keys()):
            row = _chars_to_row(row_groups[y_key])
            if any(cell for cell in row):
                table.append(row)

        while table and all(not row[0] for row in table):
            table = [row[1:] for row in table]
        while table and all(not row[-1] for row in table):
            table = [row[:-1] for row in table]

        if len(table) >= 3:
            return table

    except Exception as e:
        logger.debug(f"rect columns failed: {e}")

    return None


def detect_columns_by_header_anchors(page_plum) -> Optional[List[List[str]]]:
    """Header-anchor column detection method."""
    chars = page_plum.chars
    if not chars or len(chars) < 10:
        return None

    chars = [c for c in chars if not is_watermark_char(c)]
    if not chars:
        return None

    rows_by_y = _group_chars_into_rows(chars)
    if len(rows_by_y) < 2:
        return None

    header_row_idx = -1
    best_vocab_score = 0
    # Scan the first 15 rows: prefer vocab matches, handle KV metadata rows before the header
    for i, (y_mid, row_chars) in enumerate(rows_by_y[:15]):
        row_text = _chars_to_text(row_chars)
        cells = [t.strip() for t in row_text.split("  ") if t.strip()]
        if len(cells) < 2:
            continue
        vs = _score_header_by_vocabulary(cells)
        if vs > best_vocab_score:
            best_vocab_score = vs
            header_row_idx = i
        elif vs == 0 and header_row_idx == -1:
            # Fallback: structural heuristic (when no vocab matches)
            if all(_is_header_cell(c) for c in cells[:4]):
                header_row_idx = i

    if header_row_idx == -1:
        return None

    header_chars = rows_by_y[header_row_idx][1]
    col_bounds = _cluster_x_positions([c["x0"] for c in header_chars])

    if len(col_bounds) < 2:
        return None

    result: List[List[str]] = []
    for y_mid, row_chars in rows_by_y[header_row_idx:]:
        row = _assign_chars_to_columns(row_chars, col_bounds)
        result.append(row)

    return result if len(result) >= 2 else None


def _adjust_boundaries_by_vocab(
    col_boundaries: List[float],
    header_chars: List[dict],
) -> List[float]:
    """Vocabulary-guided column boundary adjustment.

    If a boundary falls inside a known header word, shift it past the word.

    Algorithm:
        1. Extract non-space characters from header chars to form the full header text.
        2. Use ``_find_vocab_words_in_string`` to find all vocabulary matches.
        3. For each match, determine the word's x-range from character coordinates.
        4. If any column boundary falls within a word's x-range, move it past the word.
    """
    text_chars = [c for c in header_chars if c["text"].strip()]
    if not text_chars:
        return col_boundaries

    full_text = "".join(c["text"] for c in text_chars)
    found = _find_vocab_words_in_string(full_text)
    if not found:
        return col_boundaries

    adjusted = list(col_boundaries)
    modified = False

    for word, start_idx, end_idx in found:
        if end_idx > len(text_chars):
            continue
        word_x0 = text_chars[start_idx]["x0"]
        word_x1 = text_chars[end_idx - 1]["x1"]

        for bi in range(1, len(adjusted) - 1):
            bx = adjusted[bi]
            if word_x0 + 1 < bx < word_x1 - 1:
                # Boundary falls inside a vocab word → shift past the word
                new_bx = word_x1 + 0.5
                logger.debug(
                    f"vocab boundary fix: {bx:.1f}\u2192{new_bx:.1f} "
                    f"to preserve '{word}'"
                )
                adjusted[bi] = new_bx
                modified = True
                break

    if modified:
        adjusted.sort()

    return adjusted


def detect_columns_by_whitespace_projection(
    page_plum,
) -> Optional[List[List[str]]]:
    """Vertical whitespace projection — project all rows onto the x-axis
    and detect column boundaries from whitespace bands.

    Algorithm:
        1. Collect all non-space characters, group into rows by y.
        2. For each x-position (1 pt resolution), count how many rows
           have text at that position.
        3. Positions with projection value <= 10 % of row count are "whitespace".
        4. Continuous whitespace bands >= 3 pt wide \u2192 column boundary (midpoint).
        5. Split each row's characters into cells using column boundaries.

    Suited for: borderless tables where columns are aligned by spacing.
    """
    chars = page_plum.chars
    if not chars or len(chars) < 20:
        return None

    # F-1: adaptive row-grouping tolerance
    from .utils import _adaptive_row_tolerance
    row_tol = _adaptive_row_tolerance(chars)

    # Collect non-space chars, group into rows by y (using adaptive tolerance)
    text_chars = [c for c in chars if c["text"].strip()]
    if not text_chars:
        return None
    sorted_chars = sorted(text_chars, key=lambda c: c["top"])
    y_rows: Dict[int, List] = {}
    current_yk = round(sorted_chars[0]["top"] / row_tol) * row_tol
    y_rows[current_yk] = [sorted_chars[0]]
    for c in sorted_chars[1:]:
        ck = round(c["top"] / row_tol) * row_tol
        if abs(c["top"] - current_yk) <= row_tol:
            y_rows.setdefault(current_yk, []).append(c)
        else:
            current_yk = ck
            y_rows.setdefault(current_yk, []).append(c)

    if len(y_rows) < 3:
        return None

    # x-coordinate range
    all_text_chars = [c for row in y_rows.values() for c in row]
    x_min = min(c["x0"] for c in all_text_chars)
    x_max = max(c["x1"] for c in all_text_chars)
    width = int(x_max - x_min) + 2
    if width < 20:
        return None

    # Build x-axis projection histogram
    row_count = len(y_rows)
    projection = [0] * width

    for row_chars in y_rows.values():
        marked = set()
        for c in row_chars:
            c_x0 = max(0, int(c["x0"] - x_min))
            c_x1 = min(width - 1, int(c["x1"] - x_min))
            for x in range(c_x0, c_x1 + 1):
                marked.add(x)
        for x in marked:
            projection[x] += 1

    # F-3: dynamic column-gap threshold (based on average character width)
    avg_char_w = sum(c["x1"] - c["x0"] for c in all_text_chars) / len(all_text_chars)
    min_gap_width = max(2.0, avg_char_w * 0.5)  # Minimum 2 pt or half a character width

    # Find whitespace bands: contiguous x-ranges with projection <= 10 % of row count
    threshold = row_count * 0.10
    gaps: List[Tuple[float, float, int]] = []
    in_gap = False
    gap_start = 0

    for x in range(width):
        if projection[x] <= threshold:
            if not in_gap:
                gap_start = x
                in_gap = True
        else:
            if in_gap:
                gap_width = x - gap_start
                if gap_width >= min_gap_width:  # F-3: dynamic threshold
                    gaps.append((gap_start + x_min, x - 1 + x_min, gap_width))
                in_gap = False
    # Handle trailing gap
    if in_gap:
        gap_width = width - gap_start
        if gap_width >= 3:
            gaps.append((gap_start + x_min, width - 1 + x_min, gap_width))

    if len(gaps) < 2:
        return None  # Need at least 2 gaps to define 3+ columns

    # Column boundaries = [x_min, gap1_mid, gap2_mid, ..., x_max]
    col_boundaries = [x_min]
    for g_start, g_end, _ in gaps:
        col_boundaries.append((g_start + g_end) / 2)
    col_boundaries.append(x_max + 1)

    n_cols = len(col_boundaries) - 1
    if n_cols < 3 or n_cols > 20:
        return None

    # Vocabulary-guided boundary correction: avoid splitting known header words
    first_yk = sorted(y_rows.keys())[0]
    header_chars = sorted(y_rows[first_yk], key=lambda c: c["x0"])
    col_boundaries = _adjust_boundaries_by_vocab(col_boundaries, header_chars)
    n_cols = len(col_boundaries) - 1  # Count may stay the same, positions shift

    # Split each row by column boundaries
    result: List[List[str]] = []
    for yk in sorted(y_rows.keys()):
        row_chars = sorted(y_rows[yk], key=lambda c: c["x0"])
        
        # 1. Merge adjacent characters into words (prevent words from being split mid-span)
        words = []
        curr_word = None
        for c in row_chars:
            if not str(c.get("text", "")).strip():
                continue
            if not curr_word:
                curr_word = {"x0": c["x0"], "x1": c.get("x1", c["x0"]), "text": c["text"]}
            else:
                gap = c["x0"] - curr_word["x1"]
                if gap < 2.5:
                    curr_word["x1"] = max(curr_word["x1"], c.get("x1", c["x0"]))
                    curr_word["text"] += c["text"]
                else:
                    words.append(curr_word)
                    curr_word = {"x0": c["x0"], "x1": c.get("x1", c["x0"]), "text": c["text"]}
        if curr_word:
            words.append(curr_word)

        cells: List[str] = []
        for i in range(n_cols):
            left = col_boundaries[i]
            right = col_boundaries[i + 1]
            # Assign to columns based on word centre point
            cell_words = [
                w for w in words
                if (w["x0"] + w["x1"]) / 2 >= left - 1
                and (w["x0"] + w["x1"]) / 2 < right + 1
            ]
            cell_text = " ".join(w["text"] for w in cell_words).strip()
            cells.append(cell_text)
        result.append(cells)

    logger.debug(
        f"whitespace_projection: {len(result)} rows, "
        f"{n_cols} cols from {len(gaps)} gaps"
    )

    if len(result) < 2:
        return None

    # ── Vocab scan: find the true header row, skip KV metadata rows ──
    # Some PDFs have KV metadata rows (e.g. "Account name: xxx") inside
    # the table zone before the actual header — these must be skipped
    best_header_idx = 0
    best_header_vs = 0
    scan_limit = min(15, len(result))
    for ri in range(scan_limit):
        vs = _score_header_by_vocabulary(result[ri])
        if vs > best_header_vs:
            best_header_vs = vs
            best_header_idx = ri

    if best_header_vs >= 3 and best_header_idx > 0:
        logger.info(
            f"whitespace_projection: header found at row {best_header_idx} "
            f"(vocab={best_header_vs}), skipping {best_header_idx} preamble rows"
        )
        result = result[best_header_idx:]

    return result if len(result) >= 2 else None


def detect_columns_by_clustering(page_plum) -> Optional[List[List[str]]]:
    """x-coordinate clustering method."""
    chars = page_plum.chars
    if not chars or len(chars) < 10:
        return None

    chars = [c for c in chars if not is_watermark_char(c)]
    if not chars:
        return None

    all_x0 = [c["x0"] for c in chars]
    col_bounds = _cluster_x_positions(all_x0, gap_multiplier=2.5)

    if len(col_bounds) < 2:
        return None

    rows_by_y = _group_chars_into_rows(chars)
    result: List[List[str]] = []
    for y_mid, row_chars in rows_by_y:
        row = _assign_chars_to_columns(row_chars, col_bounds)
        result.append(row)

    return result if len(result) >= 2 else None


def detect_columns_by_word_anchors(page_plum) -> Optional[List[List[str]]]:
    """Word-anchor column detection.

    Uses ``extract_words()`` to locate each header word's x-position as a
    column left boundary, then bins characters into columns at char level.

    Compared to char-level clustering, word-level gaps are more pronounced
    and can handle narrow multi-column layouts (e.g. columns with only
    8\u20139 pt spacing).
    """
    try:
        # Try a tighter x_tolerance first (distinguish 2\u20133 pt column gaps)
        # then the default \u2014 keep the result with more words (= finer column splits)
        best_words = None
        for x_tol in (2, 3):
            w = page_plum.extract_words(
                keep_blank_chars=True, x_tolerance=x_tol
            )
            if w and (best_words is None or len(w) > len(best_words)):
                best_words = w
        words = best_words
    except Exception as exc:
        logger.debug(f"operation: suppressed {exc}")
        return None
    if not words or len(words) < 5:
        return None

    # ── Group words into rows by y-coordinate ──
    ROW_TOL = 5
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    word_rows: List[Tuple[float, List[Dict]]] = []
    cur_y = sorted_words[0]["top"]
    cur_row = [sorted_words[0]]
    for w in sorted_words[1:]:
        if abs(w["top"] - cur_y) <= ROW_TOL:
            cur_row.append(w)
        else:
            y_mid = sum(ww["top"] for ww in cur_row) / len(cur_row)
            word_rows.append((y_mid, sorted(cur_row, key=lambda x: x["x0"])))
            cur_row = [w]
            cur_y = w["top"]
    if cur_row:
        y_mid = sum(ww["top"] for ww in cur_row) / len(cur_row)
        word_rows.append((y_mid, sorted(cur_row, key=lambda x: x["x0"])))

    if len(word_rows) < 2:
        return None

    # ── Find header row: the row in the first 5 that looks most like a header ──
    header_row_idx = -1
    for i, (y_mid, rw) in enumerate(word_rows[:5]):
        texts = [w["text"].strip() for w in rw if w["text"].strip()]
        if len(texts) < 3:
            continue
        header_count = sum(1 for t in texts if _is_header_cell(t))
        if header_count / len(texts) >= 0.5:
            header_row_idx = i
            break

    if header_row_idx == -1:
        return None

    header_words = word_rows[header_row_idx][1]
    if len(header_words) < 3:
        return None

    # ── Build column boundaries from header word positions ──
    # Each word's x0, x1 become boundaries; _assign_chars_to_columns places splits in the gaps
    col_bounds: List[Tuple[float, float]] = []
    for i, w in enumerate(header_words):
        x_start = w["x0"]
        x_end = w.get("x1", w["x0"] + 10)
        col_bounds.append((x_start, x_end))

    if len(col_bounds) < 3:
        return None

    # ── Extract data at character level using the column bins ──
    chars = page_plum.chars
    if not chars:
        return None
    chars = [c for c in chars if not is_watermark_char(c)]
    if not chars:
        return None

    char_rows = _group_chars_into_rows(chars)

    # Start extracting from the header row's y-position
    header_y = word_rows[header_row_idx][0]
    result: List[List[str]] = []
    for y_mid, row_chars in char_rows:
        if y_mid < header_y - 3:
            continue
        row = _assign_chars_to_columns(row_chars, col_bounds)
        result.append(row)

    if len(result) < 2:
        return None

    logger.info(
        f"word-anchors: {len(result)-1} data rows, "
        f"{len(col_bounds)} cols from {len(header_words)} header words"
    )
    return result


def detect_columns_by_data_voting(
    page_plum,
) -> Optional[List[List[str]]]:
    """Data-row-driven column boundary detection.

    Uses gap positions from data rows (rows containing dates / amounts)
    to vote on column boundaries.  More robust than header-anchors:
    no dependency on header row, handles bilingual mixed headers.
    """
    try:
        words = page_plum.extract_words(
            keep_blank_chars=True, x_tolerance=2
        )
    except Exception as exc:
        logger.debug(f"operation: suppressed {exc}")
        return None
    if not words or len(words) < 10:
        return None

    # ── Group words into rows by y-coordinate ──
    ROW_TOL = 5
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    word_rows: List[Tuple[float, List[Dict]]] = []
    cur_y = sorted_words[0]["top"]
    cur_row = [sorted_words[0]]
    for w in sorted_words[1:]:
        if abs(w["top"] - cur_y) <= ROW_TOL:
            cur_row.append(w)
        else:
            y_mid = sum(ww["top"] for ww in cur_row) / len(cur_row)
            word_rows.append(
                (y_mid, sorted(cur_row, key=lambda x: x["x0"]))
            )
            cur_row = [w]
            cur_y = w["top"]
    if cur_row:
        y_mid = sum(ww["top"] for ww in cur_row) / len(cur_row)
        word_rows.append(
            (y_mid, sorted(cur_row, key=lambda x: x["x0"]))
        )

    if len(word_rows) < 5:
        return None

    # ── Filter data rows: rows containing dates or amounts ──
    data_rows: List[Tuple[float, List[Dict]]] = []
    for y_mid, rw in word_rows:
        texts = " ".join(w["text"] for w in rw)
        if _RE_IS_DATE.search(texts):
            data_rows.append((y_mid, rw))
        elif any(
            _RE_IS_AMOUNT.match(
                w["text"].strip().replace(",", "").replace("\u00a5", "")
            )
            for w in rw
            if w["text"].strip()
        ):
            data_rows.append((y_mid, rw))

    if len(data_rows) < 3:
        return None

    # ── Collect gap midpoint positions (3 pt resolution) ──
    gap_votes: Dict[int, int] = defaultdict(int)
    page_w = page_plum.width or 600
    for _, rw in data_rows[:30]:
        for i in range(len(rw) - 1):
            gap_left = rw[i]["x1"]
            gap_right = rw[i + 1]["x0"]
            if gap_right - gap_left < 3:
                continue  # Too narrow \u2014 not a column gap
            gap_mid = (gap_left + gap_right) / 2
            bucket = round(gap_mid / 3) * 3
            gap_votes[bucket] += 1

    if not gap_votes:
        return None

    # ── Voting: gaps present in >= 40 % of data rows \u2192 column boundary ──
    n_voters = min(len(data_rows), 30)
    threshold = max(3, int(n_voters * 0.4))
    voted_gaps = sorted(
        x for x, count in gap_votes.items() if count >= threshold
    )

    if len(voted_gaps) < 2:
        return None

    # ── Merge adjacent gaps (< 8 pt \u2192 same boundary) ──
    merged_gaps: List[float] = [voted_gaps[0]]
    for g in voted_gaps[1:]:
        if g - merged_gaps[-1] < 8:
            merged_gaps[-1] = (merged_gaps[-1] + g) / 2
        else:
            merged_gaps.append(g)

    # ── Build column boundaries from gap midpoints ──
    col_bounds: List[Tuple[float, float]] = []
    col_bounds.append((0, merged_gaps[0]))
    for i in range(len(merged_gaps) - 1):
        col_bounds.append((merged_gaps[i], merged_gaps[i + 1]))
    col_bounds.append((merged_gaps[-1], page_w))

    if len(col_bounds) < 3:
        return None

    # ── Extract all rows at character level using column bins ──
    chars = page_plum.chars
    if not chars:
        return None
    chars = [c for c in chars if not is_watermark_char(c)]
    if not chars:
        return None

    char_rows = _group_chars_into_rows(chars)
    result: List[List[str]] = []
    for _, row_chars in char_rows:
        row = _assign_chars_to_columns(row_chars, col_bounds)
        result.append(row)

    if len(result) < 3:
        return None

    logger.info(
        f"data-voting: {len(result)} rows, "
        f"{len(col_bounds)} cols from "
        f"{len(data_rows)} data rows, "
        f"{len(merged_gaps)} voted gaps"
    )
    return result
