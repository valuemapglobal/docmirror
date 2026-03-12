"""Layered table extraction engine — main entry point.

Split from ``table_extraction.py``.
"""
from __future__ import annotations


import concurrent.futures
import logging
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from ...utils.text_utils import _is_cjk_char, _smart_join
from ...utils.vocabulary import _ALL_BORDER_CHARS, _is_header_row, _is_header_cell, _normalize_for_vocab, _score_header_by_vocabulary, _RE_IS_DATE, _RE_IS_AMOUNT

logger = logging.getLogger(__name__)


from .pipe_strategy import _extract_by_pipe_delimited
from .pdfplumber_strategy import _recover_header_from_zone
from .classifier import TABLE_SETTINGS, TABLE_SETTINGS_LINES, _quick_classify, _compute_table_confidence, _tables_look_valid, _cell_is_stuffed, _layer_timings_var
from .char_strategy import (
    _extract_by_hline_columns, _extract_by_rect_columns,
    detect_columns_by_header_anchors, detect_columns_by_whitespace_projection,
    detect_columns_by_clustering, detect_columns_by_word_anchors,
    detect_columns_by_data_voting,
)

def extract_tables_layered(
    page_plum,
    table_zone_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[List[List[List[str]]], str, float]:
    """Progressively layered table extraction.

    Optimisation highlights:
      - Pre-classification (``_quick_classify``): skip unlikely layers based
        on quick page features.
      - Per-layer timing: results stored in ``_layer_timings_var``.
      - Layer 2 parallel execution: 4 char-level methods run concurrently
        via ``ThreadPoolExecutor``.
      - Confidence scoring: returns a 0–1 float combining vocab_score,
        row_count, and col_consistency.

    Args:
        page_plum: pdfplumber page object.
        table_zone_bbox: Optional ``(x0, y0, x1, y1)`` bounding box of the
            table zone.  All layers operate on the cropped page to avoid
            extracting metadata tables or title text outside the zone.

    Returns:
        ``(tables, layer_label, confidence)`` 3-tuple.
    """
    timings: Dict[str, float] = {}
    t_total = time.time()

    def _t(label: str, t0: float):
        """Record per-layer elapsed time (ms)."""
        timings[label] = round((time.time() - t0) * 1000, 2)

    def _return(tables, layer):
        """Unified return: compute confidence and record total time."""
        timings["total"] = round((time.time() - t_total) * 1000, 2)
        _layer_timings_var.set(dict(timings))
        conf = _compute_table_confidence(tables, layer)
        logger.debug(
            f"extract_tables_layered -> layer={layer} conf={conf:.3f} "
            f"timings={timings}"
        )
        return tables, layer, conf

    # ── Crop to table zone (all layers work on the cropped page) ──
    work_page = page_plum
    if table_zone_bbox:
        try:
            x0, y0, x1, y1 = table_zone_bbox

            # Upward header probe: header row may sit above the data_table zone
            probe_top = max(0, y0 - 40)
            if probe_top < y0:
                try:
                    probe = page_plum.crop((x0, probe_top, x1, y0 + 1))
                    probe_words = probe.extract_words(
                        keep_blank_chars=True, x_tolerance=2
                    )
                    if probe_words:
                        # Group by y into rows, search bottom-up for the first table header row
                        from collections import defaultdict
                        _probe_rows = defaultdict(list)
                        for w in probe_words:
                            yk = round(w["top"] / 3) * 3
                            _probe_rows[yk].append(w)

                        sorted_yks = sorted(_probe_rows, reverse=True)
                        for idx_yk, yk in enumerate(sorted_yks):
                            texts = [
                                w["text"].strip()
                                for w in _probe_rows[yk]
                                if w["text"].strip()
                            ]
                            if len(texts) < 3:
                                continue
                            # Exclude KV metadata rows (e.g. "Customer name: ...")
                            kv_count = sum(
                                1 for t in texts
                                if ":" in t or "：" in t
                            )
                            if kv_count / len(texts) >= 0.5:
                                continue
                            hdr_count = sum(
                                1 for t in texts
                                if _is_header_cell(t)
                            )
                            if hdr_count / len(texts) >= 0.5:
                                header_y = min(
                                    w["top"] for w in _probe_rows[yk]
                                ) - 2

                                # ── Optimisation 2: multi-row header detection ──
                                if idx_yk + 1 < len(sorted_yks):
                                    prev_yk = sorted_yks[idx_yk + 1]
                                    prev_texts = [
                                        w["text"].strip()
                                        for w in _probe_rows[prev_yk]
                                        if w["text"].strip()
                                    ]
                                    row_gap = yk - prev_yk
                                    if (len(prev_texts) >= 2
                                            and row_gap < 20
                                            and not any(_RE_IS_DATE.search(t) for t in prev_texts)
                                            and not any(_RE_IS_AMOUNT.match(t.replace(",", "")) for t in prev_texts)):
                                        prev_hdr = sum(1 for t in prev_texts if _is_header_cell(t))
                                        if prev_hdr / len(prev_texts) >= 0.5:
                                            header_y = min(
                                                w["top"] for w in _probe_rows[prev_yk]
                                            ) - 2
                                            logger.debug(
                                                f"header probe: multi-row header detected, "
                                                f"expanded to {header_y:.0f}"
                                            )

                                y0 = max(0, header_y)
                                logger.debug(
                                    f"header probe: "
                                    f"expanded zone top "
                                    f"from {table_zone_bbox[1]:.0f}"
                                    f" to {y0:.0f}"
                                )
                                break
                except Exception as exc:
                    logger.debug(f"operation: suppressed {exc}")

            crop_x0 = 0
            crop_x1 = page_plum.width
            work_page = page_plum.crop((crop_x0, y0, crop_x1, y1))
            logger.debug(
                f"cropped to table zone: x={crop_x0:.0f}-{crop_x1:.0f}, y={y0:.0f}-{y1:.0f}"
            )
        except Exception as e:
            logger.debug(f"crop failed: {e}")

    # ── Pre-classification: determine starting layer from quick features ──
    t0 = time.time()
    classify_hint = _quick_classify(work_page)
    _t("pre_classify", t0)
    logger.debug(f"pre-classify hint: {classify_hint}")

    # ── Detect vertical border lines: bordered tables skip stuffed-cell check ──
    _lines = work_page.lines or []
    _v_line_count = sum(1 for l in _lines if abs(l.get("x0", 0) - l.get("x1", 0)) < 1)
    has_borders = _v_line_count >= 2

    # ── Segmented vertical lines → implicit row boundaries ──
    # When vertical lines are segmented per row (same x, many short lines)
    # and horizontal lines are insufficient, extract implicit row boundary
    # y-coordinates from vertical-line endpoints.
    _h_line_count = sum(1 for l in _lines if abs(l.get("top", 0) - l.get("bottom", 0)) < 1)
    _segmented_h_lines = None
    if _v_line_count >= 10 and _h_line_count < 10:
        from collections import Counter
        # Count how many vertical lines share each x-position
        _v_x_counts = Counter(round(l["x0"], 0) for l in _lines
                              if abs(l.get("x0", 0) - l.get("x1", 0)) < 1)
        # If the most common x has > 3 segments, vertical lines are row-segmented
        _max_segs = _v_x_counts.most_common(1)[0][1] if _v_x_counts else 0
        if _max_segs > 3:
            _y_set = set()
            for l in _lines:
                if abs(l.get("x0", 0) - l.get("x1", 0)) < 1:
                    _y_set.add(round(l["top"], 1))
                    _y_set.add(round(l["bottom"], 1))
            # Also include existing horizontal line y-values
            for l in _lines:
                if abs(l.get("top", 0) - l.get("bottom", 0)) < 1:
                    _y_set.add(round(l["top"], 1))
            _segmented_h_lines = sorted(_y_set)
            logger.debug(
                f"segmented v_lines → {len(_segmented_h_lines)} "
                f"implicit row boundaries (max_segs={_max_segs})"
            )

    # ── Layer 0.5: pipe separator (mainframe ASCII art) ──
    t0 = time.time()
    pipe_table = _extract_by_pipe_delimited(work_page)
    _t("L0.5_pipe", t0)
    if pipe_table and len(pipe_table) >= 3:
        return _return([pipe_table], "pipe_delimited")

    # Pre-classification jump: if hint='char', skip Layers 1–1.8
    if classify_hint != "char":
        # ── Layer 1: lines strategy ──
        t0 = time.time()
        if _segmented_h_lines:
            settings = dict(TABLE_SETTINGS_LINES)
            settings["explicit_horizontal_lines"] = _segmented_h_lines
            tables = work_page.extract_tables(table_settings=settings)
        else:
            tables = work_page.extract_tables(table_settings=TABLE_SETTINGS_LINES)
        _t("L1_lines", t0)
        if tables and _tables_look_valid(tables, has_borders=has_borders):
            return _return(
                _recover_header_from_zone(tables, work_page, table_zone_bbox, page_plum),
                "lines",
            )

        # Pre-classification jump: if hint='text', skip L1a/L1.5 to L1b
        # Also skip when many h_lines but v_lines=0 (L1a hline_columns is poor,
        # L1b TEXT is better — e.g. 192 h_lines + 0 v_lines)
        _skip_l1a = (classify_hint == "text") or (
            _h_line_count >= 20 and _v_line_count == 0
        )
        if not _skip_l1a:
            # ── Layer 1a: horizontal-line column boundary method ──
            t0 = time.time()
            table = _extract_by_hline_columns(work_page)
            _t("L1a_hline", t0)
            if table and len(table) >= 3 and _tables_look_valid([table], has_borders=has_borders):
                # Header quality check: if the first row looks like data (contains date), reject L1a
                _first_cell = (table[0][0] or "").strip()
                _header_looks_like_data = bool(
                    re.match(r"^\d{4}[-./]\d{2}[-./]\d{2}", _first_cell)
                    or re.match(r"^\d{8}$", _first_cell)
                )
                logger.debug(
                    f"L1a header check: first_cell={_first_cell!r} "
                    f"looks_like_data={_header_looks_like_data}"
                )
                if not _header_looks_like_data:
                    return _return(
                        _recover_header_from_zone([table], work_page, table_zone_bbox, page_plum),
                        "hline_columns",
                    )
                else:
                    logger.info(f"L1a rejected: header looks like data row")

            # ── Layer 1.5: rectangle column boundary method ──
            has_header_only = (
                tables and any(
                    t and len(t) == 1 and len(t[0]) >= 3
                    for t in tables
                )
            )
            if has_header_only:
                t0 = time.time()
                table = _extract_by_rect_columns(work_page)
                _t("L1.5_rect", t0)
                if table and len(table) >= 3:
                    return _return(
                        _recover_header_from_zone([table], work_page, table_zone_bbox, page_plum),
                        "rect_columns",
                    )

        # ── Layer 1b: text strategy ──
        t0 = time.time()
        tables = work_page.extract_tables(table_settings=TABLE_SETTINGS)
        _t("L1b_text", t0)
        if tables and _tables_look_valid(tables, has_borders=has_borders):
            return _return(
                _recover_header_from_zone(tables, work_page, table_zone_bbox, page_plum),
                "text",
            )

    # ── Layer 0.9: pdfplumber safety net (when L1 is skipped or all layers fail) ──
    # Try default extract_tables() first, then TABLE_SETTINGS (text strategy).
    t0 = time.time()
    tables = work_page.extract_tables()
    _t("L0.9_default", t0)
    if tables and _tables_look_valid(tables, has_borders=has_borders):
        return _return(
            _recover_header_from_zone(tables, work_page, table_zone_bbox, page_plum),
            "pdfplumber_default",
        )

    t0 = time.time()
    tables = work_page.extract_tables(table_settings=TABLE_SETTINGS)
    _t("L0.9_text", t0)
    if tables and _tables_look_valid(tables, has_borders=has_borders):
        return _return(
            _recover_header_from_zone(tables, work_page, table_zone_bbox, page_plum),
            "text_fallback",
        )

    # (RapidTable at L2.5 — too slow for early pipeline, ~10s/page CPU)

    # ── Layer 2: char-level competitive selection (parallel execution) ──
    t0 = time.time()

    def _run_method(name, func, wp):
        """Run a char-level method in a thread; return (table, name, score) or None."""
        try:
            tbl = func(wp)
            if tbl and len(tbl) >= 2:
                score = _score_header_by_vocabulary(tbl[0])
                return (tbl, name, score)
        except Exception as ex:
            logger.debug(f"L2 {name} error: {ex}")
        return None

    methods = [
        ("header_anchors", detect_columns_by_header_anchors),
        ("word_anchors", detect_columns_by_word_anchors),
        ("data_voting", detect_columns_by_data_voting),
        ("whitespace_projection", detect_columns_by_whitespace_projection),
    ]

    candidates: List[Tuple[List[List[str]], str, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_method, name, func, work_page): name
            for name, func in methods
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                candidates.append(result)

    _t("L2_char_level", t0)

    if candidates:
        # Penalty: if extracted table has many stuffed cells, lower its priority
        def _get_sort_key(c):
            tbl = c[0]
            vocab_score = c[2]
            row_count = len(tbl)
            stuffed_count = sum(
                1 for row in tbl[:10] for cell in row if _cell_is_stuffed(str(cell or ""))
            )
            return (vocab_score, row_count - stuffed_count * 10)

        candidates.sort(key=_get_sort_key, reverse=True)
        best_table, best_layer, best_score = candidates[0]
        if best_score >= 3:
            return _return([best_table], best_layer)
        _layer2_fallback = (best_table, best_layer)
    else:
        _layer2_fallback = None

    # ── Layer 2.5: RapidTable vision model (slow ~10 s, only when L2 also fails) ──
    t0 = time.time()
    rapid_result = _extract_by_rapid_table(page_plum)
    _t("L2.5_rapid_table", t0)
    if rapid_result and len(rapid_result) >= 2:
        rt_vocab = _score_header_by_vocabulary(rapid_result[0])
        if rt_vocab >= 2:  # At least 2 header vocabulary matches
            return _return([rapid_result], "rapid_table")

    # ── Layer 3: x-coordinate clustering ──
    t0 = time.time()
    table = detect_columns_by_clustering(work_page)
    _t("L3_clustering", t0)
    if table and len(table) >= 2:
        return _return([table], "x_clustering")

    # Layer 2 low-score candidate fallback (still better than pdfplumber default)
    if _layer2_fallback:
        return _return([_layer2_fallback[0]], _layer2_fallback[1])

    return _return(page_plum.extract_tables() or [], "fallback")


def _extract_by_rapid_table(page_plum) -> Optional[List[List[str]]]:
    """L1.8: table structure extraction using RapidTable ONNX vision model.

    RapidTable is a dedicated table-structure recognition model (CPU ONNX v3)
    that excels at borderless tables, three-line tables, and complex headers.
    Uses a singleton engine to avoid reloading the model.

    Returns:
        2-D table list, or ``None`` when not installed / recognition fails.
    """
    from .rapid_table_engine import get_rapid_table_engine

    engine = get_rapid_table_engine()
    if not engine.is_available:
        return None

    try:
        import numpy as np

        # Render pdfplumber page to image (200 DPI)
        img = page_plum.to_image(resolution=200)
        img_np = np.array(img.original)

        # Call RapidTable v3
        result = engine(img_np)
        if result is None or not result.pred_htmls:
            return None

        html_str = result.pred_htmls[0]
        if not html_str:
            return None

        # Parse HTML → 2-D array
        return _parse_html_table(html_str)

    except Exception as e:
        logger.debug(f"RapidTable error: {e}")
        return None


def _parse_html_table(html_str: str) -> Optional[List[List[str]]]:
    """Parse RapidTable HTML output into a 2-D array with colspan/rowspan support."""
    try:
        import re as _re

        row_pattern = _re.compile(r"<tr>(.*?)</tr>", _re.DOTALL)
        cell_pattern = _re.compile(r"(<t[dh][^>]*>)(.*?)</t[dh]>", _re.DOTALL)
        tag_cleaner = _re.compile(r"<[^>]+>")

        raw_rows = []  # [(col_idx, text, colspan, rowspan), ...] per row
        for row_match in row_pattern.finditer(html_str):
            cells = []
            for cell_match in cell_pattern.finditer(row_match.group(1)):
                tag = cell_match.group(1)
                text = tag_cleaner.sub("", cell_match.group(2)).strip()

                colspan_m = _re.search(r'colspan="(\d+)"', tag)
                rowspan_m = _re.search(r'rowspan="(\d+)"', tag)
                colspan = int(colspan_m.group(1)) if colspan_m else 1
                rowspan = int(rowspan_m.group(1)) if rowspan_m else 1

                cells.append((text, colspan, rowspan))
            if cells:
                raw_rows.append(cells)

        if len(raw_rows) < 2:
            return None

        # Expand colspan/rowspan into a 2-D grid
        grid: list = []  # List[List[str]]
        carry: dict = {}  # {col_idx: (text, remaining_rowspan)}

        for raw_cells in raw_rows:
            row_out: list = []
            col_idx = 0

            cell_iter = iter(raw_cells)
            current_cell = next(cell_iter, None)

            while current_cell is not None or col_idx in carry:
                # Fill in rowspan carry-over from previous rows
                if col_idx in carry:
                    text, remaining = carry[col_idx]
                    row_out.append(text)
                    if remaining > 1:
                        carry[col_idx] = (text, remaining - 1)
                    else:
                        del carry[col_idx]
                    col_idx += 1
                    continue

                if current_cell is None:
                    break

                text, colspan, rowspan = current_cell
                for ci in range(colspan):
                    actual_col = col_idx + ci
                    # Skip positions occupied by carry
                    while actual_col in carry:
                        ct, cr = carry[actual_col]
                        row_out.append(ct)
                        if cr > 1:
                            carry[actual_col] = (ct, cr - 1)
                        else:
                            del carry[actual_col]
                        actual_col += 1
                    row_out.append(text if ci == 0 else "")
                    if rowspan > 1:
                        carry[actual_col] = (text, rowspan - 1)

                col_idx = len(row_out)
                current_cell = next(cell_iter, None)

            # Process trailing carry entries
            while col_idx in carry:
                text, remaining = carry[col_idx]
                row_out.append(text)
                if remaining > 1:
                    carry[col_idx] = (text, remaining - 1)
                else:
                    del carry[col_idx]
                col_idx += 1

            grid.append(row_out)

        # Align column counts
        if grid:
            max_cols = max(len(r) for r in grid)
            for row in grid:
                while len(row) < max_cols:
                    row.append("")

        return grid if len(grid) >= 2 else None
    except Exception as exc:
        logger.debug(f"operation: suppressed {exc}")
        return None


def detect_merged_cells(
    page_plum,
    table_zone_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[Dict]:
    """P3-2: detect merged cells in a pdfplumber table.

    Uses pdfplumber's ``find_tables()`` API to get cell bounding boxes,
    then compares actual cell bboxes against an even grid to detect
    colspan / rowspan.

    Args:
        page_plum: pdfplumber page object.
        table_zone_bbox: Optional table-zone crop box.

    Returns:
        List of merged cells:
        ``[{"row": r, "col": c, "rowspan": rs, "colspan": cs}, ...]``
        Returns an empty list when none are detected.
    """
    try:
        work_page = page_plum
        if table_zone_bbox:
            try:
                x0, y0, x1, y1 = table_zone_bbox
                work_page = page_plum.crop((0, y0, page_plum.width, y1))
            except Exception as exc:
                logger.debug(f"operation: suppressed {exc}")

        tables = work_page.find_tables()
        if not tables or not tables[0].cells:
            return []

        cells = tables[0].cells  # List of (x0, y0, x1, y1)
        if len(cells) < 4:
            return []

        # Collect all unique x and y boundaries
        x_coords = sorted(set(round(c[0], 1) for c in cells) | set(round(c[2], 1) for c in cells))
        y_coords = sorted(set(round(c[1], 1) for c in cells) | set(round(c[3], 1) for c in cells))

        if len(x_coords) < 2 or len(y_coords) < 2:
            return []

        # Create grid row/column index mapping
        def _find_nearest_index(val, coords):
            best_idx = 0
            best_dist = abs(val - coords[0])
            for i, c in enumerate(coords[1:], 1):
                d = abs(val - c)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            return best_idx

        merged = []
        for cell_bbox in cells:
            cx0, cy0, cx1, cy1 = [round(v, 1) for v in cell_bbox]

            col_start = _find_nearest_index(cx0, x_coords)
            col_end = _find_nearest_index(cx1, x_coords)
            row_start = _find_nearest_index(cy0, y_coords)
            row_end = _find_nearest_index(cy1, y_coords)

            colspan = max(1, col_end - col_start)
            rowspan = max(1, row_end - row_start)

            if colspan > 1 or rowspan > 1:
                merged.append({
                    "row": row_start,
                    "col": col_start,
                    "rowspan": rowspan,
                    "colspan": colspan,
                })

        if merged:
            logger.debug(f"detected {len(merged)} merged cells")

        return merged

    except Exception as e:
        logger.debug(f"merged cell detection failed: {e}")
        return []


