# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Layout Analysis Engine
=======================================

Provides page-level layout analysis and spatial partitioning capabilities.
Self-contained, does not depend on any v1 code external to the MultiModal package.

=== Module Structure (post v2 refactor) ===

  This file only retains:
    - Module 1: Layout Analysis — ALPageLayout / analyze_page_layout / analyze_document_layout
    - Module 1b: Spatial Partitioning — Zone / segment_page_into_zones / _classify_zone

  Split into independent modules:
    - text_utils.py: CJK Tools / normalize_text / parse_amount / headers_match
    - vocabulary.py: VOCAB_BY_CATEGORY / KNOWN_HEADER_WORDS / Row Classifier
    - table_postprocess.py: post_process_table family
    - watermark.py: preprocess_document / filter_watermark_page / _dedup_overlapping_chars
    - table_extraction.py: extract_tables_layered (6+1 Layer) full pipeline
    - ocr_fallback.py: analyze_scanned_page (Scanned document OCR)

=== Backward Compatibility ===

  All historical public symbols are maintained backward compatible via re-export.
  Callers do not need to modify any import statements.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Backward compatible re-exports — Lazy Loading (Load on demand, avoids triggering 11+ module chain)
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping table: symbol_name -> (module_path, is_package_level)
_LAZY_MAP = {}


def _register_lazy(module: str, symbols: list):
    for s in symbols:
        _LAZY_MAP[s] = module


# text_utils
_register_lazy(
    "..utils.text_utils",
    [
        "_is_cjk_char",
        "_smart_join",
        "normalize_text",
        "normalize_table",
        "headers_match",
        "parse_amount",
    ],
)
# vocabulary & row classifiers
_register_lazy(
    "..utils.vocabulary",
    [
        "VOCAB_BY_CATEGORY",
        "KNOWN_HEADER_WORDS",
        "PIPE_CHARS",
        "HLINE_CHARS",
        "_ALL_BORDER_CHARS",
        "_RE_IS_DATE",
        "_RE_IS_AMOUNT",
        "_RE_VALID_DATE",
        "_normalize_for_vocab",
        "_score_header_by_vocabulary",
        "_is_header_cell",
        "_is_header_row",
        "_is_junk_row",
        "_is_data_row",
    ],
)
# table postprocess
_register_lazy(
    "..table.postprocess",
    [
        "_extract_preamble_kv",
        "_strip_preamble",
        "post_process_table",
        "_find_vocab_words_in_string",
        "_fix_header_by_vocabulary",
        "_clean_cell",
        "_merge_split_rows",
        "_extract_summary_entities",
    ],
)
# watermark & preprocessing
_register_lazy(
    "..utils.watermark",
    [
        "preprocess_document",
        "is_watermark_char",
        "filter_watermark_page",
        "_dedup_overlapping_chars",
    ],
)
# table extraction
_register_lazy(
    "..table.extraction",
    [
        "extract_tables_layered",
        "get_last_layer_timings",
        "_quick_classify",
        "_compute_table_confidence",
        "_tables_look_valid",
        "_cell_is_stuffed",
        "_recover_header_from_zone",
        "_extract_by_pipe_delimited",
        "_extract_by_hline_columns",
        "_extract_by_rect_columns",
        "detect_columns_by_header_anchors",
        "detect_columns_by_whitespace_projection",
        "detect_columns_by_clustering",
        "detect_columns_by_word_anchors",
        "detect_columns_by_data_voting",
    ],
)
# OCR fallback
_register_lazy("..ocr.fallback", ["analyze_scanned_page"])


def __getattr__(name):
    """Lazy loader for re-exported symbols — triggered only on first access."""
    if name in _LAZY_MAP:
        import importlib

        mod = importlib.import_module(_LAZY_MAP[name], package=__package__)
        val = getattr(mod, name)
        globals()[name] = val  # cache for subsequent accesses
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# Module 1: Layout Analyzer
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ContentRegion:
    """A content region on the page."""

    type: str  # "text" | "table" | "image"
    bbox: tuple[float, float, float, float]
    page: int
    text_preview: str = ""
    area: float = 0.0

    def __post_init__(self):
        x0, y0, x1, y1 = self.bbox
        self.area = max(0, (x1 - x0) * (y1 - y0))


@dataclass
class ALPageLayout:
    """Single page layout analysis result."""

    page_index: int
    width: float
    height: float
    regions: list[ContentRegion] = field(default_factory=list)
    has_table: bool = False
    table_count: int = 0
    image_count: int = 0
    text_region_count: int = 0
    is_continuation: bool = False
    is_scanned: bool = False
    header_text: str = ""
    footer_text: str = ""


def _detect_borderless_table(text_dict: dict, page_height: float) -> bool:
    """
    Heuristic detection of borderless tables.
    If >= 3 rows have >= 2 independent x segments -> determined as a borderless table.
    """
    spans = []
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                bbox = span["bbox"]
                spans.append(
                    {
                        "x0": bbox[0],
                        "x1": bbox[2],
                        "y_mid": (bbox[1] + bbox[3]) / 2,
                        "text": text,
                    }
                )

    if len(spans) < 6:
        return False

    spans.sort(key=lambda s: s["y_mid"])
    rows: list[list[dict]] = []
    current_row: list[dict] = [spans[0]]
    for s in spans[1:]:
        if abs(s["y_mid"] - current_row[-1]["y_mid"]) <= 3.0:
            current_row.append(s)
        else:
            rows.append(current_row)
            current_row = [s]
    rows.append(current_row)

    multi_col_rows = 0
    for row in rows:
        if len(row) < 2:
            continue
        row.sort(key=lambda s: s["x0"])
        segments = 1
        for i in range(1, len(row)):
            gap = row[i]["x0"] - row[i - 1]["x1"]
            if gap > 20:
                segments += 1
        if segments >= 2:
            multi_col_rows += 1

    return multi_col_rows >= 3


def analyze_page_layout(page, page_idx: int) -> ALPageLayout:
    """Analyze single page layout structure (~30ms/page)."""
    rect = page.rect
    layout = ALPageLayout(page_index=page_idx, width=rect.width, height=rect.height)

    text_dict = page.get_text("dict", flags=0)
    text_blocks = [b for b in text_dict.get("blocks", []) if b.get("type") == 0]
    image_blocks = [b for b in text_dict.get("blocks", []) if b.get("type") == 1]

    for b in text_blocks:
        bbox = (b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3])
        preview = ""
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                preview += span.get("text", "")
        preview = preview.strip()[:80]
        if preview:
            layout.regions.append(ContentRegion(type="text", bbox=bbox, page=page_idx, text_preview=preview))

    layout.text_region_count = len([r for r in layout.regions if r.type == "text"])

    for b in image_blocks:
        bbox = (b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3])
        layout.regions.append(
            ContentRegion(
                type="image",
                bbox=bbox,
                page=page_idx,
                text_preview=f"image_{b.get('width', 0)}x{b.get('height', 0)}",
            )
        )
    layout.image_count = len(image_blocks)

    # ── Fast table detection: line-count heuristic (~1ms vs ~2000ms) ──
    # The actual table extraction happens later in extract_tables_layered().
    # Here we only need a boolean has_table for routing decisions.
    try:
        drawings = page.get_drawings()
        v_lines = 0
        h_lines = 0
        for d in drawings:
            for item in d.get("items", []):
                if item[0] == "l":  # line item
                    p1, p2 = item[1], item[2]
                    dx = abs(p1.x - p2.x)
                    dy = abs(p1.y - p2.y)
                    if dx < 1 and dy > 5:
                        v_lines += 1
                    elif dy < 1 and dx > 5:
                        h_lines += 1
            if item[0] == "re":  # rectangle item → implies borders
                v_lines += 2
                h_lines += 2
        # Bordered table: has both vertical and horizontal lines
        if v_lines >= 2 and h_lines >= 2:
            layout.has_table = True
            layout.table_count = 1  # approximate; exact count not needed for routing
    except Exception as exc:
        logger.debug(f"fast table detection: suppressed {exc}")

    # Fallback: borderless table detection (existing heuristic)
    if not layout.has_table:
        if _detect_borderless_table(text_dict, rect.height):
            layout.has_table = True

    total_chars = sum(
        len(span.get("text", "")) for b in text_blocks for line in b.get("lines", []) for span in line.get("spans", [])
    )
    if total_chars < 50 and layout.image_count > 0:
        page_area = rect.width * rect.height
        for b in image_blocks:
            bx = b["bbox"]
            img_area = max(0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
            if img_area > page_area * 0.4:
                layout.is_scanned = True
                break

    layout.regions.sort(key=lambda r: r.bbox[1])

    if layout.has_table:
        table_regions = [r for r in layout.regions if r.type == "table"]
        if table_regions:
            table_top = min(r.bbox[1] for r in table_regions)
            table_bottom = max(r.bbox[3] for r in table_regions)
            layout.header_text = " | ".join(
                r.text_preview for r in layout.regions if r.type == "text" and r.bbox[3] <= table_top + 5
            )
            layout.footer_text = " | ".join(
                r.text_preview for r in layout.regions if r.type == "text" and r.bbox[1] >= table_bottom - 5
            )

    if layout.has_table:
        table_regions = [r for r in layout.regions if r.type == "table"]
        if table_regions:
            earliest_table_top = min(r.bbox[1] for r in table_regions)
            above_table_text = sum(
                1 for r in layout.regions if r.type == "text" and r.bbox[3] <= earliest_table_top + 5
            )
            layout.is_continuation = earliest_table_top < rect.height * 0.15 and above_table_text <= 2

    return layout


def analyze_document_layout(fitz_doc) -> list[ALPageLayout]:
    """Analyze the layout structure of the entire document."""
    layouts = []
    for page_idx in range(len(fitz_doc)):
        layouts.append(analyze_page_layout(fitz_doc[page_idx], page_idx))

    if layouts and layouts[0].is_continuation:
        layouts[0].is_continuation = False

    logger.info(
        f"{len(layouts)} pages: "
        + " | ".join(
            f"P{l.page_index + 1}({'cont' if l.is_continuation else 'new'}:"
            f"T{l.table_count}/I{l.image_count}/Txt{l.text_region_count})"
            for l in layouts
        )
    )
    return layouts


def _analyze_page_layout_worker(args: tuple[str, int]) -> tuple[int, ALPageLayout]:
    """
    Worker for process-pool layout analysis: open PDF at path, analyze one page, return (page_idx, ALPageLayout).
    Must be a top-level function for pickle; used by analyze_document_layout_parallel.
    """
    path, page_idx = args
    import fitz

    doc = fitz.open(path)
    try:
        page = doc[page_idx]
        layout = analyze_page_layout(page, page_idx)
        return (page_idx, layout)
    finally:
        doc.close()


def analyze_document_layout_parallel(
    path: str,
    num_pages: int,
    max_workers: int = 4,
) -> list[ALPageLayout]:
    """
    Analyze document layout in parallel across pages using a process pool.
    Each process opens the PDF at path and runs analyze_page_layout for one page.
    Use when max_page_concurrency > 1 to reduce layout stage time on multi-page documents.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    path = str(path)
    workers = min(max_workers, num_pages, os.cpu_count() or 4)
    if workers <= 1 or num_pages <= 1:
        # Fallback to sequential in caller by not using this path, or we could open and run here
        import fitz

        doc = fitz.open(path)
        try:
            layouts = [analyze_page_layout(doc[i], i) for i in range(num_pages)]
        finally:
            doc.close()
    else:
        args_list = [(path, i) for i in range(num_pages)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_analyze_page_layout_worker, args_list))
        layouts = [r[1] for r in sorted(results, key=lambda x: x[0])]

    if layouts and layouts[0].is_continuation:
        layouts[0].is_continuation = False

    logger.info(
        f"{len(layouts)} pages (parallel workers={workers}): "
        + " | ".join(
            f"P{l.page_index + 1}({'cont' if l.is_continuation else 'new'}:"
            f"T{l.table_count}/I{l.image_count}/Txt{l.text_region_count})"
            for l in layouts
        )
    )
    return layouts


# ═══════════════════════════════════════════════════════════════════════════════
# Module 1b: Spatial Partitioning
# ═══════════════════════════════════════════════════════════════════════════════


def _reconstruct_rows_from_chars(chars, col_gap: float = 8.0):
    """Fallback: Reconstruct table rows directly from chars."""
    if not chars:
        return []
    y_groups = defaultdict(list)
    for c in chars:
        y_key = round(c["top"] / 3) * 3
        y_groups[y_key].append(c)

    rows = []
    for y_key in sorted(y_groups.keys()):
        row_chars = sorted(y_groups[y_key], key=lambda c: c["x0"])

        def _chars_to_cell(cell_chars):
            if not cell_chars:
                return ""
            out = cell_chars[0]["text"]
            for j in range(1, len(cell_chars)):
                gap = cell_chars[j]["x0"] - cell_chars[j - 1]["x1"]
                if gap > 2.5:
                    out += " "
                out += cell_chars[j]["text"]
            return out.strip()

        cells = []
        current_cell = [row_chars[0]]
        for i in range(1, len(row_chars)):
            if row_chars[i]["x0"] - row_chars[i - 1]["x1"] > col_gap:
                cells.append(_chars_to_cell(current_cell))
                current_cell = [row_chars[i]]
            else:
                current_cell.append(row_chars[i])
        cells.append(_chars_to_cell(current_cell))
        if any(c for c in cells):
            rows.append(cells)
    return rows


@dataclass(slots=True)
class Zone:
    """A large zone on the page (3~5 zones/page)."""

    type: str  # "title" | "summary" | "data_table" | "footer" | "formula" | "unknown"
    bbox: tuple[float, float, float, float]
    page: int = 0
    chars: list = field(default_factory=list)
    rects: list = field(default_factory=list)
    text: str = ""
    confidence: float = 1.0  # Model Detection Confidence, rule method default 1.0


def _isolate_formula_components(chars: list[dict], page_w: float, page_h: float) -> tuple[list[dict], list[Zone]]:
    """
    Isolates formula regions using Union-Find connected component clustering
    of character bounding boxes. No cv2/numpy dependency required.

    Algorithm:
      1. Identify math seed characters (extreme aspect ratio or Unicode math symbols).
      2. Build dilated AABBs: seeds get +15pt horizontal, +8pt vertical expansion;
         all chars get a morphological close equivalent (+7.5pt H, +2.5pt V).
      3. Sort by Y, sweep-line merge overlapping AABBs via Union-Find.
      4. Connected components containing ≥1 math seed and <150 chars → formula zones.

    Returns: (remaining_chars, formula_zones)
    """
    if not chars or page_w <= 0 or page_h <= 0:
        return chars, []

    MATH_UNICODE = set("∑∫∏√∞∂∇±×÷≈≡≠≤≥⊂⊃⊆⊇∈∉∪∩")

    # ── Step 1: Identify math seed indices ──
    math_seed_set = set()
    for i, c in enumerate(chars):
        h = c.get("bottom", 0) - c.get("top", 0)
        w = c.get("x1", 0) - c.get("x0", 0)
        text = c.get("text", "").strip()

        if h > 0 and w > 0:
            aspect = h / w
            if aspect > 2.5 or aspect < 0.2:
                math_seed_set.add(i)
            elif text and text[0] in MATH_UNICODE:
                math_seed_set.add(i)

    if not math_seed_set:
        return chars, []

    # ── Step 2: Build dilated AABBs ──
    # Morph-close equivalent: expand every char slightly so adjacent
    # chars in the same formula merge; seeds get extra dilation.
    SEED_EXPAND_X, SEED_EXPAND_Y = 15.0, 8.0
    CLOSE_EXPAND_X, CLOSE_EXPAND_Y = 7.5, 2.5

    aabbs = []  # (x0, y0, x1, y1) per char — dilated
    for i, c in enumerate(chars):
        cx0, cy0 = c.get("x0", 0), c.get("top", 0)
        cx1, cy1 = c.get("x1", 0), c.get("bottom", 0)
        if i in math_seed_set:
            aabbs.append(
                (
                    cx0 - SEED_EXPAND_X,
                    cy0 - SEED_EXPAND_Y,
                    cx1 + SEED_EXPAND_X,
                    cy1 + SEED_EXPAND_Y,
                )
            )
        else:
            aabbs.append(
                (
                    cx0 - CLOSE_EXPAND_X,
                    cy0 - CLOSE_EXPAND_Y,
                    cx1 + CLOSE_EXPAND_X,
                    cy1 + CLOSE_EXPAND_Y,
                )
            )

    # ── Step 3: Union-Find with sweep-line AABB overlap ──
    n = len(chars)
    parent = list(range(n))
    rank = [0] * n

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Sort char indices by dilated y0 for sweep-line
    sorted_indices = sorted(range(n), key=lambda i: aabbs[i][1])

    # Sweep-line: for each char, check overlap with active chars
    # Active set: chars whose dilated y1 >= current char's dilated y0
    # Use a simple list scan (chars per page typically < 3000, fast enough)
    active = []  # list of indices currently in the sweep window
    for idx in sorted_indices:
        ax0, ay0, ax1, ay1 = aabbs[idx]

        # Prune expired active entries
        active = [j for j in active if aabbs[j][3] >= ay0]

        # Check overlap with each active entry
        for j in active:
            bx0, by0, bx1, by1 = aabbs[j]
            # AABB overlap test (Y already guaranteed by sweep)
            if ax0 <= bx1 and ax1 >= bx0:
                _union(idx, j)

        active.append(idx)

    # ── Step 4: Extract connected components with math seeds ──
    components: dict[int, list] = defaultdict(list)
    for i in range(n):
        components[_find(i)].append(i)

    formula_zones = []
    used_char_indices = set()

    for root, member_indices in components.items():
        # Must contain at least one math seed
        math_count = sum(1 for i in member_indices if i in math_seed_set)
        if math_count == 0:
            continue

        # Skip massive text blocks (> 150 chars → not a formula)
        if len(member_indices) >= 150:
            continue

        # Skip components spanning > 90% page width or > 50% page height
        comp_chars = [chars[i] for i in member_indices]
        fx0 = min(c["x0"] for c in comp_chars)
        fy0 = min(c["top"] for c in comp_chars)
        fx1 = max(c["x1"] for c in comp_chars)
        fy1 = max(c["bottom"] for c in comp_chars)

        if (fx1 - fx0) > page_w * 0.9 or (fy1 - fy0) > page_h * 0.5:
            continue

        used_char_indices.update(member_indices)
        ftext = "".join(c["text"] for c in sorted(comp_chars, key=lambda c: (c["top"], c["x0"])))
        formula_zones.append(
            Zone(
                type="formula",
                bbox=(fx0, fy0, fx1, fy1),
                chars=comp_chars,
                text=ftext.strip(),
                confidence=0.9,
            )
        )

    remaining_chars = [c for i, c in enumerate(chars) if i not in used_char_indices]
    return remaining_chars, formula_zones


def _column_consensus(
    chars: list[dict],
    page_w: float,
    page_h: float,
    min_cols: int = 3,
    min_rows: int = 3,
    cell_gap: float = 8.0,
    y_bin: float | None = None,
) -> tuple | None:
    """Column Consensus: detect table extent by structural repetition.

    Human visual pattern: a table is N rows where each row has M text
    segments at consistent X-positions.  Uses X-position clustering
    instead of quantization — no parameters to tune, no boundary effects.

    Algorithm:
      1. Group chars into rows by y_mid.
      2. Split each row into cells by x-gap.
      3. Collect ALL cell-start X positions → cluster by natural gaps.
      4. Map each row's cells to nearest column cluster.
      5. Rows with consistent column count → table rows.
      6. Return (table_y_top, table_y_bottom, column_x_positions).

    Returns:
        (y_top, y_bottom, [col_x_positions]) or None if no table found.
    """
    if not chars or page_w <= 0 or page_h <= 0:
        return None

    # ── Step 0: Bypass for explicitly bordered grid tables ──
    # Strongly bordered tables naturally compress whitespace within cells (e.g. 4pt gaps),
    # rendering spatial-gap clustering fundamentally unsuitable. If dense structural
    # borders are detected, immediately bypass spatial consensus.
    _ALL_BORDER_CHARS = globals().get("_ALL_BORDER_CHARS")
    if _ALL_BORDER_CHARS is None:
        _ALL_BORDER_CHARS = __getattr__("_ALL_BORDER_CHARS")

    border_chars_count = sum(1 for c in chars if c.get("text", "") in _ALL_BORDER_CHARS)
    if border_chars_count > 30:
        return None  # Defer to grid-aware fallback (legacy Y-band)

    # ── Step 1: Group chars into rows by y_mid ──
    # y_bin default: 3pt (proven for standard 12pt fonts).
    # Adaptive mode passes median_height * 0.4 for abnormal font sizes.
    _y_bin = y_bin if y_bin is not None else 3.0
    y_groups: dict[int, list] = defaultdict(list)
    for c in chars:
        y_mid = (c.get("top", 0) + c.get("bottom", 0)) / 2
        y_key = round(y_mid / _y_bin) * _y_bin
        y_groups[y_key].append(c)

    # ── Step 2: Split each row into cells by x-gap ──
    # Adaptive cell_gap based on median character width
    char_widths = [c.get("x1", 0) - c.get("x0", 0) for c in chars if c.get("x1", 0) > c.get("x0", 0)]
    if char_widths:
        sorted_w = sorted(char_widths)
        median_w = sorted_w[len(sorted_w) // 2]
        cell_gap = max(cell_gap, median_w * 1.5)

    row_cell_starts: dict[int, list] = {}  # y_key → [raw x_start, ...]

    for y_key in sorted(y_groups.keys()):
        row_chars = sorted(y_groups[y_key], key=lambda c: c.get("x0", 0))
        if len(row_chars) < 2:
            continue

        cell_starts = [row_chars[0]["x0"]]
        for i in range(1, len(row_chars)):
            gap = row_chars[i]["x0"] - row_chars[i - 1].get("x1", row_chars[i - 1]["x0"])
            if gap > cell_gap:
                cell_starts.append(row_chars[i]["x0"])

        if len(cell_starts) >= min_cols:
            row_cell_starts[y_key] = cell_starts

    if not row_cell_starts:
        return None

    # ── Step 3: Cluster all cell-start X positions ──
    # Collect every cell-start X from every multi-cell row.
    all_x = []
    for starts in row_cell_starts.values():
        all_x.extend(starts)
    all_x.sort()

    if not all_x:
        return None

    # Gap-based clustering: sort X values, split where gap > threshold.
    # Threshold = median of all inter-X gaps × 3 (large gaps = column breaks).
    inter_gaps = [all_x[i + 1] - all_x[i] for i in range(len(all_x) - 1)]
    if not inter_gaps:
        return None

    sorted_gaps = sorted(inter_gaps)
    median_inter_gap = sorted_gaps[len(sorted_gaps) // 2]
    # Cluster split threshold: gaps significantly larger than typical
    # within-cluster variation are column boundaries.
    cluster_threshold = max(median_inter_gap * 3, 15.0)

    clusters: list[list] = [[all_x[0]]]
    for i in range(1, len(all_x)):
        if all_x[i] - all_x[i - 1] > cluster_threshold:
            clusters.append([all_x[i]])
        else:
            clusters[-1].append(all_x[i])

    if len(clusters) < min_cols:
        return None

    # Compute cluster centers (median of each cluster)
    col_centers = []
    for cl in clusters:
        cl.sort()
        col_centers.append(cl[len(cl) // 2])

    logger.debug(
        f"column_consensus: {len(clusters)} column clusters from "
        f"{len(all_x)} X-positions, threshold={cluster_threshold:.1f}"
    )

    # ── Step 4: Map each row's cells to column clusters ──
    def _map_to_cols(cell_starts: list) -> tuple:
        """Map cell starts to closest column clusters → column ID tuple."""
        col_ids = []
        for x in cell_starts:
            best_col = min(range(len(col_centers)), key=lambda i: abs(x - col_centers[i]))
            col_ids.append(best_col)
        return tuple(col_ids)

    row_col_ids: dict[int, tuple] = {}
    for y_key, starts in row_cell_starts.items():
        col_ids = _map_to_cols(starts)
        row_col_ids[y_key] = col_ids

    # ── Step 5: Find consensus — best signature ──
    # A "signature" is now the tuple of column IDs the row's cells belong to.
    sig_counter: dict[tuple, list] = defaultdict(list)
    for y_key, col_ids in row_col_ids.items():
        sig_counter[col_ids].append(y_key)

    def _count_absorbable(candidate_sig: tuple) -> int:
        """Count rows absorbable by this signature using subset logic."""
        cn = len(candidate_sig)
        cand_set = set(candidate_sig)
        total_rows = list(sig_counter[candidate_sig])
        for other_sig, other_rows in sig_counter.items():
            if other_sig == candidate_sig:
                continue
            on = len(other_sig)
            other_set = set(other_sig)

            # Fewer cols: subset wrap. Every col in other MUST be in candidate.
            if on < cn and on >= min_cols:
                if other_set.issubset(cand_set):
                    total_rows.extend(other_rows)
            # More cols (+2 allowance): candidate cols MUST be in other.
            elif on > cn and on <= cn + 2:
                if cand_set.issubset(other_set):
                    total_rows.extend(other_rows)
        return len(set(total_rows))

    # Best signature: most absorbable rows weighted by column count
    # Deterministic tie-breaker: (score, ncols, signature_tuple)
    best_sig = max(sig_counter.keys(), key=lambda s: (_count_absorbable(s) * len(s), len(s), s))
    best_rows = list(sig_counter[best_sig])
    best_ncols = len(best_sig)
    best_set = set(best_sig)

    # Absorb all valid subset/superset rows into the best table extent
    for sig, sig_rows in sig_counter.items():
        if sig == best_sig:
            continue
        ncols = len(sig)
        sig_set = set(sig)
        if ncols < best_ncols and ncols >= min_cols:
            if sig_set.issubset(best_set):
                best_rows.extend(sig_rows)
        elif ncols > best_ncols and ncols <= best_ncols + 2:
            if best_set.issubset(sig_set):
                best_rows.extend(sig_rows)

    best_rows = sorted(set(best_rows))

    if len(best_rows) < min_rows:
        return None

    # ── Step 6: Determine table extent ──
    table_y_top = best_rows[0]
    table_y_bottom = best_rows[-1]

    # Extend y_bottom to include the full height of the last row's chars
    last_row_chars = y_groups.get(best_rows[-1], [])
    if last_row_chars:
        table_y_bottom = max(c.get("bottom", table_y_bottom) for c in last_row_chars)

    # Gap-fill: include ALL y-groups between first and last matched row.
    # Multi-line continuation rows (wrapped company names) sit between
    # data rows and must be included in the table extent.
    all_ys = sorted(y_groups.keys())
    for yk in all_ys:
        if table_y_top <= yk <= best_rows[-1]:
            max_bottom = max(c.get("bottom", 0) for c in y_groups[yk])
            table_y_bottom = max(table_y_bottom, max_bottom)

    # Extend past the last matched row for trailing continuation rows
    last_idx = all_ys.index(best_rows[-1]) if best_rows[-1] in all_ys else -1
    if last_idx >= 0:
        for check_idx in range(last_idx + 1, min(last_idx + 4, len(all_ys))):
            next_y = all_ys[check_idx]
            if next_y not in row_cell_starts:
                gap = next_y - best_rows[-1]
                if gap < 30:
                    max_bottom = max(c.get("bottom", 0) for c in y_groups[next_y])
                    table_y_bottom = max(table_y_bottom, max_bottom)
                else:
                    break
            else:
                break

    # Include header row: the row immediately above table_top
    top_idx = all_ys.index(best_rows[0]) if best_rows[0] in all_ys else -1
    if top_idx > 0:
        candidate_y = all_ys[top_idx - 1]
        if candidate_y in row_cell_starts:
            cand_ncols = len(row_cell_starts[candidate_y])
            if abs(cand_ncols - best_ncols) <= 2:
                table_y_top = candidate_y
        elif candidate_y in y_groups:
            gap = best_rows[0] - candidate_y
            if gap < 30 and len(y_groups[candidate_y]) >= 3:
                table_y_top = candidate_y

    logger.debug(
        f"column_consensus: found table y={table_y_top:.0f}-{table_y_bottom:.0f} "
        f"cols={best_ncols} rows={len(best_rows)} "
        f"centers={[round(c) for c in col_centers]}"
    )

    return (table_y_top, table_y_bottom, [round(c) for c in col_centers])


def _refine_by_lines(
    table_extent: tuple,
    lines: list | None,
    rects: list | None = None,
) -> tuple:
    """Refine table extent using drawing lines (Tier 2).

    If horizontal lines exist near the table boundary, snap the boundary
    to the line position for pixel-perfect accuracy.
    """
    y_top, y_bottom, col_xs = table_extent
    if not lines and not rects:
        return table_extent

    all_lines = list(lines or [])

    # Extract horizontal line Y positions
    h_line_ys = []
    for ln in all_lines:
        ly0 = ln.get("top", ln.get("y0", 0))
        ly1 = ln.get("bottom", ln.get("y1", 0))
        lx0 = ln.get("x0", 0)
        lx1 = ln.get("x1", 0)
        # Horizontal line: height < 2pt, width > 50pt
        if abs(ly1 - ly0) < 2 and abs(lx1 - lx0) > 50:
            h_line_ys.append((ly0 + ly1) / 2)

    if not h_line_ys:
        return table_extent

    h_line_ys.sort()

    # Snap y_top to nearest h-line above (within 15pt)
    for ly in h_line_ys:
        if y_top - 15 <= ly <= y_top + 5:
            y_top = ly
            break

    # Snap y_bottom to nearest h-line below (within 15pt)
    for ly in reversed(h_line_ys):
        if y_bottom - 5 <= ly <= y_bottom + 15:
            y_bottom = ly
            break

    return (y_top, y_bottom, col_xs)


def _build_zones_from_extent(
    chars: list[dict],
    rects: list,
    table_extent: tuple,
    page_w: float,
    page_h: float,
    page_idx: int,
) -> list[Zone]:
    """Derive all zones from a precise table extent.

    Zones:
      - chars above table_top → title / summary
      - chars within table_top to table_bottom → data_table
      - chars below table_bottom → footer
    """
    y_top, y_bottom, col_xs = table_extent
    zones = []

    # Partition chars into above / table / below
    above_chars = [c for c in chars if c.get("bottom", 0) <= y_top + 3]
    table_chars = [c for c in chars if c.get("top", 0) >= y_top - 3 and c.get("bottom", 0) <= y_bottom + 3]
    below_chars = [c for c in chars if c.get("top", 0) >= y_bottom - 3]

    # Remove overlap: a char should be in exactly one group
    table_set = set(id(c) for c in table_chars)
    above_chars = [c for c in above_chars if id(c) not in table_set]
    below_chars = [c for c in below_chars if id(c) not in table_set]

    # ── Above table → title and/or summary ──
    if above_chars:
        # Split above chars into title (large/centered) and summary (has "：")
        # Group by y-bands first
        above_y_groups: dict[int, list] = defaultdict(list)
        for c in above_chars:
            yk = round(c["top"] / 3) * 3
            above_y_groups[yk].append(c)

        title_chars = []
        summary_chars = []
        for yk in sorted(above_y_groups.keys()):
            band = above_y_groups[yk]
            band_text = "".join(c["text"] for c in sorted(band, key=lambda c: c["x0"]))
            # title: no "：" and generally a heading
            if re.search(r"[\u4e00-\u9fff][：:]", band_text):
                summary_chars.extend(band)
            elif not title_chars and len(band_text.strip()) < 80:
                # First non-KV band is the title
                title_chars.extend(band)
            else:
                summary_chars.extend(band)

        if title_chars:
            x0 = min(c["x0"] for c in title_chars)
            x1 = max(c["x1"] for c in title_chars)
            y0 = min(c["top"] for c in title_chars)
            y1 = max(c["bottom"] for c in title_chars)
            text = "".join(c["text"] for c in sorted(title_chars, key=lambda c: (c["top"], c["x0"])))
            zones.append(
                Zone(
                    type="title",
                    bbox=(x0, y0, x1, y1),
                    page=page_idx,
                    chars=title_chars,
                    text=text.strip(),
                )
            )

        if summary_chars:
            x0 = min(c["x0"] for c in summary_chars)
            x1 = max(c["x1"] for c in summary_chars)
            y0 = min(c["top"] for c in summary_chars)
            y1 = max(c["bottom"] for c in summary_chars)
            text = "".join(c["text"] for c in sorted(summary_chars, key=lambda c: (c["top"], c["x0"])))
            zones.append(
                Zone(
                    type="summary",
                    bbox=(x0, y0, x1, y1),
                    page=page_idx,
                    chars=summary_chars,
                    text=text.strip(),
                )
            )

    # ── Table zone ──
    if table_chars:
        x0 = min(c["x0"] for c in table_chars)
        x1 = max(c["x1"] for c in table_chars)
        table_rects = [r for r in rects if r.get("top", 0) >= y_top - 3 and r.get("top", 0) <= y_bottom + 3]
        text = "".join(c["text"] for c in sorted(table_chars, key=lambda c: (c["top"], c["x0"])))
        zones.append(
            Zone(
                type="data_table",
                bbox=(x0, y_top, x1, y_bottom),
                page=page_idx,
                chars=table_chars,
                rects=table_rects,
                text=text.strip(),
            )
        )

    # ── Below table → footer ──
    if below_chars:
        x0 = min(c["x0"] for c in below_chars)
        x1 = max(c["x1"] for c in below_chars)
        y0 = min(c["top"] for c in below_chars)
        y1 = max(c["bottom"] for c in below_chars)
        text = "".join(c["text"] for c in sorted(below_chars, key=lambda c: (c["top"], c["x0"])))
        zones.append(
            Zone(
                type="footer",
                bbox=(x0, y0, x1, y1),
                page=page_idx,
                chars=below_chars,
                text=text.strip(),
            )
        )

    return zones


def _legacy_y_band_zones(
    chars: list[dict],
    rects: list,
    page_w: float,
    page_h: float,
    page_idx: int,
    gap_threshold: float = 15.0,
) -> list[Zone]:
    """Legacy Y-band splitting fallback.

    Used when Column Consensus finds no table pattern.
    This is the original segment_page_into_zones logic preserved as fallback.
    """
    # Lazy load classify helper
    _PIPE_CHARS = globals().get("PIPE_CHARS")
    if _PIPE_CHARS is None:
        _PIPE_CHARS = __getattr__("PIPE_CHARS")
    _KNOWN_HEADER_WORDS = globals().get("KNOWN_HEADER_WORDS")
    if _KNOWN_HEADER_WORDS is None:
        _KNOWN_HEADER_WORDS = __getattr__("KNOWN_HEADER_WORDS")

    # Dynamic gap_threshold
    char_heights = [c["bottom"] - c["top"] for c in chars if c.get("bottom", 0) > c.get("top", 0)]
    if char_heights:
        sorted_h = sorted(char_heights)
        median_h = sorted_h[len(sorted_h) // 2]
        gap_threshold = max(12.0, median_h * 1.5)

    row_ys = sorted(set(round(c["top"] / 3) * 3 for c in chars))

    # Font-size change boundaries
    row_font_sizes: dict[int, float] = {}
    for y_key in row_ys:
        row_chars = [c for c in chars if round(c["top"] / 3) * 3 == y_key]
        sizes = [c.get("size", 0) for c in row_chars if c.get("size", 0) > 0]
        if sizes:
            sizes.sort()
            row_font_sizes[y_key] = sizes[len(sizes) // 2]

    # Pre-compute border character counts per y-key for structural cut detection
    _HLINE_CHARS = globals().get("HLINE_CHARS")
    if _HLINE_CHARS is None:
        _HLINE_CHARS = __getattr__("HLINE_CHARS")

    row_border_counts: dict[int, int] = {}
    for y_key in row_ys:
        row_chars = [c for c in chars if round(c["top"] / 3) * 3 == y_key]
        row_border_counts[y_key] = sum(
            1 for c in row_chars if c.get("text", "") in _PIPE_CHARS or c.get("text", "") in _HLINE_CHARS
        )

    cuts = [row_ys[0]]
    for i in range(1, len(row_ys)):
        y_gap = row_ys[i] - row_ys[i - 1]
        is_gap = y_gap > gap_threshold

        # structural cut: split when transitioning between bordered (table) and non-bordered (text) rows
        cur_has_borders = row_border_counts.get(row_ys[i], 0) >= 3
        prev_has_borders = row_border_counts.get(row_ys[i - 1], 0) >= 3
        is_structural_cut = cur_has_borders != prev_has_borders

        if not is_gap and not is_structural_cut and row_ys[i] in row_font_sizes and row_ys[i - 1] in row_font_sizes:
            if abs(row_font_sizes[row_ys[i]] - row_font_sizes[row_ys[i - 1]]) > 2.0:
                is_gap = True

        if is_gap or is_structural_cut:
            cuts.append(row_ys[i - 1])
            cuts.append(row_ys[i])
    cuts.append(row_ys[-1])

    bands = []
    for i in range(0, len(cuts) - 1, 2):
        bands.append((cuts[i], cuts[i + 1]))
    if not bands:
        bands = [(row_ys[0], row_ys[-1])]

    # Track which band start y-values came from structural cuts
    structural_cut_ys = set()
    for i in range(1, len(row_ys)):
        y_gap = row_ys[i] - row_ys[i - 1]
        is_gap = y_gap > gap_threshold
        cur_has = row_border_counts.get(row_ys[i], 0) >= 3
        prev_has = row_border_counts.get(row_ys[i - 1], 0) >= 3
        if cur_has != prev_has:
            structural_cut_ys.add(row_ys[i])

    zones = []
    for y_start, y_end in bands:
        margin = 5
        band_chars = [c for c in chars if y_start - margin <= c["top"] <= y_end + margin]
        band_rects = [r for r in rects if y_start - margin <= r["top"] <= y_end + margin]
        if not band_chars:
            continue
        x0 = min(c["x0"] for c in band_chars)
        x1 = max(c["x1"] for c in band_chars)
        text = "".join(c["text"] for c in sorted(band_chars, key=lambda c: (c["top"], c["x0"])))
        zone = Zone(
            type="unknown",
            bbox=(x0, y_start, x1, y_end),
            page=page_idx,
            chars=band_chars,
            rects=band_rects,
            text=text.strip(),
        )
        _zone_border_count = sum(
            1 for c in zone.chars if c.get("text", "") in _PIPE_CHARS or c.get("text", "") in _HLINE_CHARS
        )
        zone.type = _classify_zone_legacy(
            zone,
            page_h,
            _PIPE_CHARS,
            _KNOWN_HEADER_WORDS,
            is_border_zone=(_zone_border_count >= 10),
            has_structural_context=bool(structural_cut_ys),
        )
        zones.append(zone)

    # Merge adjacent data_table zones ONLY if not separated by a structural cut
    merged = []
    for z in zones:
        if merged and z.type == "data_table" and merged[-1].type == "data_table" and z.bbox[1] not in structural_cut_ys:
            prev = merged[-1]
            prev.bbox = (min(prev.bbox[0], z.bbox[0]), prev.bbox[1], max(prev.bbox[2], z.bbox[2]), z.bbox[3])
            prev.chars.extend(z.chars)
            prev.rects.extend(z.rects)
            prev.text += z.text
        else:
            merged.append(z)
    return merged


def _classify_zone_legacy(
    zone: Zone,
    page_h: float,
    pipe_chars: set,
    known_header_words: set,
    is_border_zone: bool = False,
    has_structural_context: bool = False,
) -> str:
    """Legacy zone classifier — used only by _legacy_y_band_zones fallback.

    Args:
        is_border_zone: True if this zone contains ≥10 structural border chars.
        has_structural_context: True when structural cuts exist on this page,
            meaning a bordered table was detected. Non-bordered zones adjacent
            to bordered tables are metadata, not tables.
    """
    y_ratio = zone.bbox[1] / page_h if page_h else 0
    text = zone.text
    char_count = len(zone.chars)

    if y_ratio > 0.85 and char_count < 30 and "页" in text:
        return "footer"

    # Bordered zone → always data_table
    if is_border_zone:
        return "data_table"

    # Non-bordered zone adjacent to a bordered table → metadata, not table
    if has_structural_context and not is_border_zone:
        if re.search(r"[\u4e00-\u9fff][：:]", text):
            return "summary"
        return "summary"

    pipe_count = sum(1 for c in zone.chars if c.get("text") in pipe_chars)
    if pipe_count >= 10:
        return "data_table"
    if bool(re.search(r"\d{8}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", text)) and bool(
        re.search(r"(?:RMB|USD|CNY)\s*[\d,.]+|\d+\.\d{2}", text)
    ):
        return "data_table"
    _vocab_hits = sum(1 for w in known_header_words if w in text)
    if _vocab_hits >= 3:
        return "data_table"
    if y_ratio < 0.15 and char_count < 80:
        if not re.search(r"[\u4e00-\u9fff][：:]", text):
            return "title"
    if char_count < 300 and re.search(r"[\u4e00-\u9fff][：:]", text):
        return "summary"
    row_ys = sorted(set(round(c["top"] / 3) * 3 for c in zone.chars))
    if len(row_ys) < 2 and not any(ch.isdigit() for ch in text):
        return "summary"
    x_positions = set(round(c["x0"] / 10) * 10 for c in zone.chars)
    if len(x_positions) >= 5 and char_count > 20:
        return "data_table"
    if len(zone.rects) >= 3:
        return "data_table"
    if len(row_ys) >= 3:
        return "data_table"
    return "unknown"


def segment_page_into_zones(
    page_plum,
    page_idx: int,
    gap_threshold: float = 15.0,
) -> list[Zone]:
    """Spatial partitioning: Column Consensus architecture.

    Primary path: detect table extent by structural column alignment
    (Column Consensus), then derive all zones from the table extent.

    Fallback: legacy Y-band splitting when no table pattern is found.
    """
    chars = page_plum.chars
    rects = page_plum.rects or []
    page_h = page_plum.height
    page_w = page_plum.width

    if not chars:
        return []

    # ── Step 1: Column Consensus on ALL chars (before formula isolation) ──
    # Formula isolation can eat table data characters, breaking column
    # alignment detection.  Run Column Consensus first on the full char set.
    # Competitive strategy: try proven 3pt binning first; if it fails,
    # retry with adaptive binning based on median character height.
    table_extent = _column_consensus(chars, page_w, page_h)
    if table_extent is None:
        char_heights = [
            c.get("bottom", 0) - c.get("top", 0) for c in chars if (c.get("bottom", 0) - c.get("top", 0)) > 0
        ]
        if char_heights:
            sorted_h = sorted(char_heights)
            median_h = sorted_h[len(sorted_h) // 2]
            adaptive_bin = max(2.0, median_h * 0.4)
            # Only retry if adaptive bin differs meaningfully from default 3pt
            if abs(adaptive_bin - 3.0) > 0.5:
                table_extent = _column_consensus(chars, page_w, page_h, y_bin=adaptive_bin)
                if table_extent is not None:
                    logger.debug(
                        f"column_consensus: adaptive y_bin={adaptive_bin:.1f}pt succeeded (median_h={median_h:.1f}pt)"
                    )

    # ── Step 2: Line enhancement (Tier 2) ──
    if table_extent:
        lines = page_plum.lines or []
        table_extent = _refine_by_lines(table_extent, lines, rects)

    # ── Step 2.5: Rect-grid table extent (Tier 1.5 fallback) ──
    # When column consensus fails but the page has dense rect grids (e.g.
    # 东莞银行: 165 rects forming 15 rows × 11 cols), derive table extent
    # from the rect bounding box.
    if not table_extent and len(rects) >= 20:
        x_lefts = sorted(set(round(r["x0"]) for r in rects))
        if len(x_lefts) >= 4:
            # Rects share ≥ 4 distinct left-edge x positions → grid structure
            y_min = min(r["top"] for r in rects)
            y_max = max(r["bottom"] for r in rects)
            if y_max - y_min > 50:  # At least 50pt tall
                table_extent = (y_min, y_max, x_lefts)
                logger.debug(
                    f"segment_page_into_zones: Rect-grid fallback → "
                    f"table extent y={y_min:.0f}-{y_max:.0f} "
                    f"({len(rects)} rects, {len(x_lefts)} x-cols)"
                )

    # ── Step 2.6: Line-grid table extent (Tier 1.5b fallback) ──
    # When rect-grid also fails but the page has a clear line grid (e.g.
    # 交通银行: 644 lines forming 13 rows × 18 cols), derive table extent
    # from the h-line/v-line bounding box.
    if not table_extent:
        lines = page_plum.lines or []
        if len(lines) >= 10:
            h_lines = [l for l in lines if abs(l["top"] - l["bottom"]) < 2]
            v_lines = [l for l in lines if abs(l["x0"] - l["x1"]) < 2]
            h_ys = sorted(set(round(l["top"]) for l in h_lines))
            v_xs = sorted(set(round(l["x0"]) for l in v_lines))
            if len(h_ys) >= 5 and len(v_xs) >= 4:
                y_min = min(h_ys)
                y_max = max(h_ys)
                if y_max - y_min > 50:
                    table_extent = (y_min, y_max, v_xs)
                    logger.debug(
                        f"segment_page_into_zones: Line-grid fallback → "
                        f"table extent y={y_min}-{y_max} "
                        f"({len(h_lines)} h-lines, {len(v_lines)} v-lines, "
                        f"{len(h_ys)} rows × {len(v_xs)} cols)"
                    )

    # ── Step 3: Build zones ──
    if table_extent:
        zones = _build_zones_from_extent(chars, rects, table_extent, page_w, page_h, page_idx)
        logger.debug(
            f"segment_page_into_zones: Column Consensus path → "
            f"{len(zones)} zones (table y={table_extent[0]:.0f}-{table_extent[1]:.0f})"
        )
    else:
        # Fallback: formula isolation + legacy Y-band splitting
        remaining_chars, formula_zones = _isolate_formula_components(chars, page_w, page_h)
        if remaining_chars:
            zones = _legacy_y_band_zones(remaining_chars, rects, page_w, page_h, page_idx, gap_threshold)
        else:
            zones = []
        zones.extend(formula_zones)
        logger.debug(f"segment_page_into_zones: Legacy fallback → {len(zones)} zones")

    # ── Step 4: GraphRouter reading order (preserved) ──
    from .graph_router import GraphRouter

    router = GraphRouter(page_width=page_w, page_height=page_h)
    causal_zones = router.build_flow(zones)

    return causal_zones


# ═══════════════════════════════════════════════════════════════════════════════
# Perf #9: Zone Template Reuse for Layout-Homogeneous Documents
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ZoneTemplate:
    """Cached zone layout template from a reference page.

    Stores normalized (0-1) bbox ratios so the template can be applied
    to pages of any DPI/dimension.
    """

    zones: list  # List of (zone_type, bbox_ratio, confidence)
    # bbox_ratio = (x0/page_w, y0/page_h, x1/page_w, y1/page_h)
    source_page: int = 0
    zone_count: int = 0


def build_zone_template(zones: list[Zone], page_w: float, page_h: float, page_idx: int = 0) -> ZoneTemplate:
    """Build a reusable zone template from a fully-segmented page.

    Captures the zone types and their normalized bbox positions.
    Called once on page 0, result reused for pages 1~N.

    Args:
        zones: Zone list from segment_page_into_zones.
        page_w: Page width in points.
        page_h: Page height in points.
        page_idx: Source page index.

    Returns:
        ZoneTemplate with normalized coordinates.
    """
    template_zones = []
    for z in zones:
        x0, y0, x1, y1 = z.bbox
        bbox_ratio = (
            x0 / max(page_w, 1),
            y0 / max(page_h, 1),
            x1 / max(page_w, 1),
            y1 / max(page_h, 1),
        )
        template_zones.append((z.type, bbox_ratio, z.confidence))

    return ZoneTemplate(
        zones=template_zones,
        source_page=page_idx,
        zone_count=len(template_zones),
    )


def apply_zone_template(
    template: ZoneTemplate,
    page_plum,
    page_idx: int,
) -> list[Zone] | None:
    """Apply a cached zone template to a new page, skipping full segmentation.

    Maps the template's normalized bboxes onto the new page dimensions,
    then assigns each character to the zone whose bbox contains it.

    Safety: If fewer than 50% of chars are captured by template zones,
    returns None (caller should fall back to full segmentation).

    Args:
        template: ZoneTemplate from build_zone_template.
        page_plum: pdfplumber page object.
        page_idx: Current page index.

    Returns:
        list[Zone] if template applied successfully, None if fallback needed.
    """
    chars = page_plum.chars
    rects = page_plum.rects or []
    page_h = page_plum.height
    page_w = page_plum.width

    if not chars or not template.zones:
        return None

    # Project template bboxes onto this page's dimensions
    projected_zones = []
    for zone_type, bbox_ratio, confidence in template.zones:
        rx0, ry0, rx1, ry1 = bbox_ratio
        bbox = (
            rx0 * page_w,
            ry0 * page_h,
            rx1 * page_w,
            ry1 * page_h,
        )
        projected_zones.append(
            {
                "type": zone_type,
                "bbox": bbox,
                "confidence": confidence,
                "chars": [],
                "rects": [],
            }
        )

    # Assign each char to the closest containing zone
    assigned_count = 0
    for c in chars:
        cx = (c.get("x0", 0) + c.get("x1", 0)) / 2
        cy = (c.get("top", 0) + c.get("bottom", 0)) / 2
        best_zone = None
        best_dist = float("inf")

        for pz in projected_zones:
            bx0, by0, bx1, by1 = pz["bbox"]
            margin = 3.0
            if bx0 - margin <= cx <= bx1 + margin and by0 - margin <= cy <= by1 + margin:
                best_zone = pz
                break
            else:
                zcx = (bx0 + bx1) / 2
                zcy = (by0 + by1) / 2
                dist = abs(cx - zcx) + abs(cy - zcy)
                if dist < best_dist:
                    best_dist = dist
                    best_zone = pz

        if best_zone is not None:
            best_zone["chars"].append(c)
            assigned_count += 1

    if assigned_count < len(chars) * 0.5:
        logger.debug(
            f"[DocMirror] Perf #9: template mismatch on page {page_idx} "
            f"({assigned_count}/{len(chars)} chars assigned), falling back"
        )
        return None

    for r in rects:
        ry = r.get("top", 0)
        for pz in projected_zones:
            bx0, by0, bx1, by1 = pz["bbox"]
            if by0 - 3 <= ry <= by1 + 3:
                pz["rects"].append(r)
                break

    result_zones = []
    for pz in projected_zones:
        if not pz["chars"]:
            continue

        zone_chars = sorted(pz["chars"], key=lambda c: (c["top"], c["x0"]))
        text = "".join(c["text"] for c in zone_chars)

        actual_x0 = min(c["x0"] for c in zone_chars)
        actual_y0 = min(c["top"] for c in zone_chars)
        actual_x1 = max(c["x1"] for c in zone_chars)
        actual_y1 = max(c["bottom"] for c in zone_chars)

        zone = Zone(
            type=pz["type"],
            bbox=(actual_x0, actual_y0, actual_x1, actual_y1),
            page=page_idx,
            chars=list(pz["chars"]),
            rects=list(pz["rects"]),
            text=text.strip(),
            confidence=pz["confidence"],
        )
        result_zones.append(zone)

    logger.debug(
        f"[DocMirror] Perf #9: template applied on page {page_idx} → "
        f"{len(result_zones)} zones ({assigned_count}/{len(chars)} chars)"
    )
    return result_zones
