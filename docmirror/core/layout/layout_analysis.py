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
from collections import Counter, defaultdict
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
_register_lazy("..utils.text_utils", [
    "_is_cjk_char", "_smart_join", "normalize_text", "normalize_table",
    "headers_match", "parse_amount",
])
# vocabulary & row classifiers
_register_lazy("..utils.vocabulary", [
    "VOCAB_BY_CATEGORY", "KNOWN_HEADER_WORDS", "PIPE_CHARS", "HLINE_CHARS",
    "_ALL_BORDER_CHARS", "_RE_IS_DATE", "_RE_IS_AMOUNT", "_RE_VALID_DATE",
    "_normalize_for_vocab", "_score_header_by_vocabulary", "_is_header_cell",
    "_is_header_row", "_is_junk_row", "_is_data_row",
])
# table postprocess
_register_lazy("..table.postprocess", [
    "_extract_preamble_kv", "_strip_preamble", "post_process_table",
    "_find_vocab_words_in_string", "_fix_header_by_vocabulary", "_clean_cell",
    "_merge_split_rows", "_extract_summary_entities",
])
# watermark & preprocessing
_register_lazy("..utils.watermark", [
    "preprocess_document", "is_watermark_char", "filter_watermark_page",
    "_dedup_overlapping_chars",
])
# table extraction
_register_lazy("..table.extraction", [
    "extract_tables_layered", "get_last_layer_timings", "_quick_classify",
    "_compute_table_confidence", "_tables_look_valid", "_cell_is_stuffed",
    "_recover_header_from_zone", "_extract_by_pipe_delimited",
    "_extract_by_hline_columns", "_extract_by_rect_columns",
    "detect_columns_by_header_anchors", "detect_columns_by_whitespace_projection",
    "detect_columns_by_clustering", "detect_columns_by_word_anchors",
    "detect_columns_by_data_voting",
])
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
    bbox: Tuple[float, float, float, float]
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
    regions: List[ContentRegion] = field(default_factory=list)
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
                spans.append({
                    "x0": bbox[0], "x1": bbox[2],
                    "y_mid": (bbox[1] + bbox[3]) / 2,
                    "text": text,
                })

    if len(spans) < 6:
        return False

    spans.sort(key=lambda s: s["y_mid"])
    rows: List[List[dict]] = []
    current_row: List[dict] = [spans[0]]
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
            layout.regions.append(ContentRegion(
                type="text", bbox=bbox, page=page_idx, text_preview=preview
            ))

    layout.text_region_count = len([r for r in layout.regions if r.type == "text"])

    for b in image_blocks:
        bbox = (b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3])
        layout.regions.append(ContentRegion(
            type="image", bbox=bbox, page=page_idx,
            text_preview=f"image_{b.get('width', 0)}x{b.get('height', 0)}",
        ))
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
        len(span.get("text", ""))
        for b in text_blocks
        for line in b.get("lines", [])
        for span in line.get("spans", [])
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
                r.text_preview for r in layout.regions
                if r.type == "text" and r.bbox[3] <= table_top + 5
            )
            layout.footer_text = " | ".join(
                r.text_preview for r in layout.regions
                if r.type == "text" and r.bbox[1] >= table_bottom - 5
            )

    if layout.has_table:
        table_regions = [r for r in layout.regions if r.type == "table"]
        if table_regions:
            earliest_table_top = min(r.bbox[1] for r in table_regions)
            above_table_text = sum(
                1 for r in layout.regions
                if r.type == "text" and r.bbox[3] <= earliest_table_top + 5
            )
            layout.is_continuation = (
                earliest_table_top < rect.height * 0.15
                and above_table_text <= 2
            )

    return layout


def analyze_document_layout(fitz_doc) -> List[ALPageLayout]:
    """Analyze the layout structure of the entire document."""
    layouts = []
    for page_idx in range(len(fitz_doc)):
        layouts.append(analyze_page_layout(fitz_doc[page_idx], page_idx))

    if layouts and layouts[0].is_continuation:
        layouts[0].is_continuation = False

    logger.info(
        f"{len(layouts)} pages: "
        + " | ".join(
            f"P{l.page_index+1}({'cont' if l.is_continuation else 'new'}:"
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
            if not cell_chars: return ""
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


@dataclass
class Zone:
    """A large zone on the page (3~5 zones/page)."""
    type: str  # "title" | "summary" | "data_table" | "footer" | "formula" | "unknown"
    bbox: Tuple[float, float, float, float]
    page: int = 0
    chars: list = field(default_factory=list)
    rects: list = field(default_factory=list)
    text: str = ""
    confidence: float = 1.0  # Model Detection Confidence, rule method default 1.0


def _isolate_formula_components(chars: List[dict], page_w: float, page_h: float) -> Tuple[List[dict], List[Zone]]:
    """
    Isolates formula regions using morphological clustering of character bounding boxes.
    By finding 'seed' math symbols and dilating them to absorb adjacent subscripts/text,
    we can accurately segment inline and block formulas without VLM.
    
    Returns: (remaining_chars, formula_zones)
    """
    if not chars or page_w <= 0 or page_h <= 0:
        return chars, []
        
    try:
        import cv2
        import numpy as np
    except ImportError:
        return chars, []
        
    MATH_UNICODE = set("∑∫∏√∞∂∇±×÷≈≡≠≤≥⊂⊃⊆⊇∈∉∪∩")
    
    math_seeds = []
    
    for i, c in enumerate(chars):
        h = c.get("bottom", 0) - c.get("top", 0)
        w = c.get("x1", 0) - c.get("x0", 0)
        text = c.get("text", "").strip()
        
        is_math = False
        if h > 0 and w > 0:
            aspect = h / w
            if aspect > 2.5 or aspect < 0.2:  # extremely tall or wide (integral, fraction bar)
                is_math = True
            elif text and text[0] in MATH_UNICODE:
                is_math = True
                
        if is_math:
            math_seeds.append(i)
            
    if not math_seeds:
        return chars, []
        
    # Create low-res canvas for morphology (scale=2 is enough for bounding boxes)
    scale = 2.0
    canvas_h, canvas_w = int(page_h * scale), int(page_w * scale)
    if canvas_h <= 0 or canvas_w <= 0:
        return chars, []
        
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    # Draw seeds with generous dilation to catch subscripts/neighbors
    for idx in math_seeds:
        c = chars[idx]
        x0, y0 = int(c["x0"] * scale), int(c["top"] * scale)
        x1, y1 = int(c["x1"] * scale), int(c["bottom"] * scale)
        
        # Expand seed by 15pt horizontally, 8pt vertically
        mx, my = int(15 * scale), int(8 * scale)
        cv2.rectangle(canvas, (max(0, x0 - mx), max(0, y0 - my)), 
                      (min(canvas_w, x1 + mx), min(canvas_h, y1 + my)), 255, -1)
                      
    # Draw standard chars to build connected components
    for c in chars:
        x0, y0 = int(c["x0"] * scale), int(c["top"] * scale)
        x1, y1 = int(c["x1"] * scale), int(c["bottom"] * scale)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), 255, -1)
        
    # Morphological Close to connect fragmented formula parts
    kernel = np.ones((int(5 * scale), int(15 * scale)), np.uint8)
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(canvas, connectivity=8)
    
    formula_zones = []
    used_char_indices = set()
    
    for i in range(1, num_labels):
        _, _, w, h, area = stats[i]
        
        # Skip absurd components (entire page) or tiny noise
        if w > canvas_w * 0.9 or h > canvas_h * 0.5 or area < 20:
            continue
            
        comp_mask = (labels == i)
        comp_char_indices = []
        comp_math_count = 0
        
        for idx, c in enumerate(chars):
            if idx in used_char_indices: continue
            cx = int((c["x0"] + c["x1"]) / 2 * scale)
            cy = int((c["top"] + c["bottom"]) / 2 * scale)
            
            if 0 <= cy < canvas_h and 0 <= cx < canvas_w and comp_mask[cy, cx]:
                comp_char_indices.append(idx)
                if idx in math_seeds:
                    comp_math_count += 1
                    
        # If CC contains mathematical seed and isn't a massive text block (< 150 chars)
        if comp_char_indices and comp_math_count > 0 and len(comp_char_indices) < 150:
            comp_chars = [chars[idx] for idx in comp_char_indices]
            used_char_indices.update(comp_char_indices)
            
            fx0 = min(c["x0"] for c in comp_chars)
            fy0 = min(c["top"] for c in comp_chars)
            fx1 = max(c["x1"] for c in comp_chars)
            fy1 = max(c["bottom"] for c in comp_chars)
            ftext = "".join(c["text"] for c in sorted(comp_chars, key=lambda c: (c["top"], c["x0"])))
            
            formula_zones.append(Zone(
                type="formula",
                bbox=(fx0, fy0, fx1, fy1),
                chars=comp_chars,
                text=ftext.strip(),
                confidence=0.9
            ))
            
    remaining_chars = [c for i, c in enumerate(chars) if i not in used_char_indices]
    return remaining_chars, formula_zones


def segment_page_into_zones(
    page_plum, page_idx: int, gap_threshold: float = 15.0,
) -> List[Zone]:
    """Spatial partitioning: Simulates human eye to split page into 3~5 large zones."""
    chars = page_plum.chars
    rects = page_plum.rects or []
    page_h = page_plum.height
    page_w = page_plum.width

    if not chars:
        return []

    # ── Morphological Formula Extraction ──
    # Extract formula zones first so they don't corrupt text parsing
    chars, formula_zones = _isolate_formula_components(chars, page_w, page_h)

    if not chars:
        return formula_zones

    # ── Optimize 4: Dynamic gap_threshold ──
    # Use median character height x 1.5 instead of fixed 15pt, adapts to different font sizes
    char_heights = [c["bottom"] - c["top"] for c in chars if c.get("bottom", 0) > c.get("top", 0)]
    if char_heights:
        sorted_h = sorted(char_heights)
        median_h = sorted_h[len(sorted_h) // 2]
        gap_threshold = max(12.0, median_h * 1.5)
        logger.debug(f"zone split: median_char_h={median_h:.1f}, gap_threshold={gap_threshold:.1f}")

    row_ys = sorted(set(round(c["top"] / 3) * 3 for c in chars))

    # ── H3 Enhancement: Detect paragraph boundaries based on font size changes ──
    # Collect median font size for each row
    row_font_sizes: Dict[int, float] = {}
    for y_key in row_ys:
        row_chars = [c for c in chars if round(c["top"] / 3) * 3 == y_key]
        sizes = [c.get("size", 0) for c in row_chars if c.get("size", 0) > 0]
        if sizes:
            sizes.sort()
            row_font_sizes[y_key] = sizes[len(sizes) // 2]

    cuts = [row_ys[0]]
    for i in range(1, len(row_ys)):
        y_gap = row_ys[i] - row_ys[i - 1]
        is_gap = y_gap > gap_threshold

        # H3: Font size changes are also considered paragraph separators
        if not is_gap and row_ys[i] in row_font_sizes and row_ys[i - 1] in row_font_sizes:
            fs_curr = row_font_sizes[row_ys[i]]
            fs_prev = row_font_sizes[row_ys[i - 1]]
            if abs(fs_curr - fs_prev) > 2.0:  # Font size difference > 2pt
                is_gap = True

        if is_gap:
            cuts.append(row_ys[i - 1])
            cuts.append(row_ys[i])
    cuts.append(row_ys[-1])

    bands = []
    for i in range(0, len(cuts) - 1, 2):
        bands.append((cuts[i], cuts[i + 1]))

    if not bands:
        bands = [(row_ys[0], row_ys[-1])]

    zones = []
    for y_start, y_end in bands:
        margin = 5
        band_chars = [
            c for c in chars
            if y_start - margin <= c["top"] <= y_end + margin
        ]
        band_rects = [
            r for r in rects
            if y_start - margin <= r["top"] <= y_end + margin
        ]
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

        zone.type = _classify_zone(zone, page_h)
        zones.append(zone)
        
    zones.extend(formula_zones)

    # Merge adjacent data_table zones
    merged_zones = []
    for z in zones:
        if (merged_zones
            and z.type == "data_table"
            and merged_zones[-1].type == "data_table"):
            prev = merged_zones[-1]
            prev.bbox = (
                min(prev.bbox[0], z.bbox[0]),
                prev.bbox[1],
                max(prev.bbox[2], z.bbox[2]),
                z.bbox[3],
            )
            prev.chars.extend(z.chars)
            prev.rects.extend(z.rects)
            prev.text += z.text
        else:
            merged_zones.append(z)

    from .graph_router import GraphRouter

    # Semantic priority reading order (borrowing OCR2 causal flow): Abandon rigid y-band
    router = GraphRouter(page_width=page_plum.width, page_height=page_plum.height)
    causal_zones = router.build_flow(merged_zones)

    return causal_zones


def _classify_zone(zone: Zone, page_h: float) -> str:
    """Determine Zone Type."""
    # Lazy load module-level symbols: __getattr__ only works externally,
    # Internal module needs to use globals() or explicit import to get the lazy registered names.
    _PIPE_CHARS = globals().get("PIPE_CHARS")
    if _PIPE_CHARS is None:
        _PIPE_CHARS = __getattr__("PIPE_CHARS")
    _KNOWN_HEADER_WORDS = globals().get("KNOWN_HEADER_WORDS")
    if _KNOWN_HEADER_WORDS is None:
        _KNOWN_HEADER_WORDS = __getattr__("KNOWN_HEADER_WORDS")

    y_ratio = zone.bbox[1] / page_h if page_h else 0
    text = zone.text
    char_count = len(zone.chars)

    if y_ratio > 0.85 and char_count < 30 and "页" in text:
        return "footer"

    # ── Pipe grid detection: ASCII drawn line table (mainframe) ──
    pipe_count = sum(1 for c in zone.chars if c.get("text") in _PIPE_CHARS)
    if pipe_count >= 10:
        return "data_table"

    # ── Data content detection: Zones with Date+Amount are prioritized as data_table ──
    # Prevent continuation data rows from being misjudged as title due to small y_ratio / char_count
    _has_date = bool(re.search(r'\d{8}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}', text))
    _has_amount = bool(re.search(r'(?:RMB|USD|CNY)\s*[\d,.]+|\d+\.\d{2}', text))
    if _has_date and _has_amount:
        return "data_table"

    # ── Header vocabulary detection: Zones containing >=3 known column names are Table headers -> data_table ──
    _vocab_hits = sum(1 for w in _KNOWN_HEADER_WORDS if w in text)
    if _vocab_hits >= 3:
        return "data_table"

    if y_ratio < 0.15 and char_count < 80:
        if not re.search(r'[\u4e00-\u9fff][：:]', text):
            return "title"

    if char_count < 300 and re.search(r'[\u4e00-\u9fff][：:]', text):
        return "summary"

    row_ys = sorted(set(round(c["top"] / 3) * 3 for c in zone.chars))
    if len(row_ys) < 2 and not any(ch.isdigit() for ch in text):
        return "summary"

    x_positions = set(round(c["x0"] / 10) * 10 for c in zone.chars)
    if len(x_positions) >= 5 and char_count > 20:
        return "data_table"

    if len(zone.rects) >= 3:
        return "data_table"

    row_ys = sorted(set(round(c["top"] / 3) * 3 for c in zone.chars))
    if len(row_ys) >= 5:
        x_starts = Counter(round(c["x0"] / 5) * 5 for c in zone.chars)
        aligned = sum(1 for _, cnt in x_starts.items() if cnt >= 3)
        if aligned >= 3:
            return "data_table"

    if len(row_ys) >= 3:
        return "data_table"

    return "unknown"
