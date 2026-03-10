"""
版面分析引擎 (Layout Analysis Engine)
=======================================

提供页面级版面分析与空间分区能力。
自包含，不依赖 MultiModal 包外部的任何 v1 代码。

=== 模块结构 (v2 重构后) ===

  本文件仅保留:
    - Module 1:  版面分析  — ALPageLayout / analyze_page_layout / analyze_document_layout
    - Module 1b: 空间分区  — Zone / segment_page_into_zones / _classify_zone

  已拆分至独立模块:
    - text_utils.py:         CJK 工具 / normalize_text / parse_amount / headers_match
    - vocabulary.py:         VOCAB_BY_CATEGORY / KNOWN_HEADER_WORDS / 行分类器
    - table_postprocess.py:  post_process_table 全家族
    - watermark.py:          preprocess_pdf / filter_watermark_page / _dedup_overlapping_chars
    - table_extraction.py:   extract_tables_layered (6+1 Layer) 全链路
    - ocr_fallback.py:       analyze_scanned_page (扫描件 OCR)

=== 向后兼容 ===

  所有历史公开符号均通过 re-export 保持向后兼容。
  调用方无需修改任何 import 语句。
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 向后兼容 re-exports  — Lazy Loading (按需导入，避免触发 11+ 模块链)
# ═══════════════════════════════════════════════════════════════════════════════

# 映射表: symbol_name → (module_path, is_package_level)
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
    "preprocess_pdf", "is_watermark_char", "filter_watermark_page",
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
# Module 1: 版面分析器
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContentRegion:
    """页面上的一个内容区域。"""
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
    """单页版面分析结果。"""
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
    启发式检测无线表格。
    如果 ≥3 行都有 ≥2 个独立 x 段 → 判定为无线表格。
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
    """分析单页版面结构 (~30ms/页)。"""
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

    try:
        tables = page.find_tables()
        for tbl in tables.tables:
            layout.regions.append(ContentRegion(
                type="table", bbox=tbl.bbox, page=page_idx,
                text_preview=f"table_{len(tbl.cells)}cells",
            ))
        layout.table_count = len(tables.tables)
        layout.has_table = layout.table_count > 0
    except Exception:
        pass

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
    """分析整个文档的版面结构。"""
    layouts = []
    for page_idx in range(len(fitz_doc)):
        layouts.append(analyze_page_layout(fitz_doc[page_idx], page_idx))

    if layouts and layouts[0].is_continuation:
        layouts[0].is_continuation = False

    logger.info(
        f"[v2] {len(layouts)} pages: "
        + " | ".join(
            f"P{l.page_index+1}({'cont' if l.is_continuation else 'new'}:"
            f"T{l.table_count}/I{l.image_count}/Txt{l.text_region_count})"
            for l in layouts
        )
    )
    return layouts


# ═══════════════════════════════════════════════════════════════════════════════
# Module 1b: 空间分区
# ═══════════════════════════════════════════════════════════════════════════════

def _reconstruct_rows_from_chars(chars, col_gap: float = 8.0):
    """Fallback: 从 chars 直接重建表格行。"""
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
    """页面上的一个大区域 (3~5 个/页)。"""
    type: str  # "title" | "summary" | "data_table" | "footer" | "formula" | "unknown"
    bbox: Tuple[float, float, float, float]
    page: int = 0
    chars: list = field(default_factory=list)
    rects: list = field(default_factory=list)
    text: str = ""
    confidence: float = 1.0  # 模型检测置信度, 规则方法默认1.0


def segment_page_into_zones(
    page_plum, page_idx: int, gap_threshold: float = 15.0,
) -> List[Zone]:
    """空间分区: 模拟人眼把页面切成 3~5 个大区域。"""
    chars = page_plum.chars
    rects = page_plum.rects or []
    page_h = page_plum.height

    if not chars:
        return []

    # ── 优化4: 动态 gap_threshold ──
    # 用字符中位高度 × 1.5 替代固定 15pt, 自适应不同字号
    char_heights = [c["bottom"] - c["top"] for c in chars if c.get("bottom", 0) > c.get("top", 0)]
    if char_heights:
        sorted_h = sorted(char_heights)
        median_h = sorted_h[len(sorted_h) // 2]
        gap_threshold = max(12.0, median_h * 1.5)
        logger.debug(f"[v2] zone split: median_char_h={median_h:.1f}, gap_threshold={gap_threshold:.1f}")

    row_ys = sorted(set(round(c["top"] / 3) * 3 for c in chars))

    # ── H3 增强: 基于字体大小变化检测段落边界 ──
    # 收集每行的中位字体大小
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

        # H3: 字体大小变化也视为段落分隔
        if not is_gap and row_ys[i] in row_font_sizes and row_ys[i - 1] in row_font_sizes:
            fs_curr = row_font_sizes[row_ys[i]]
            fs_prev = row_font_sizes[row_ys[i - 1]]
            if abs(fs_curr - fs_prev) > 2.0:  # 字号差 > 2pt
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

    # 合并相邻 data_table zones
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

    # 语义优先的阅读顺序（借鉴 OCR2 因果流）：抛弃死板 y-band
    router = GraphRouter(page_width=page_plum.width, page_height=page_plum.height)
    causal_zones = router.build_flow(merged_zones)

    return causal_zones


def _classify_zone(zone: Zone, page_h: float) -> str:
    """判定 Zone 类型。"""
    # 懒加载模块级符号: __getattr__ 仅对模块外部生效,
    # 本模块内部需通过 globals() 或显式 import 才能取得延迟注册的名字。
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

    # ── 管道网格检测: ASCII 画线表格 (mainframe) ──
    pipe_count = sum(1 for c in zone.chars if c.get("text") in _PIPE_CHARS)
    if pipe_count >= 10:
        return "data_table"

    # ── 数据内容检测: 含日期+金额的 zone 优先判为 data_table ──
    # 防止续页数据行因 y_ratio 小 / char_count 少被误判为 title
    _has_date = bool(re.search(r'\d{8}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}', text))
    _has_amount = bool(re.search(r'(?:RMB|USD|CNY)\s*[\d,.]+|\d+\.\d{2}', text))
    if _has_date and _has_amount:
        return "data_table"

    # ── 词表表头检测: 含 ≥3 个已知列名的 zone 是表头 → data_table ──
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
