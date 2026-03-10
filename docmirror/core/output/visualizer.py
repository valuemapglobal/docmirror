"""
可视化调试工具 (Debug PDF Visualizer)
======================================

在原始 PDF 上叠加 Zone/Block 边界和标注信息，
输出颜色编码的调试 PDF，便于快速排查版面分析结果。

使用方式::

    from docmirror.core.output.visualizer import render_debug_pdf
    render_debug_pdf(fitz_doc, pages, Path("output_debug.pdf"))

颜色编码:
    - table:     蓝色 (#3B82F6)
    - title:     红色 (#EF4444)
    - text:      绿色 (#22C55E)
    - key_value: 橙色 (#F97316)
    - footer:    灰色 (#9CA3AF)
    - image:     紫色 (#A855F7)
    - formula:   青色 (#06B6D4)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.domain import PageLayout

logger = logging.getLogger(__name__)

# 颜色映射: block_type → (R, G, B) 0-1 范围
_COLOR_MAP = {
    "table":     (0.231, 0.510, 0.965),   # 蓝
    "title":     (0.937, 0.267, 0.267),   # 红
    "text":      (0.133, 0.773, 0.369),   # 绿
    "key_value": (0.976, 0.451, 0.086),   # 橙
    "footer":    (0.612, 0.639, 0.686),   # 灰
    "image":     (0.659, 0.333, 0.969),   # 紫
    "formula":   (0.024, 0.714, 0.831),   # 青
}


def render_debug_pdf(
    fitz_doc,
    pages: List[PageLayout],
    output_path: Path,
) -> Path:
    """在原 PDF 上绘制 Zone/Block 边界，生成调试 PDF。

    Args:
        fitz_doc: 已打开的 PyMuPDF 文档对象。
        pages: CoreExtractor 输出的 PageLayout 列表。
        output_path: 调试 PDF 的输出路径。

    Returns:
        输出文件路径。
    """
    try:
        import fitz as pymupdf  # noqa: F811
    except ImportError:
        logger.warning("[visualizer] PyMuPDF not available, skipping debug PDF")
        return output_path

    output_path = Path(output_path)

    for page_layout in pages:
        page_idx = page_layout.page_number - 1
        if page_idx >= len(fitz_doc):
            continue

        fitz_page = fitz_doc[page_idx]

        for block in page_layout.blocks:
            x0, y0, x1, y1 = block.bbox
            if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
                continue  # 无 bbox 信息

            rect = pymupdf.Rect(x0, y0, x1, y1)
            color = _COLOR_MAP.get(block.block_type, (0.5, 0.5, 0.5))

            # 画矩形边框
            fitz_page.draw_rect(rect, color=color, width=1.5)

            # 标注信息: block_type + reading_order + heading_level
            label_parts = [f"#{block.reading_order}", block.block_type]
            if block.heading_level is not None:
                label_parts.append(f"h{block.heading_level}")

            label = " ".join(label_parts)

            # 在矩形左上角写标签
            label_point = pymupdf.Point(x0 + 2, y0 + 10)
            try:
                fitz_page.insert_text(
                    label_point,
                    label,
                    fontsize=7,
                    color=color,
                )
            except Exception:
                pass  # 某些页面可能不支持写入

    # 保存调试 PDF
    try:
        fitz_doc.save(str(output_path))
        logger.info(f"[visualizer] Debug PDF saved: {output_path}")
    except Exception as e:
        logger.error(f"[visualizer] Failed to save debug PDF: {e}")

    return output_path
