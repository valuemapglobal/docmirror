"""
Debug PDF Visualizer
======================================

Overlay Zone/Block boundaries and annotations on the original PDF,
outputting a color-coded debug PDF for quick layout analysis inspection.

Usage::

    from docmirror.core.output.visualizer import render_debug_pdf
    render_debug_pdf(fitz_doc, pages, Path("output_debug.pdf"))

Color coding:
    - table:     blue (#3B82F6)
    - title:     red (#EF4444)
    - text:      green (#22C55E)
    - key_value: orange (#F97316)
    - footer:    gray (#9CA3AF)
    - image:     purple (#A855F7)
    - formula:   cyan (#06B6D4)
"""


from __future__ import annotations

import logging
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.domain import PageLayout

logger = logging.getLogger(__name__)

# Color mapping: block_type → (R, G, B) in 0-1 range
_COLOR_MAP = {
    "table":     (0.231, 0.510, 0.965),   # blue
    "title":     (0.937, 0.267, 0.267),   # red
    "text":      (0.133, 0.773, 0.369),   # green
    "key_value": (0.976, 0.451, 0.086),   # orange
    "footer":    (0.612, 0.639, 0.686),   # gray
    "image":     (0.659, 0.333, 0.969),   # purple
    "formula":   (0.024, 0.714, 0.831),   # cyan
}


def render_debug_pdf(
    fitz_doc,
    pages: List[PageLayout],
    output_path: Path,
) -> Path:
    """Draw Zone/Block boundaries on original PDF, generating a debug PDF.

    Args:
        fitz_doc: An opened PyMuPDF document object.
        pages: PageLayout list from CoreExtractor.
        output_path: Output path for the debug PDF.

    Returns:
        Output file path.
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
                continue  # no bbox info

            rect = pymupdf.Rect(x0, y0, x1, y1)
            color = _COLOR_MAP.get(block.block_type, (0.5, 0.5, 0.5))

            # Draw rectangle border
            fitz_page.draw_rect(rect, color=color, width=1.5)

            # Annotation: block_type + reading_order + heading_level
            label_parts = [f"#{block.reading_order}", block.block_type]
            if block.heading_level is not None:
                label_parts.append(f"h{block.heading_level}")

            label = " ".join(label_parts)

            # Write label at top-left corner of rectangle
            label_point = pymupdf.Point(x0 + 2, y0 + 10)
            try:
                fitz_page.insert_text(
                    label_point,
                    label,
                    fontsize=7,
                    color=color,
                )
            except Exception as exc:
                logger.debug(f"operation: suppressed {exc}")
                pass  # Some pages may not support text insertion

    # Save debug PDF
    try:
        fitz_doc.save(str(output_path))
        logger.info(f"[visualizer] Debug PDF saved: {output_path}")
    except Exception as e:
        logger.error(f"[visualizer] Failed to save debug PDF: {e}")

    return output_path
