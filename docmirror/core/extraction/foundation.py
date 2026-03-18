# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Foundation Engine Wrappers
==========================================

Unified wrappers around low-level PDF libraries, isolating third-party
dependencies:
    - FitzEngine:       PyMuPDF fast text/font/metadata extraction
    - PDFPlumberEngine: pdfplumber high-precision table recognition
    - OCREngine:        PaddleOCR/RapidOCR lazy-loading wrapper

Upstream code accesses low-level capabilities only through these Engine
classes, making future library replacements straightforward.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ===============================================================================
# PyMuPDF Engine
# ===============================================================================


class FitzEngine:
    """
    PyMuPDF wrapper — ultra-fast text extraction + font analysis.

    Primary uses:
        1. Text layer pre-check (digital vs scanned)
        2. Full text extraction + text coordinates
        3. Font/colour/bold and other visual feature extraction
    """

    @staticmethod
    def open(file_path: Path):
        """Open a PDF and return a fitz.Document."""
        import fitz

        return fitz.open(str(file_path))

    @staticmethod
    def has_text_layer(fitz_doc) -> bool:
        """
        Quick check whether the PDF contains a text layer.

        Strategy: check the first 3 pages; if any page has > 20 chars
        of text, the document is considered to have a text layer.
        """
        for page_idx in range(min(3, len(fitz_doc))):
            text = fitz_doc[page_idx].get_text()
            if text and len(text.strip()) > 20:
                return True
        return False

    @staticmethod
    def extract_page_text(fitz_page) -> str:
        """Extract full text from a single page."""
        return fitz_page.get_text()

    @staticmethod
    def extract_page_blocks_with_style(fitz_page) -> list[dict[str, Any]]:
        """
        Extract text blocks from a single page with font/colour info.

        Returns:
            List of {
                "text": str,
                "bbox": (x0, y0, x1, y1),
                "font_name": str,
                "font_size": float,
                "color": int,
                "flags": int,  # bit 0=superscript, 1=italic, 2=serif, 3=monospace, 4=bold
            }
        """
        result = []
        text_dict = fitz_page.get_text("dict", flags=11)
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # only process text blocks
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    result.append(
                        {
                            "text": span.get("text", ""),
                            "bbox": (
                                span.get("bbox", (0, 0, 0, 0))[0],
                                span.get("bbox", (0, 0, 0, 0))[1],
                                span.get("bbox", (0, 0, 0, 0))[2],
                                span.get("bbox", (0, 0, 0, 0))[3],
                            ),
                            "font_name": span.get("font", ""),
                            "font_size": span.get("size", 0.0),
                            "color": span.get("color", 0),
                            "flags": span.get("flags", 0),
                        }
                    )
        return result

    @staticmethod
    def get_page_dimensions(fitz_page) -> tuple[float, float]:
        """Return page (width, height)."""
        rect = fitz_page.rect
        return rect.width, rect.height

    @staticmethod
    def extract_raw_text_from_bbox(fitz_page, bbox: tuple[float, float, float, float]) -> str:
        """
        Extract 100% accurate low-level text within a bounding box.

        Used as a Hybrid Text-Vision Prior injected into multimodal LLMs
        to prevent digit/character hallucinations.
        """
        import fitz

        x0, y0, x1, y1 = bbox
        rect = fitz.Rect(x0, y0, x1, y1)
        # flags=0 extracts plain text in reading order
        return fitz_page.get_text("text", clip=rect).strip()
