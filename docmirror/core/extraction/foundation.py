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
from typing import Any, Dict, List, Optional, Tuple

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
    def extract_page_words(fitz_page) -> List[Tuple]:
        """
        Extract word list from a single page.

        Each word: (x0, y0, x1, y1, text, block_no, line_no, word_no)
        """
        return fitz_page.get_text("words")

    @staticmethod
    def extract_page_blocks_with_style(fitz_page) -> List[Dict[str, Any]]:
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
                    result.append({
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
                    })
        return result

    @staticmethod
    def get_page_dimensions(fitz_page) -> Tuple[float, float]:
        """Return page (width, height)."""
        rect = fitz_page.rect
        return rect.width, rect.height

    @staticmethod
    def extract_raw_text_from_bbox(fitz_page, bbox: Tuple[float, float, float, float]) -> str:
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

    @staticmethod
    def extract_multicrop_payload(fitz_page, rois: List[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
        """
        Multi-crop tokenisation inspired by DeepSeek-OCR2:
        Constructs a Global Base (low-res overview) + Local Focus
        (high-res region patches) multi-image payload.

        Args:
            fitz_page: PyMuPDF page object
            rois: Regions of interest as [(x0, y0, x1, y1), ...]
                  e.g. table or dense data area bounding boxes

        Returns:
            Dict containing:
               'global_img': 150 DPI global image (numpy RGB)
               'local_patches': list of 300 DPI local patch images (numpy RGB)
        """
        import numpy as np
        import cv2

        payload = {"global_img": None, "local_patches": []}

        # 1. Generate Global Base (low-resolution, ~150 DPI, long edge <= 1024)
        pix_global = fitz_page.get_pixmap(dpi=150)
        img_global = np.frombuffer(pix_global.samples, dtype=np.uint8).reshape(pix_global.h, pix_global.w, pix_global.n)
        if pix_global.n == 4:
            img_global = cv2.cvtColor(img_global, cv2.COLOR_RGBA2RGB)

        payload["global_img"] = img_global

        # 2. If no ROIs provided, return early
        if not rois:
            return payload

        # 3. Generate Local Focus Patches (high-resolution, 300 DPI)
        # To avoid full-page 300 DPI rendering, use fitz clip to render only ROI areas
        import fitz
        for roi in rois:
            x0, y0, x1, y1 = roi
            # Expand boundary by 5px to prevent clipping characters
            rect = fitz.Rect(max(0, x0 - 5), max(0, y0 - 5), x1 + 5, y1 + 5)

            pix_patch = fitz_page.get_pixmap(dpi=300, clip=rect)
            img_patch = np.frombuffer(pix_patch.samples, dtype=np.uint8).reshape(pix_patch.h, pix_patch.w, pix_patch.n)
            if pix_patch.n == 4:
                img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGBA2RGB)

            payload["local_patches"].append(img_patch)

        return payload


# ===============================================================================
# pdfplumber Engine
# ===============================================================================

class PDFPlumberEngine:
    """
    pdfplumber wrapper — high-precision table structure recognition.

    Primary uses:
        1. Table detection and extraction (line-based / text-based strategies)
        2. Character-level coordinate extraction
    """

    @staticmethod
    def open(file_path: Path):
        """Open a PDF and return a pdfplumber.PDF."""
        import pdfplumber
        return pdfplumber.open(str(file_path))

    @staticmethod
    def extract_tables(page_plum, **kwargs) -> List[List[List[str]]]:
        """
        Extract all tables from a single page.

        Returns:
            List of tables, each table is List[List[str]].
        """
        try:
            tables = page_plum.extract_tables(kwargs) if kwargs else page_plum.extract_tables()
            if not tables:
                return []
            # Clean None values
            result = []
            for tbl in tables:
                if tbl:
                    cleaned = [
                        [str(cell) if cell is not None else "" for cell in row]
                        for row in tbl
                    ]
                    result.append(cleaned)
            return result
        except Exception as e:
            logger.debug(f"pdfplumber table extraction error: {e}")
            return []

    @staticmethod
    def get_page_chars(page_plum) -> List[Dict[str, Any]]:
        """Return all character coordinate information for a single page."""
        return page_plum.chars if hasattr(page_plum, 'chars') else []


# ===============================================================================
# OCR Engine
# ===============================================================================

class OCREngine:
    """
    OCR engine wrapper — proxy to engines.vision.rapidocr_engine singleton.

    Model is only loaded on first call during scanned document processing
    to avoid startup overhead.
    """

    _instance: Optional[Any] = None

    @classmethod
    def get_engine(cls):
        """Get OCR engine singleton (proxied to rapidocr_engine)."""
        if cls._instance is None:
            try:
                from docmirror.core.ocr.vision.rapidocr_engine import get_ocr_engine
                cls._instance = get_ocr_engine()
            except ImportError:
                logger.warning("RapidOCR not available, OCR features disabled")
                return None
        return cls._instance

    @classmethod
    def is_available(cls) -> bool:
        """Check whether the OCR engine is ready."""
        engine = cls.get_engine()
        return engine is not None and hasattr(engine, '_engine') and engine._engine is not None
