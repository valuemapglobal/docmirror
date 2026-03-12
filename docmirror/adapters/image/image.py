"""
Image Adapter — Image → BaseResult
====================================

Converts image files (JPG, PNG, TIFF, etc.) into structured data using 
RapidOCR (ONNX Runtime) for plain text extraction. This adapter produces a single 
text Block without complex structured table/entity data, as it currently operates
in a purely CPU-bound environment without Vision-Language Models.
"""
from __future__ import annotations


import logging
from pathlib import Path

from docmirror.framework.base import BaseParser
from docmirror.models.entities.domain import BaseResult, Block, PageLayout

logger = logging.getLogger(__name__)


class ImageAdapter(BaseParser):
    """
    Image format adapter using OCR extraction.
    
    Produces a single text Block containing all recognized text lines joined by newlines.
    """

    async def to_base_result(self, file_path: Path) -> BaseResult:
        """
        Convert an image file to BaseResult using OCR.
        """
        return await self._ocr_fallback(file_path)

    async def _ocr_fallback(self, file_path: Path) -> BaseResult:
        """
        Fallback path: extract text from the image using RapidOCR.

        Returns a BaseResult with a single text Block containing all
        recognized text lines joined by newlines. If OCR is unavailable
        or produces no output, returns an empty result.
        """
        try:
            import cv2
            from docmirror.core.ocr.vision.rapidocr_engine import get_ocr_engine
            engine = get_ocr_engine()
            img = cv2.imread(str(file_path))
            if img is None:
                logger.warning(f"[ImageAdapter] Cannot read image: {file_path.name}")
                text = ""
            else:
                words = engine.detect_image_words(img)
                text = "\n".join(w[4] for w in words) if words else ""
        except Exception as e:
            logger.warning(f"[ImageAdapter] OCR fallback failed: {e}")
            text = ""

        blocks = [Block(block_type="text", raw_content=text, page=0)] if text else []
        page = PageLayout(page_number=0, blocks=tuple(blocks))
        return BaseResult(pages=(page,), full_text=text, metadata={"source_format": "image_ocr"})
