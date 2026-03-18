"""
PDF Adapter — PDF → ParseResult
====================================================

Converts PDF files into structured output.

- ``to_parse_result()``: Extracts via ``CoreExtractor`` → BaseResult,
  then bridges to ParseResult via ``ParseResultBridge``.

- ``perceive()``: Inherited from ``BaseParser``. Routes through the
  shared ``Orchestrator`` middleware pipeline (SceneDetector,
  EntityExtractor, Validator, etc.) then produces ``ParseResult``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from docmirror.framework.base import BaseParser
from docmirror.models.entities.domain import BaseResult

logger = logging.getLogger(__name__)


class PDFAdapter(BaseParser):
    """
    PDF format adapter.

    Uses CoreExtractor for raw extraction, then relies on the base class
    ``perceive()`` to run the shared Orchestrator middleware pipeline.

    Args:
        enhance_mode: Enhancement level for the middleware pipeline.
            One of "raw" (extraction only), "standard" (default), or "full".
    """

    def __init__(self, enhance_mode: str = "standard", **kwargs):
        self._enhance_mode = enhance_mode

    async def to_parse_result(self, file_path: Path, **kwargs) -> ParseResult:
        """
        Extract a PDF into a ParseResult.

        Pipeline: CoreExtractor → BaseResult → ParseResultBridge → ParseResult.
        """
        from docmirror.core.extraction.extractor import CoreExtractor
        from docmirror.models.construction.parse_result_bridge import ParseResultBridge

        logger.info(f"[PDFAdapter] Starting extraction for: {file_path}")
        extractor = CoreExtractor()
        base_result = await extractor.extract(file_path)
        logger.info(f"[PDFAdapter] Completed extraction for: {file_path}")

        pr = ParseResultBridge.from_base_result(base_result)

        # PDF-specific parser_info
        pr.parser_info.parser_name = "DocMirror"
        pr.parser_info.table_engine = "pymupdf_native"
        pr.parser_info.page_count = len(base_result.pages)

        return pr

    async def perceive(self, file_path: Path, **context) -> ParseResult:
        """
        Full pipeline: PDF → middleware → ParseResult.

        Injects the adapter's enhance_mode into context before delegating
        to the base class implementation.
        """
        context.setdefault("enhance_mode", self._enhance_mode)
        context.setdefault("file_type", "pdf")
        return await super().perceive(file_path, **context)
