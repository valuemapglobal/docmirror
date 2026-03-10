"""
Web Adapter — HTML → BaseResult
=================================

Reads HTML files and extracts raw text content. This is a minimal
implementation that reads the file as UTF-8 text and stores it in
a single text Block, truncated to 10,000 characters.

.. note::
    This adapter does NOT perform HTML parsing or tag stripping.
    A future enhancement should integrate BeautifulSoup or lxml
    to extract structured content (headings, tables, links, etc.)
    from the HTML DOM.

Metadata includes:
    - source_format: "html"
"""

from __future__ import annotations

import logging
from pathlib import Path

from docmirror.framework.base import BaseParser, ParserOutput, ParserStatus
from docmirror.models.domain import BaseResult, Block, PageLayout

logger = logging.getLogger(__name__)


class WebAdapter(BaseParser):
    """HTML/Web content format adapter — raw text extraction (no DOM parsing)."""

    async def to_base_result(self, file_path: Path) -> BaseResult:
        """
        Read an HTML file and return its raw text content as a BaseResult.

        Content is truncated to 10,000 characters to prevent excessive
        memory usage for large HTML files. No HTML tag stripping or
        DOM-level structured extraction is performed in this version.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # TODO: integrate BeautifulSoup for structured HTML extraction
        # (headings, paragraphs, tables, links, metadata)
        blocks = [Block(block_type="text", raw_content=content[:10000], page=0)]
        page = PageLayout(page_number=0, blocks=tuple(blocks))
        return BaseResult(
            pages=(page,),
            full_text=content[:10000],
            metadata={"source_format": "html"},
        )
