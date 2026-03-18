# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Web Adapter — HTML → ParseResult
=================================

Reads HTML files and extracts structured content using readability-lxml
and BeautifulSoup (with fallback to raw text).

Processing logic:
    1. Use readability-lxml to strip noise (navbars, footers, ads).
    2. Extract title → TextBlock(level=TITLE).
    3. Parse clean HTML via BeautifulSoup for structured blocks:
       - Headings (h1-h6) → TextBlock with appropriate level.
       - Paragraphs/lists → TextBlock(level=BODY).
    4. Fallback to raw text if libraries not available.

Metadata includes:
    - parser_name: "WebAdapter"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from docmirror.framework.base import BaseParser

logger = logging.getLogger(__name__)


class WebAdapter(BaseParser):
    """HTML/Web content format adapter — structured extraction with readability + BeautifulSoup."""

    async def to_parse_result(self, file_path: Path, **kwargs) -> ParseResult:
        """
        Read an HTML file and return its clean extracted content as a ParseResult.

        Uses readability-lxml to strip noise and BeautifulSoup for structured blocks.
        Falls back to raw text if libraries are not available.
        """
        from docmirror.models.entities.parse_result import (
            PageContent,
            ParseResult,
            ParserInfo,
            TextBlock,
            TextLevel,
        )

        logger.info(f"[WebAdapter] Starting extraction for HTML: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        texts: list[TextBlock] = []

        try:
            from bs4 import BeautifulSoup
            from readability import Document

            # 1. Strip noise via readability-lxml
            doc = Document(content)
            clean_html = doc.summary()

            # 2. Extract title
            title = doc.title()
            if title:
                texts.append(TextBlock(content=title, level=TextLevel.TITLE))

            # 3. Parse clean html via BeautifulSoup for structured blocks
            soup = BeautifulSoup(clean_html, "html.parser")

            level_map = {
                "h1": TextLevel.H1,
                "h2": TextLevel.H2,
                "h3": TextLevel.H3,
                "h4": TextLevel.H3,
                "h5": TextLevel.H3,
                "h6": TextLevel.H3,
            }

            allowed_tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
            for elem in soup.find_all(allowed_tags):
                text = elem.get_text(separator=" ", strip=True)
                if not text:
                    continue

                level = level_map.get(elem.name, TextLevel.BODY)
                texts.append(TextBlock(content=text, level=level))

        except ImportError:
            logger.warning("[WebAdapter] readability-lxml or bs4 not installed. Falling back to raw text.")
            texts = [TextBlock(content=content[:10000], level=TextLevel.BODY)]
        except Exception as e:
            logger.warning(f"[WebAdapter] readability extraction failed: {e}. Falling back to raw text.")
            texts = [TextBlock(content=content[:10000], level=TextLevel.BODY)]

        page = PageContent(page_number=0, texts=texts)

        return ParseResult(
            pages=[page],
            parser_info=ParserInfo(
                parser_name="WebAdapter",
                page_count=1,
                overall_confidence=0.9,
            ),
        )
