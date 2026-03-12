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

from docmirror.framework.base import BaseParser
from docmirror.models.domain import BaseResult, Block, PageLayout

logger = logging.getLogger(__name__)


class WebAdapter(BaseParser):
    """HTML/Web content format adapter — raw text extraction (no DOM parsing)."""

    async def to_base_result(self, file_path: Path) -> BaseResult:
        """
        Read an HTML file and return its clean extracted text as a BaseResult.
        
        Uses readability-lxml to strip noise (navbars, footers, ads) and
        extracts semantic blocks (headings, paragraphs, lists) via BeautifulSoup.
        Fallback to raw text if readability processing fails.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = []
        try:
            from readability import Document
            from bs4 import BeautifulSoup
            
            # 1. Strip noise via readability-lxml
            doc = Document(content)
            clean_html = doc.summary()
            
            # 2. Extract title
            title = doc.title()
            if title:
                blocks.append(Block(
                    block_type="title", 
                    raw_content=title, 
                    page=0, 
                    heading_level=1
                ))
            
            # 3. Parse clean html via BeautifulSoup for structured blocks
            soup = BeautifulSoup(clean_html, "html.parser")
            
            # Find relevant text-containing tags
            allowed_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th']
            for elem in soup.find_all(allowed_tags):
                text = elem.get_text(separator=" ", strip=True)
                if not text:
                    continue
                
                btype = "text"
                level = None
                
                if elem.name.startswith('h'):
                    btype = "title"
                    level = int(elem.name[1])
                elif elem.name in ['td', 'th']:
                    # Simple table cells map to text blocks in generic HTML
                    btype = "text"
                    
                blocks.append(Block(
                    block_type=btype, 
                    raw_content=text, 
                    page=0, 
                    heading_level=level
                ))
                
            full_text = "\n\n".join(b.raw_content for b in blocks if isinstance(b.raw_content, str))
            
        except ImportError:
            logger.warning("[WebAdapter] readability-lxml or bs4 not installed. Falling back to raw text.")
            blocks = [Block(block_type="text", raw_content=content[:10000], page=0)]
            full_text = content[:10000]
        except Exception as e:
            logger.warning(f"[WebAdapter] readability extraction failed: {e}. Falling back to raw text.")
            blocks = [Block(block_type="text", raw_content=content[:10000], page=0)]
            full_text = content[:10000]

        page = PageLayout(page_number=0, blocks=tuple(blocks))
        return BaseResult(
            pages=(page,),
            full_text=full_text,
            metadata={"source_format": "html"},
        )
