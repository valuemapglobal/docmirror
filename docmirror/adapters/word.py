"""
Word Adapter — Word → BaseResult

使用 python-docx 提取段落和表格。
"""

from __future__ import annotations

import logging
from pathlib import Path

from docmirror.framework.base import BaseParser, ParserOutput, ParserStatus
from docmirror.models.domain import BaseResult, Block, PageLayout

logger = logging.getLogger(__name__)


class WordAdapter(BaseParser):
    """Word (.docx) 格式适配器。"""

    async def to_base_result(self, file_path: Path) -> BaseResult:
        from docx import Document
        doc = Document(str(file_path))

        blocks = []
        text_parts = []

        for para in doc.paragraphs:
            if not para.text.strip():
                continue
            btype = "title" if para.style and para.style.name.startswith("Heading") else "text"
            level = None
            if btype == "title":
                try:
                    level = int(para.style.name[-1])
                except (IndexError, ValueError):
                    level = 1
            blocks.append(Block(
                block_type=btype,
                raw_content=para.text,
                page=0,
                heading_level=level,
            ))
            text_parts.append(para.text)

        for table in doc.tables:
            rows = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                if any(row_data):
                    rows.append(row_data)
            if rows:
                blocks.append(Block(block_type="table", raw_content=rows, page=0))

        page = PageLayout(page_number=0, blocks=tuple(blocks))
        metadata = {
            "source_format": "docx",
            "paragraph_count": len(doc.paragraphs),
            "table_count": len(doc.tables),
        }

        return BaseResult(pages=(page,), full_text="\n\n".join(text_parts), metadata=metadata)


