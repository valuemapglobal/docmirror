"""
PPT Adapter — PowerPoint → BaseResult

使用 python-pptx 提取幻灯片文本和表格。
"""

from __future__ import annotations

import logging
from pathlib import Path

from docmirror.framework.base import BaseParser, ParserOutput, ParserStatus
from docmirror.models.domain import BaseResult, Block, PageLayout

logger = logging.getLogger(__name__)


class PPTAdapter(BaseParser):
    """PowerPoint (.pptx) 格式适配器。"""

    async def to_base_result(self, file_path: Path) -> BaseResult:
        from pptx import Presentation
        prs = Presentation(str(file_path))

        pages = []
        text_parts = []

        for i, slide in enumerate(prs.slides):
            blocks = []
            slide_texts = []

            # Title
            if slide.shapes.title and slide.shapes.title.text:
                blocks.append(Block(
                    block_type="title",
                    raw_content=slide.shapes.title.text,
                    page=i,
                    heading_level=2,
                ))
                slide_texts.append(f"### Slide {i+1}: {slide.shapes.title.text}")
            else:
                slide_texts.append(f"### Slide {i+1}")

            # Text + Tables
            for shape in slide.shapes:
                if shape == slide.shapes.title:
                    continue
                if hasattr(shape, "text") and shape.text:
                    blocks.append(Block(block_type="text", raw_content=shape.text, page=i))
                    slide_texts.append(shape.text)
                if shape.has_table:
                    rows = [[cell.text.strip() for cell in row.cells] for row in shape.table.rows]
                    if rows:
                        blocks.append(Block(block_type="table", raw_content=rows, page=i))

            pages.append(PageLayout(page_number=i, blocks=tuple(blocks)))
            text_parts.append("\n".join(slide_texts))

        return BaseResult(
            pages=tuple(pages),
            full_text="\n\n".join(text_parts),
            metadata={"source_format": "pptx", "slide_count": len(prs.slides)},
        )


