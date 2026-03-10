"""
Web Adapter — HTML → BaseResult
"""

from __future__ import annotations

import logging
from pathlib import Path

from docmirror.framework.base import BaseParser, ParserOutput, ParserStatus
from docmirror.models.domain import BaseResult, Block, PageLayout

logger = logging.getLogger(__name__)


class WebAdapter(BaseParser):
    """HTML/Web content 格式适配器。"""

    async def to_base_result(self, file_path: Path) -> BaseResult:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # TODO: integrate BeautifulSoup for structured extraction
        blocks = [Block(block_type="text", raw_content=content[:10000], page=0)]
        page = PageLayout(page_number=0, blocks=tuple(blocks))
        return BaseResult(
            pages=(page,),
            full_text=content[:10000],
            metadata={"source_format": "html"},
        )


