"""
Structured Adapter — JSON/CSV → BaseResult
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from docmirror.framework.base import BaseParser, ParserOutput, ParserStatus
from docmirror.models.domain import BaseResult, Block, PageLayout

logger = logging.getLogger(__name__)


class StructuredAdapter(BaseParser):
    """Structured data (JSON/CSV) 格式适配器。"""

    async def to_base_result(self, file_path: Path) -> BaseResult:
        ext = file_path.suffix.lower()
        blocks = []
        text = ""

        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                blocks.append(Block(block_type="key_value", raw_content=data, page=0))
            text = json.dumps(data, indent=2, ensure_ascii=False)

        elif ext == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if rows:
                blocks.append(Block(block_type="table", raw_content=rows, page=0))
                text = "\n".join(",".join(r) for r in rows)

        page = PageLayout(page_number=0, blocks=tuple(blocks))
        return BaseResult(pages=(page,), full_text=text, metadata={"source_format": ext.lstrip(".")})


