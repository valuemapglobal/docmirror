"""
Structured Data Adapter — JSON/CSV → BaseResult
=================================================

Handles structured data files that already have a well-defined schema.

Processing logic by format:

**JSON (.json)**:
    - Loads the entire file into a Python object.
    - If the root object is a dict, creates a ``key_value`` Block with the
      dict as raw_content (suitable for flat key-value documents).
    - The full_text is the pretty-printed JSON (2-space indent).

**CSV (.csv)**:
    - Reads all rows via Python's csv.reader (default dialect).
    - Creates a single ``table`` Block with the 2D list of row data.
    - The full_text is the comma-joined rows.

Both formats produce a single-page BaseResult with:
    - metadata.source_format set to the file extension (without dot).

.. note::
    For JSON arrays (e.g., list of records), the current implementation
    does not create structured blocks. A future enhancement could
    detect list-of-dicts patterns and convert them to table Blocks.
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
    """Structured data (JSON/CSV) format adapter."""

    async def to_base_result(self, file_path: Path) -> BaseResult:
        """
        Parse a JSON or CSV file into a BaseResult.

        Dispatches to format-specific logic based on file extension.
        JSON dicts become key_value Blocks; CSV files become table Blocks.
        """
        ext = file_path.suffix.lower()
        blocks = []
        text = ""

        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Dict objects → key-value Block for flat entity data
            if isinstance(data, dict):
                blocks.append(Block(block_type="key_value", raw_content=data, page=0))
            text = json.dumps(data, indent=2, ensure_ascii=False)

        elif ext == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if rows:
                # All CSV rows (including header) → single table Block
                blocks.append(Block(block_type="table", raw_content=rows, page=0))
                text = "\n".join(",".join(r) for r in rows)

        page = PageLayout(page_number=0, blocks=tuple(blocks))
        return BaseResult(pages=(page,), full_text=text, metadata={"source_format": ext.lstrip(".")})
