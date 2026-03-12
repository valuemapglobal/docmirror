"""
Excel Adapter — .xlsx → BaseResult
====================================

Extracts worksheet data from Excel files using openpyxl.

Processing logic:
    1. Opens the workbook in read-only/data-only mode (formulas are
       resolved to their cached values).
    2. Each worksheet becomes a separate ``PageLayout`` (page index
       matches the sheet order).
    3. For each sheet:
       - A ``title`` Block is created with the sheet name (heading level 2).
       - All non-empty rows are collected into a 2D list → ``table`` Block.
       - Empty sheets (all rows blank) are skipped entirely.
    4. The full_text preview includes the first 5 rows of each sheet
       in a pipe-delimited format.

Metadata includes:
    - source_format: "excel"
    - sheet_names: list of all worksheet names
    - table_count: total tables extracted
"""
from __future__ import annotations


import logging
from pathlib import Path

from docmirror.framework.base import BaseParser
from docmirror.models.domain import BaseResult, Block, PageLayout

logger = logging.getLogger(__name__)


class ExcelAdapter(BaseParser):
    """Excel (.xlsx/.xls) format adapter — native parsing prioritizing `openpyxl`.

    Processing strategy:
        1. **Modern formats (.xlsx):** Direct deep extraction via ``openpyxl``
           preserving tabular accuracy and sheet separations natively.
        2. **Legacy formats (.xls):** Automatically transcoded to PDF via LibreOffice,
           then processed through the standard PDF pipeline.
    """

    async def perceive(self, file_path: Path, **context):
        """
        Native primary extraction for modern .xlsx.
        """
        return await super().perceive(file_path, **context)

    async def to_base_result(self, file_path: Path) -> BaseResult:
        """
        Parse an .xlsx file into a BaseResult.

        Each worksheet maps to a PageLayout. Cell values are stringified
        (None → empty string). Sheets with no non-empty rows are skipped.
        """
        import openpyxl
        wb = openpyxl.load_workbook(str(file_path), data_only=True)

        pages = []
        text_parts = []

        for idx, sheet_name in enumerate(wb.sheetnames):
            sheet = wb[sheet_name]

            # Collect non-empty rows, converting all cell values to strings
            rows = []
            for row in sheet.iter_rows(values_only=True):
                row_data = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in row_data):
                    rows.append(row_data)

            if not rows:
                continue  # skip empty sheets

            # Create a title block with the sheet name
            title_block = Block(
                block_type="title",
                raw_content=f"Sheet: {sheet_name}",
                page=idx,
                heading_level=2,
            )

            # Create a table block with all row data
            table_block = Block(
                block_type="table",
                raw_content=rows,
                page=idx,
            )

            pages.append(PageLayout(
                page_number=idx,
                blocks=(title_block, table_block),
            ))

            # Build text preview (first 5 rows, pipe-delimited)
            text_parts.append(
                f"## {sheet_name}\n" + "\n".join(" | ".join(r) for r in rows[:5])
            )

        return BaseResult(
            pages=tuple(pages),
            full_text="\n\n".join(text_parts),
            metadata={
                "source_format": "excel", 
                "sheet_names": wb.sheetnames,
                "table_count": len(pages),
            },
        )
