# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Structured Data Adapter — JSON/CSV → ParseResult
===================================================

Handles structured data files that already have a well-defined schema.

Processing logic by format:

**JSON (.json)**:
    - Loads the entire file into a Python object.
    - If the root object is a dict, creates KeyValuePairs.
    - If the root object is a list of dicts, creates a TableBlock.

**CSV (.csv)**:
    - Reads all rows via Python's csv.reader.
    - First row treated as headers, rest as data rows with typed CellValue.

Both formats produce a single-page ParseResult.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import List

from docmirror.framework.base import BaseParser

logger = logging.getLogger(__name__)

# Currency-like pattern
_CURRENCY_RE = re.compile(r"^[¥$€£₹]?\s*-?\d{1,3}(,\d{3})*(\.\d+)?$")


class StructuredAdapter(BaseParser):
    """Structured data (JSON/CSV) format adapter."""

    async def to_parse_result(self, file_path: Path, **kwargs) -> ParseResult:
        """
        Parse a JSON or CSV file into a ParseResult.

        JSON dicts → KeyValuePairs; JSON list-of-dicts → TableBlock.
        CSV files → TableBlock with typed CellValue.
        """
        from docmirror.models.entities.parse_result import (
            CellValue,
            DataType,
            KeyValuePair,
            PageContent,
            ParseResult,
            ParserInfo,
            TableBlock,
            TableRow,
            TextBlock,
            TextLevel,
        )

        ext = file_path.suffix.lower()
        logger.info(f"[StructuredAdapter] Starting extraction for {ext} file: {file_path}")

        texts: list[TextBlock] = []
        tables: list[TableBlock] = []
        key_values: list[KeyValuePair] = []

        if ext == ".json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                for k, v in data.items():
                    key_values.append(KeyValuePair(key=str(k), value=str(v)))
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                # List of dicts → table
                headers = list(data[0].keys())
                rows = []
                for record in data:
                    cells = [CellValue(text=str(record.get(h, "")), data_type=DataType.TEXT) for h in headers]
                    rows.append(TableRow(cells=cells))
                tables.append(
                    TableBlock(
                        table_id="json_records",
                        headers=headers,
                        rows=rows,
                        page=0,
                    )
                )

        elif ext == ".csv":
            with open(file_path, encoding="utf-8") as f:
                csv_rows = list(csv.reader(f))

            if csv_rows:
                headers = csv_rows[0]
                data_rows = []
                for row_data in csv_rows[1:]:
                    cells = [_classify_csv_cell(v) for v in row_data]
                    if any(c.text for c in cells):
                        data_rows.append(TableRow(cells=cells))

                tables.append(
                    TableBlock(
                        table_id="csv_data",
                        headers=headers,
                        rows=data_rows,
                        page=0,
                    )
                )

        page = PageContent(
            page_number=0,
            texts=texts,
            tables=tables,
            key_values=key_values,
        )

        return ParseResult(
            pages=[page],
            parser_info=ParserInfo(
                parser_name="StructuredAdapter",
                page_count=1,
                overall_confidence=1.0,
            ),
        )


def _classify_csv_cell(value: str) -> CellValue:
    """Classify a CSV cell string into typed CellValue."""
    from docmirror.models.entities.parse_result import CellValue, DataType

    text = value.strip()
    if not text:
        return CellValue(text="", data_type=DataType.EMPTY)

    # Try numeric
    try:
        numeric = float(text)
        return CellValue(text=text, cleaned=text, numeric=numeric, data_type=DataType.NUMBER)
    except ValueError:
        pass

    # Try currency-like
    if _CURRENCY_RE.match(text):
        cleaned = re.sub(r"[¥$€£₹,\s]", "", text)
        try:
            numeric = float(cleaned)
            return CellValue(text=text, cleaned=cleaned, numeric=numeric, data_type=DataType.CURRENCY)
        except ValueError:
            pass

    return CellValue(text=text, data_type=DataType.TEXT)
