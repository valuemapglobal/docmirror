# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
ParseResult — Unified Document Parsing Output Contract
========================================================

The **single standard output model** for all document parsers in DocMirror.
All parsers (PDF, Image, Excel, Word, etc.) produce this structure.

Architecture:
    Zone 1 (pages)       → Content: "What I saw"
    Zone 2 (entities)    → Entities: "What I recognized"
    Zone 3 (parser_info) → Meta: "How I did it"
    Zone 4 (trust)       → Trust: "How much to trust it"
    Zone 5 (provenance)  → Provenance: "Where it came from"

Design principles:
    - **Strong typing**: No ``Dict[str, Any]`` — every field has a Pydantic type.
    - **Confidence penetration**: Cell → Row → Table → Page → Document.
    - **Separation of concerns**: Content / Entities / Meta are independent zones.
    - **Parser-agnostic**: DocMirror, Docling, PaddleOCR all output this structure.
    - **Backward compatible**: ``from_legacy_parser_output()`` bridges old formats.

Usage::

    from docmirror.models.entities.parse_result import ParseResult, PageContent

    result = ParseResult(
        pages=[PageContent(...)],
        entities=DocumentEntities(document_type="bank_statement"),
        parser_info=ParserInfo(parser_name="docmirror"),
    )
"""

from __future__ import annotations

import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from docmirror.models.tracking.mutation import Mutation


# ══════════════════════════════════════════════════════════════════════════════
# Enumerations
# ══════════════════════════════════════════════════════════════════════════════


class DataType(str, Enum):
    """Cell data type classification."""

    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    CURRENCY = "currency"
    EMPTY = "empty"
    MIXED = "mixed"


class RowType(str, Enum):
    """Table row semantic role."""

    HEADER = "header"
    DATA = "data"
    SUMMARY = "summary"
    SEPARATOR = "separator"
    SUBHEADER = "subheader"


class TextLevel(str, Enum):
    """Text hierarchy level."""

    TITLE = "title"
    H1 = "h1"
    H2 = "h2"
    H3 = "h3"
    BODY = "body"
    FOOTER = "footer"
    WATERMARK = "watermark"


class ExtractionMethod(str, Enum):
    """Document extraction method."""

    DIGITAL = "digital"
    OCR = "ocr"
    HYBRID = "hybrid"
    IMAGE = "image"


class ResultStatus(str, Enum):
    """Parse result status."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


# ══════════════════════════════════════════════════════════════════════════════
# Zone 1: Content — "What I saw"
# ══════════════════════════════════════════════════════════════════════════════


class CellValue(BaseModel):
    """
    Atomic unit of extraction — a single table cell.

    - ``text``: Raw OCR/extraction text, kept as-is.
    - ``cleaned``: Pre-cleaned text (stripped thousand separators, currency symbols).
    - ``numeric``: Parsed numeric value (if applicable).
    - ``confidence``: Extraction/OCR confidence [0.0, 1.0].
    """

    text: str = ""
    cleaned: str | None = None
    numeric: float | None = None
    confidence: float = 1.0
    bbox: list[float] | None = None
    data_type: DataType = DataType.TEXT

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "text": "15,000.00",
                    "cleaned": "15000.00",
                    "numeric": 15000.0,
                    "confidence": 0.97,
                    "data_type": "currency",
                },
            ]
        }


class TableRow(BaseModel):
    """
    A single table row with typed cells and semantic role.

    ``row_type`` distinguishes:
        - ``header``: Column name definition row.
        - ``data``: Core content row.
        - ``summary``: Aggregation row (e.g. "Total", "本页合计").
        - ``separator``: Divider/empty row.
        - ``subheader``: Sub-group title (e.g. "2024年7月").
    """

    cells: list[CellValue] = Field(default_factory=list)
    row_type: RowType = RowType.DATA
    confidence: float = 1.0
    source_page: int = 0

    @property
    def cell_texts(self) -> list[str]:
        """Convenience: list of all cell text values."""
        return [c.text for c in self.cells]


class TableBlock(BaseModel):
    """
    A complete table with headers, typed rows, and metadata.

    ``headers`` may be empty if the parser cannot determine the header row.
    ``rows`` contain all rows (data + summary + separators).
    """

    table_id: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[TableRow] = Field(default_factory=list)
    page: int = 1
    page_span: int = 1
    bbox: list[float] | None = None
    confidence: float = 1.0
    caption: str | None = None

    @property
    def data_rows(self) -> list[TableRow]:
        """Only data rows (excluding headers, summaries, separators)."""
        return [r for r in self.rows if r.row_type == RowType.DATA]

    @property
    def summary_rows(self) -> list[TableRow]:
        """Only summary/aggregation rows."""
        return [r for r in self.rows if r.row_type == RowType.SUMMARY]

    @property
    def row_count(self) -> int:
        """Number of data rows."""
        return len(self.data_rows)

    def to_dicts(self) -> list[dict[str, str]]:
        """
        Flatten data rows to ``[{column_name: cell_value}]``.

        Uses ``cleaned`` text if available, otherwise raw ``text``.
        """
        if not self.headers:
            return []
        return [
            {self.headers[i]: (c.cleaned or c.text) for i, c in enumerate(row.cells) if i < len(self.headers)}
            for row in self.data_rows
        ]

    class Config:
        json_schema_extra = {
            "example": {
                "table_id": "page1_table0",
                "headers": ["交易日期", "摘要", "交易金额", "余额"],
                "rows": [
                    {
                        "cells": [
                            {"text": "2024-06-20", "data_type": "date", "confidence": 0.99},
                            {"text": "工资", "data_type": "text", "confidence": 0.95},
                            {"text": "15,000.00", "numeric": 15000.0, "data_type": "currency"},
                            {"text": "135,530.00", "numeric": 135530.0, "data_type": "currency"},
                        ],
                        "row_type": "data",
                        "confidence": 0.95,
                    }
                ],
                "confidence": 0.96,
            }
        }


class TextBlock(BaseModel):
    """A text paragraph or heading with hierarchy level."""

    content: str = ""
    level: TextLevel = TextLevel.BODY
    confidence: float = 1.0
    bbox: list[float] | None = None


class KeyValuePair(BaseModel):
    """
    A key-value pair extracted from the document.

    Examples: "开户行: 建设银行", "纳税人识别号: 91110..."
    """

    key: str = ""
    value: str = ""
    confidence: float = 1.0
    bbox: list[float] | None = None


class PageContent(BaseModel):
    """
    Content of a single page — maintains page-level organization.

    Each page contains typed collections of tables, text blocks, and KV pairs.
    """

    page_number: int = 1
    tables: list[TableBlock] = Field(default_factory=list)
    texts: list[TextBlock] = Field(default_factory=list)
    key_values: list[KeyValuePair] = Field(default_factory=list)
    page_confidence: float = 1.0
    width: int | None = None
    height: int | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Zone 2: Entities — "What I recognized"
# ══════════════════════════════════════════════════════════════════════════════


class DocumentEntities(BaseModel):
    """
    Structured entities recognized from the document.

    Two layers:
        1. Universal fields — applicable to all document types.
        2. ``domain_specific`` — populated per document_type for domain-specific data.
    """

    document_type: str = "unknown"
    organization: str | None = None
    subject_name: str | None = None
    subject_id: str | None = None
    document_date: str | None = None
    period_start: str | None = None
    period_end: str | None = None

    domain_specific: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific fields populated by document type",
    )

    class Config:
        json_schema_extra = {
            "examples": {
                "bank_statement": {
                    "document_type": "bank_statement",
                    "organization": "中国建设银行",
                    "subject_name": "张三",
                    "subject_id": "6217XXXXXXXXXXXX",
                    "period_start": "2024-06-20",
                    "period_end": "2025-06-20",
                    "domain_specific": {
                        "account_number": "6217001820010XXXXXX",
                        "opening_balance": 12345.67,
                        "closing_balance": 135530.00,
                        "transaction_count": 347,
                        "currency": "CNY",
                    },
                },
            }
        }


# ══════════════════════════════════════════════════════════════════════════════
# Zone 3: Meta — "How I did it"
# ══════════════════════════════════════════════════════════════════════════════


class ParserInfo(BaseModel):
    """
    Parser self-description metadata.

    Middleware uses this to decide enhancement strategies:
        - ``extraction_method="ocr"`` → enable OCR repair middleware
        - ``overall_confidence < 0.7`` → trigger re-parse or degradation
        - ``table_engine="camelot"`` → skip table re-detection
    """

    parser_name: str = ""
    parser_version: str = ""
    elapsed_ms: float = 0
    page_count: int = 0

    extraction_method: ExtractionMethod = ExtractionMethod.DIGITAL
    ocr_engine: str | None = None
    table_engine: str | None = None

    overall_confidence: float = 1.0
    warnings: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Zone 4: Trust — "How much to trust it"
# (Absorbed from PerceptionResult.provenance.validation)
# ══════════════════════════════════════════════════════════════════════════════


class TrustResult(BaseModel):
    """
    Trust and validation assessment of the parsed content.

    Populated by the Validator middleware after the enhancement pipeline.
    """

    validation_score: float = 0.0
    validation_passed: bool = False
    trust_score: float = 0.0
    is_forged: bool | None = None
    forgery_reasons: list[str] = Field(default_factory=list)

    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-check validation breakdown (e.g. balance_continuity, date_order)",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Zone 5: Provenance — "Where it came from"
# (Absorbed from PerceptionResult.provenance.source)
# ══════════════════════════════════════════════════════════════════════════════


class ProvenanceInfo(BaseModel):
    """Source file provenance for audit trail."""

    file_type: str = ""
    file_size: int = 0
    checksum: str = ""
    mime_type: str = ""
    document_properties: dict[str, Any] = Field(
        default_factory=dict,
        description="PDF metadata, EXIF data, etc.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Envelope — Error handling
# ══════════════════════════════════════════════════════════════════════════════


class ErrorDetail(BaseModel):
    """Structured error information for failed parses."""

    code: str = ""
    message: str = ""
    details: str | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Top-Level: ParseResult
# ══════════════════════════════════════════════════════════════════════════════


class ParseResult(BaseModel):
    """
    Universal document parsing output contract.

    **The single standard output model for all DocMirror parsers.**

    Five zones + envelope:
        - Envelope: ``status``, ``confidence``, ``error``
        - Zone 1 (pages): Content — "What I saw"
        - Zone 2 (entities): Entities — "What I recognized"
        - Zone 3 (parser_info): Meta — "How I did it"
        - Zone 4 (trust): Trust — "How much to trust it"
        - Zone 5 (provenance): Provenance — "Where it came from"

    Usage::

        result = ParseResult(
            pages=[PageContent(page_number=1, tables=[...])],
            entities=DocumentEntities(document_type="bank_statement", ...),
            parser_info=ParserInfo(parser_name="docmirror", ...),
        )

        # Convenience accessors
        result.total_tables   # → 3
        result.total_rows     # → 45
        result.flatten_rows() # → [{col: val}, ...]
    """

    # ── Envelope ──
    status: ResultStatus = ResultStatus.SUCCESS
    confidence: float = 1.0
    error: ErrorDetail | None = None

    # ── Zone 1: Content ──
    pages: list[PageContent] = Field(default_factory=list)

    # ── Zone 2: Entities ──
    entities: DocumentEntities = Field(default_factory=DocumentEntities)

    # ── Zone 3: Meta ──
    parser_info: ParserInfo = Field(default_factory=ParserInfo)

    # ── Zone 4: Trust ──
    trust: TrustResult | None = None

    # ── Zone 5: Provenance ──
    provenance: ProvenanceInfo | None = None

    # ── Pipeline state (populated by middleware) ──
    mutations: list[Any] = Field(default_factory=list, exclude=True)
    processing_time: float = Field(default=0.0, exclude=True)
    errors: list[str] = Field(default_factory=list, exclude=True)

    # ── Computed properties ──

    @property
    def success(self) -> bool:
        """Whether parsing succeeded (fully or partially)."""
        return self.status in (ResultStatus.SUCCESS, ResultStatus.PARTIAL)

    @property
    def total_tables(self) -> int:
        """Total number of tables across all pages."""
        return sum(len(p.tables) for p in self.pages)

    @property
    def total_rows(self) -> int:
        """Total number of data rows across all tables."""
        return sum(t.row_count for p in self.pages for t in p.tables)

    @property
    def page_count(self) -> int:
        """Number of pages."""
        return len(self.pages)

    @property
    def full_text(self) -> str:
        """Reconstruct full text from all pages (texts + table markdown)."""
        return self._build_full_text()

    def all_tables(self) -> list[TableBlock]:
        """Collect all tables across pages."""
        return [t for p in self.pages for t in p.tables]

    @property
    def kv_entities(self) -> dict[str, str]:
        """Key-value entities from all pages (for SceneDetector/EntityExtractor)."""
        return self.all_key_values()

    @property
    def mutation_count(self) -> int:
        return len(self.mutations)

    @property
    def mutation_summary(self) -> dict[str, int]:
        """Summarize mutations per middleware."""
        summary: dict[str, int] = {}
        for m in self.mutations:
            summary[m.middleware_name] = summary.get(m.middleware_name, 0) + 1
        return summary

    # ── Middleware helper methods ──

    def record_mutation(
        self,
        middleware_name: str,
        target_block_id: str,
        field_changed: str,
        old_value: Any,
        new_value: Any,
        confidence: float = 1.0,
        reason: str = "",
    ) -> None:
        """Create and attach a Mutation audit record."""
        from docmirror.models.tracking.mutation import Mutation

        self.mutations.append(
            Mutation.create(
                middleware_name=middleware_name,
                target_block_id=target_block_id,
                field_changed=field_changed,
                old_value=old_value,
                new_value=new_value,
                confidence=confidence,
                reason=reason,
            )
        )

    def add_mutation(self, mutation: Any) -> None:
        """Append a pre-built Mutation object."""
        self.mutations.append(mutation)

    def add_error(self, error: str) -> None:
        """Record an error and downgrade status."""
        self.errors.append(error)
        if self.status == ResultStatus.SUCCESS:
            self.status = ResultStatus.PARTIAL

    def flatten_rows(self) -> list[dict[str, str]]:
        """
        Flatten all data rows into ``[{column_name: cell_value}]``.

        This is the direct data source for downstream structured consumers.
        """
        rows: list[dict[str, str]] = []
        for table in self.all_tables():
            rows.extend(table.to_dicts())
        return rows

    def all_key_values(self) -> dict[str, str]:
        """Collect all key-value pairs across pages into a single dict."""
        result: dict[str, str] = {}
        for page in self.pages:
            for kv in page.key_values:
                if kv.key:
                    result[kv.key] = kv.value
        return result

    # ── API output ──

    def to_api_dict(
        self,
        *,
        include_text: bool = False,
        request_id: str = "",
    ) -> dict[str, Any]:
        """Serialize to RESTful API dict per ``parser_interface.md`` v1.0.

        Structure::

            Success (HTTP 200):
            {
              "code": 200,
              "message": "success",
              "api_version": "1.0",
              "request_id": "...",
              "timestamp": "...",
              "data": { "document": {...}, "quality": {...} },
              "meta": { ... }
            }

            Failure (HTTP 422):
            {
              "code": 422,
              "message": "error",
              ...
              "error": { "type": "...", "detail": "..." },
              "meta": { ... }
            }

        Args:
            include_text: If True, include ``data.document.text`` with
                full markdown rendering of the document.
            request_id: Optional request ID for traceability.
        """
        from datetime import datetime, timezone

        # ── data.document ──
        # properties: flat dict from entities (exclude document_type + domain_specific)
        properties: dict[str, Any] = {}
        if self.entities.organization:
            properties["organization"] = self.entities.organization
        if self.entities.subject_name:
            properties["subject_name"] = self.entities.subject_name
        if self.entities.subject_id:
            properties["subject_id"] = self.entities.subject_id
        if self.entities.document_date:
            properties["document_date"] = self.entities.document_date
        if self.entities.period_start and self.entities.period_end:
            properties["period"] = f"{self.entities.period_start} ~ {self.entities.period_end}"
        elif self.entities.period_start:
            properties["period"] = self.entities.period_start
        # Merge domain_specific (currency, institution, etc.) into properties
        for k, v in self.entities.domain_specific.items():
            if k not in ("extracted_entities", "step_timings", "mutation_analysis"):
                properties[k] = v

        document: dict[str, Any] = {
            "type": self.entities.document_type,
            "properties": properties,
            "pages": self._build_api_pages(),
        }

        if include_text:
            document["text"] = self._build_full_text()
            document["text_format"] = "markdown"

        # ── data.quality ──
        quality: dict[str, Any] = {
            "confidence": self.confidence,
        }
        if self.trust:
            quality["trust_score"] = self.trust.trust_score
            quality["validation_passed"] = self.trust.validation_passed
            quality["issues"] = self.trust.forgery_reasons or []
        else:
            quality["trust_score"] = 1.0
            quality["validation_passed"] = True
            quality["issues"] = []

        # ── meta ──
        meta: dict[str, Any] = {
            "parser": self.parser_info.parser_name or "DocMirror",
            "version": self.parser_info.parser_version,
            "elapsed_ms": round(self.parser_info.elapsed_ms, 1),
            "extraction_method": self.parser_info.extraction_method.value,
            "page_count": self.parser_info.page_count or self.page_count,
            "table_count": self.total_tables,
            "row_count": self.total_rows,
        }
        if self.parser_info.table_engine:
            meta["table_engine"] = self.parser_info.table_engine
        if self.parser_info.ocr_engine:
            meta["ocr_engine"] = self.parser_info.ocr_engine
        if self.provenance:
            meta["provenance"] = self.provenance.model_dump(exclude_none=True)

        # ── Standard RESTful envelope ──
        # Top-level: code + message + request_id + timestamp (first thing consumer sees)
        # Then: data (business payload) + meta (parser diagnostics)
        if self.success:
            return {
                "code": 200,
                "message": "success",
                "api_version": "1.0",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "document": document,
                    "quality": quality,
                },
                "meta": meta,
            }
        else:
            error_msg = self.error.message if self.error else "parse failed"
            return {
                "code": 422,
                "message": "error",
                "api_version": "1.0",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": {
                    "type": self.status.value,
                    "detail": error_msg,
                },
                "meta": meta,
            }

    # ── Legacy bridge ──

    @classmethod
    def from_legacy_parser_output(cls, output: Any) -> ParseResult:
        """
        Bridge from legacy ``ParserOutput`` — for gradual migration.

        Converts the old ``document_structure`` list and ``key_entities``
        dict into the typed ParseResult structure.
        """
        pages: list[PageContent] = []
        current_page = PageContent(page_number=1)

        for item in getattr(output, "document_structure", None) or []:
            if not isinstance(item, dict):
                continue

            page_num = item.get("page", 1)
            if page_num != current_page.page_number:
                pages.append(current_page)
                current_page = PageContent(page_number=page_num)

            block_type = item.get("type", "")

            if block_type == "table":
                headers = item.get("headers", [])
                raw_rows = item.get("rows", [])
                if not headers and "data" in item:
                    data = item["data"]
                    if data:
                        headers = data[0]
                        raw_rows = data[1:]

                table = TableBlock(
                    table_id=f"page{page_num}_table{len(current_page.tables)}",
                    headers=headers,
                    page=page_num,
                )
                for r in raw_rows:
                    cells = [CellValue(text=str(v)) for v in (r if isinstance(r, list) else [])]
                    table.rows.append(TableRow(cells=cells))
                current_page.tables.append(table)

            elif block_type in ("text", "title", "heading"):
                level = {
                    "title": TextLevel.TITLE,
                    "heading": TextLevel.H1,
                }.get(block_type, TextLevel.BODY)
                current_page.texts.append(TextBlock(content=item.get("content", ""), level=level))

            elif block_type == "key_value":
                for k, v in (item.get("pairs") or {}).items():
                    current_page.key_values.append(KeyValuePair(key=k, value=str(v)))

        pages.append(current_page)

        # Entities from key_entities
        ent = DocumentEntities()
        ke = getattr(output, "key_entities", None) or {}
        ent.subject_name = ke.get("户名") or ke.get("account_holder", "")
        ent.organization = ke.get("银行") or ke.get("bank_name", "")
        ent.domain_specific = ke

        return cls(
            pages=pages,
            entities=ent,
            parser_info=ParserInfo(
                overall_confidence=getattr(output, "confidence", 1.0),
                page_count=len(pages),
            ),
        )

    # ── Internal helpers ──

    def _build_full_text(self) -> str:
        """Reconstruct full document text from page content."""
        parts: list[str] = []
        for page in self.pages:
            for text in page.texts:
                if text.content.strip():
                    if text.level in (TextLevel.TITLE, TextLevel.H1):
                        parts.append(f"# {text.content}")
                    elif text.level == TextLevel.H2:
                        parts.append(f"## {text.content}")
                    elif text.level == TextLevel.H3:
                        parts.append(f"### {text.content}")
                    else:
                        parts.append(text.content)
            for kv in page.key_values:
                parts.append(f"**{kv.key}**: {kv.value}")
            for table in page.tables:
                parts.append(self._table_to_markdown(table))
        return "\n\n".join(parts)

    @staticmethod
    def _table_to_markdown(table: TableBlock) -> str:
        """Render a table as Markdown."""
        if not table.headers:
            return ""
        lines = []
        lines.append("| " + " | ".join(table.headers) + " |")
        lines.append("|" + "|".join("---" for _ in table.headers) + "|")
        for row in table.data_rows:
            cells = [c.text for c in row.cells]
            # Pad to header count
            while len(cells) < len(table.headers):
                cells.append("")
            lines.append("| " + " | ".join(cells[: len(table.headers)]) + " |")
        return "\n".join(lines)

    def _build_api_pages(self) -> list[dict[str, Any]]:
        """Build API pages structure with full CellValue serialization."""
        api_pages: list[dict[str, Any]] = []
        for page in self.pages:
            api_page: dict[str, Any] = {
                "page_number": page.page_number,
            }

            # Tables with typed CellValue objects
            if page.tables:
                api_page["tables"] = []
                for table in page.tables:
                    api_page["tables"].append(
                        {
                            "table_id": table.table_id,
                            "headers": table.headers,
                            "rows": [
                                {
                                    "cells": [self._serialize_cell(c) for c in row.cells],
                                    "row_type": row.row_type.value,
                                    "confidence": row.confidence,
                                    "source_page": row.source_page,
                                }
                                for row in table.rows
                            ],
                            "page": table.page,
                            "row_count": table.row_count,
                            "confidence": table.confidence,
                        }
                    )

            # Text blocks with level
            if page.texts:
                api_page["texts"] = [
                    {
                        "content": text.content,
                        "level": text.level.value,
                        "confidence": text.confidence,
                    }
                    for text in page.texts
                ]

            # Key-value pairs
            if page.key_values:
                api_page["key_values"] = [
                    {
                        "key": kv.key,
                        "value": kv.value,
                        "confidence": kv.confidence,
                    }
                    for kv in page.key_values
                ]

            api_pages.append(api_page)
        return api_pages

    @staticmethod
    def _serialize_cell(cell: CellValue) -> dict[str, Any]:
        """Serialize CellValue to API dict — minimal output.

        Only ``text`` (always) and ``data_type`` (when not default "text").
        Consumer parses text according to data_type hint.
        """
        d: dict[str, Any] = {"text": cell.text}
        if cell.data_type.value != "text":
            d["data_type"] = cell.data_type.value
        return d


__all__ = [
    # Enums
    "DataType",
    "RowType",
    "TextLevel",
    "ExtractionMethod",
    "ResultStatus",
    # Zone 1: Content
    "CellValue",
    "TableRow",
    "TableBlock",
    "TextBlock",
    "KeyValuePair",
    "PageContent",
    # Zone 2: Entities
    "DocumentEntities",
    # Zone 3: Meta
    "ParserInfo",
    # Zone 4: Trust
    "TrustResult",
    # Zone 5: Provenance
    "ProvenanceInfo",
    # Envelope
    "ErrorDetail",
    # Top-level
    "ParseResult",
]
