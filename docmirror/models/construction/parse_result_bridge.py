# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
ParseResult Bridge — Bidirectional Conversion Layer
=====================================================

Bridges between the new ``ParseResult`` unified model and the existing
``BaseResult`` / ``EnhancedResult`` models.

Conversion paths:

    ParseResult  →  BaseResult       (for middleware pipeline consumption)
    EnhancedResult  →  ParseResult   (middleware pipeline output → final model)

Usage::

    from docmirror.models.construction.parse_result_bridge import ParseResultBridge

    # After middleware pipeline: EnhancedResult → ParseResult
    parse_result = ParseResultBridge.from_enhanced_result(enhanced, **context)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _infer_cell_value(text: str) -> CellValue:
    """Infer CellValue type from raw text string.

    Returns CellValue with proper data_type, numeric, and cleaned fields.
    """
    import re

    from docmirror.models.entities.parse_result import CellValue, DataType

    text = str(text).strip()
    if not text:
        return CellValue(text=text, data_type=DataType.EMPTY)

    # Date patterns: 2025-03-27, 2025/03/27, 2025年03月27日
    if re.match(r"^\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?$", text):
        return CellValue(text=text, data_type=DataType.DATE)

    # Time pattern: 14:21:48
    if re.match(r"^\d{2}:\d{2}(:\d{2})?$", text):
        return CellValue(text=text, data_type=DataType.TEXT)

    # Currency/Number: try parsing
    cleaned = text.replace(",", "").replace("，", "").replace(" ", "")
    # Remove currency symbols
    cleaned = re.sub(r"^[¥$€£]", "", cleaned)

    # Try numeric parse
    try:
        float(cleaned)
        # Long digit-only strings (>10 chars, no decimal) are identifiers
        # (account numbers, ID numbers, invoice codes), not values
        if re.match(r"^\d{10,}$", cleaned):
            return CellValue(text=text, data_type=DataType.TEXT)
        # Determine if currency (has comma formatting or decimal places typical of money)
        has_comma = "," in text or "，" in text
        has_decimal = "." in cleaned and len(cleaned.split(".")[-1]) == 2
        if has_comma or has_decimal:
            return CellValue(text=text, data_type=DataType.CURRENCY)
        else:
            return CellValue(text=text, data_type=DataType.NUMBER)
    except (ValueError, TypeError):
        pass

    return CellValue(text=text, data_type=DataType.TEXT)


def _blocks_to_pages(base: BaseResult):
    """Convert BaseResult pages/blocks → List[PageContent] for ParseResult.

    Mapping:
        - Block(type=table, raw_content=List[List[str]]) → TableBlock with typed CellValue
        - Block(type=text/title) → TextBlock with heading level
        - Block(type=key_value, raw_content=dict) → KeyValuePair
    """
    from docmirror.models.entities.parse_result import (
        CellValue,
        KeyValuePair,
        PageContent,
        RowType,
        TableBlock,
        TableRow,
        TextBlock,
        TextLevel,
    )

    pages = []
    for page_layout in base.pages:
        tables = []
        texts = []
        key_values = []

        for block in page_layout.blocks:
            if block.block_type == "table" and isinstance(block.raw_content, list):
                raw = block.raw_content
                headers = []
                rows = []
                if raw:
                    headers = [str(h) for h in raw[0]]
                    for row_data in raw[1:]:
                        if isinstance(row_data, list):
                            cells = [_infer_cell_value(v) for v in row_data]
                            rows.append(
                                TableRow(
                                    cells=cells,
                                    row_type=RowType.DATA,
                                    source_page=page_layout.page_number,
                                )
                            )
                tables.append(
                    TableBlock(
                        table_id=f"page{page_layout.page_number}_table{len(tables)}",
                        headers=headers,
                        rows=rows,
                        page=page_layout.page_number,
                    )
                )

            elif block.block_type in ("text", "title") and isinstance(block.raw_content, str):
                level = TextLevel.BODY
                if block.block_type == "title" or block.heading_level == 1:
                    level = TextLevel.H1
                elif block.heading_level == 2:
                    level = TextLevel.H2
                elif block.heading_level == 3:
                    level = TextLevel.H3
                texts.append(
                    TextBlock(
                        content=block.raw_content,
                        level=level,
                    )
                )

            elif block.block_type == "key_value" and isinstance(block.raw_content, dict):
                for k, v in block.raw_content.items():
                    key_values.append(KeyValuePair(key=str(k), value=str(v)))

            elif block.block_type == "footer" and isinstance(block.raw_content, str):
                texts.append(
                    TextBlock(
                        content=block.raw_content,
                        level=TextLevel.FOOTER,
                    )
                )

        pages.append(
            PageContent(
                page_number=page_layout.page_number,
                tables=tables,
                texts=texts,
                key_values=key_values,
            )
        )

    return pages


class ParseResultBridge:
    """Unified converter between ParseResult and internal models.

    Primary methods:
        - ``to_base_result(pr)``       → ParseResult → BaseResult (for middleware)
        - ``from_base_result(base)``    → BaseResult → ParseResult (for adapters)
        - ``merge_enhanced_into(pr, e)``→ merge middleware results into ParseResult
    """

    # ══════════════════════════════════════════════════════════════════════
    # BaseResult → ParseResult (for adapters that extract to BaseResult)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def from_base_result(base: BaseResult) -> ParseResult:
        """
        Convert BaseResult → ParseResult.

        Used by adapters (e.g. PDFAdapter) that extract to BaseResult
        and need to convert to ParseResult before the middleware pipeline.

        Mapping:
            - Block(type=table) → TableBlock with CellValue per cell
            - Block(type=text/title) → TextBlock with appropriate level
            - Block(type=key_value) → KeyValuePair
        """
        from docmirror.models.entities.parse_result import (
            ParseResult,
            ParserInfo,
        )

        pages = _blocks_to_pages(base)
        return ParseResult(
            pages=pages,
            parser_info=ParserInfo(
                parser_name="DocMirror",
                page_count=len(base.pages),
            ),
        )

    @staticmethod
    def to_base_result(pr: ParseResult) -> BaseResult:
        """
        Convert ParseResult → BaseResult for middleware pipeline consumption.

        Mapping:
            - PageContent → PageLayout (1:1)
            - TableBlock.rows → Block(block_type="table", raw_content=List[List[str]])
            - TextBlock → Block(block_type="text"/"title")
            - KeyValuePair → Block(block_type="key_value", raw_content={key: value})
        """
        from docmirror.models.entities.domain import BaseResult, Block, PageLayout

        pages = []
        reading_order = 0

        for page_content in pr.pages:
            blocks = []

            for text in page_content.texts:
                from docmirror.models.entities.parse_result import TextLevel

                block_type = "title" if text.level in (TextLevel.TITLE, TextLevel.H1) else "text"
                blocks.append(
                    Block(
                        block_type=block_type,
                        raw_content=text.content,
                        page=page_content.page_number,
                        reading_order=reading_order,
                        heading_level=(
                            1
                            if text.level == TextLevel.TITLE
                            else 1
                            if text.level == TextLevel.H1
                            else 2
                            if text.level == TextLevel.H2
                            else 3
                            if text.level == TextLevel.H3
                            else None
                        ),
                    )
                )
                reading_order += 1

            for kv in page_content.key_values:
                blocks.append(
                    Block(
                        block_type="key_value",
                        raw_content={kv.key: kv.value},
                        page=page_content.page_number,
                        reading_order=reading_order,
                    )
                )
                reading_order += 1

            for table in page_content.tables:
                # Convert CellValue rows to List[List[str]]
                raw_rows = []
                if table.headers:
                    raw_rows.append(table.headers)
                for row in table.rows:
                    raw_rows.append([c.text for c in row.cells])

                blocks.append(
                    Block(
                        block_type="table",
                        raw_content=raw_rows,
                        page=page_content.page_number,
                        reading_order=reading_order,
                    )
                )
                reading_order += 1

            pages.append(
                PageLayout(
                    page_number=page_content.page_number,
                    blocks=tuple(blocks),
                )
            )

        # Build full text from ParseResult
        full_text = pr.full_text

        # Build metadata from entities + parser_info
        metadata: dict[str, Any] = {
            "source_format": pr.provenance.file_type if pr.provenance else "unknown",
        }
        # Carry entities into metadata for downstream middleware access
        if pr.entities.organization:
            metadata["organization"] = pr.entities.organization
        if pr.entities.subject_name:
            metadata["subject_name"] = pr.entities.subject_name

        return BaseResult(
            pages=tuple(pages),
            full_text=full_text,
            metadata=metadata,
        )

    # ══════════════════════════════════════════════════════════════════════
    # EnhancedResult → ParseResult (middleware pipeline output → final)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def from_enhanced_result(
        enhanced: EnhancedResult,
        **context: Any,
    ) -> ParseResult:
        """
        Convert EnhancedResult → ParseResult after middleware pipeline.

        Mapping:
            - BaseResult.pages → ParseResult.pages (Block → typed content)
            - enhanced.scene → entities.document_type
            - enhanced.institution → entities.organization
            - enhanced.enhanced_data["entities"] → entities.domain_specific
            - enhanced.enhanced_data["validation"] → trust
            - enhanced.mutations → parser_info.warnings (summary)
            - context (file_type, checksum, etc.) → provenance
        """
        from docmirror.models.entities.parse_result import (
            CellValue,
            DataType,
            DocumentEntities,
            ErrorDetail,
            ExtractionMethod,
            KeyValuePair,
            PageContent,
            ParseResult,
            ParserInfo,
            ProvenanceInfo,
            ResultStatus,
            RowType,
            TableBlock,
            TableRow,
            TextBlock,
            TextLevel,
            TrustResult,
        )

        base = enhanced.base_result
        if base is None:
            return ParseResult(
                status=ResultStatus.FAILURE,
                confidence=0.0,
                error=ErrorDetail(code="NO_BASE_RESULT", message="No base result available"),
            )

        # ── Build pages from BaseResult blocks ──
        pages = _blocks_to_pages(base)

        # ── Build entities from enhanced data ──
        entities = _build_entities(enhanced)

        # ── Build parser info ──
        meta = base.metadata or {}
        diag = meta.get("_diagnostics", {})
        meta.get("pre_analysis", {})

        extraction_method = ExtractionMethod.DIGITAL
        method_str = diag.get("extraction_method", "").lower()
        if method_str == "ocr":
            extraction_method = ExtractionMethod.OCR
        elif method_str == "hybrid":
            extraction_method = ExtractionMethod.HYBRID
        elif method_str == "image":
            extraction_method = ExtractionMethod.IMAGE

        parser_info = ParserInfo(
            parser_name=context.get("parser_name", "DocMirror"),
            parser_version=_get_version(),
            elapsed_ms=enhanced.processing_time,
            page_count=len(base.pages),
            extraction_method=extraction_method,
            table_engine=diag.get("template_source", None),
            overall_confidence=1.0,
            warnings=[str(e) for e in enhanced.errors] if enhanced.errors else [],
        )

        # ── Build trust from validation ──
        trust = _build_trust(enhanced, context)

        # ── Build provenance from context ──
        provenance = ProvenanceInfo(
            file_type=context.get("file_type", ""),
            file_size=context.get("file_size", 0),
            checksum=context.get("checksum", ""),
            mime_type=context.get("mime_type", ""),
            document_properties=meta.get("document_properties", {}),
        )

        # ── Map status ──
        status_map = {
            "success": ResultStatus.SUCCESS,
            "partial": ResultStatus.PARTIAL,
            "failed": ResultStatus.FAILURE,
        }
        status = status_map.get(enhanced.status, ResultStatus.SUCCESS)

        # ── Compute confidence ──
        validation = enhanced.enhanced_data.get("validation", {})
        confidence = (
            validation.get("trust_score", 1.0) / 100.0
            if validation.get("trust_score", 0) > 1
            else validation.get("trust_score", 1.0)
        )

        return ParseResult(
            status=status,
            confidence=confidence,
            pages=pages,
            entities=entities,
            parser_info=parser_info,
            trust=trust,
            provenance=provenance,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Merge enhanced data into existing ParseResult (Path A support)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def merge_enhanced_into(
        pr: ParseResult,
        enhanced: EnhancedResult,
        **context: Any,
    ) -> ParseResult:
        """
        Merge middleware-enhanced data into an existing ParseResult.

        **Purpose**: When an adapter produces a direct ParseResult via
        ``to_parse_result()`` (Path A), it has Cell-level precision
        (typed CellValue, numeric, data_type, confidence) that would be
        lost in a round-trip through BaseResult. This method keeps the
        original pages intact while injecting enhanced data.

        What gets merged:
            - entities (scene, institution, domain_specific)
            - parser_info (elapsed_ms, warnings, extraction_method)
            - trust (validation results)
            - provenance (file metadata)
            - confidence (from validation)

        What is preserved:
            - pages (Cell-level precision intact)
            - status (kept from original PR)
            - error (kept from original PR)
        """
        from docmirror.models.entities.parse_result import (
            DocumentEntities,
            ExtractionMethod,
            ParserInfo,
            ProvenanceInfo,
            TrustResult,
        )

        meta = (enhanced.base_result.metadata or {}) if enhanced.base_result else {}

        # ── Merge entities ──
        entities = _build_entities(enhanced)
        # Preserve any domain_specific data the adapter already set
        if pr.entities.domain_specific:
            merged_domain = dict(entities.domain_specific)
            merged_domain.update(pr.entities.domain_specific)
            entities.domain_specific = merged_domain
        # Preserve adapter-set fields if middleware didn't detect them
        if not entities.organization and pr.entities.organization:
            entities.organization = pr.entities.organization
        if not entities.subject_name and pr.entities.subject_name:
            entities.subject_name = pr.entities.subject_name

        # ── Merge parser_info ──
        diag = meta.get("_diagnostics", {})
        extraction_method = ExtractionMethod.DIGITAL
        method_str = diag.get("extraction_method", "").lower()
        if method_str == "ocr":
            extraction_method = ExtractionMethod.OCR
        elif method_str == "hybrid":
            extraction_method = ExtractionMethod.HYBRID

        parser_info = ParserInfo(
            parser_name=pr.parser_info.parser_name or context.get("parser_name", "DocMirror"),
            parser_version=pr.parser_info.parser_version or _get_version(),
            elapsed_ms=enhanced.processing_time or pr.parser_info.elapsed_ms,
            page_count=pr.parser_info.page_count or len(pr.pages),
            extraction_method=extraction_method,
            table_engine=diag.get("template_source") or pr.parser_info.table_engine,
            overall_confidence=pr.parser_info.overall_confidence,
            warnings=[str(e) for e in enhanced.errors] if enhanced.errors else pr.parser_info.warnings,
        )

        # ── Merge trust ──
        trust = _build_trust(enhanced, context) or pr.trust

        # ── Merge provenance ──
        provenance = ProvenanceInfo(
            file_type=context.get("file_type", pr.provenance.file_type if pr.provenance else ""),
            file_size=context.get("file_size", pr.provenance.file_size if pr.provenance else 0),
            checksum=context.get("checksum", pr.provenance.checksum if pr.provenance else ""),
            mime_type=context.get("mime_type", pr.provenance.mime_type if pr.provenance else ""),
            document_properties=meta.get("document_properties", {}),
        )

        # ── Merge confidence ──
        validation = enhanced.enhanced_data.get("validation", {})
        trust_score = validation.get("trust_score", 0)
        confidence = trust_score / 100.0 if trust_score > 1 else trust_score if trust_score else pr.confidence

        # Build new ParseResult preserving original pages (Cell-level precision)
        from docmirror.models.entities.parse_result import ParseResult

        return ParseResult(
            status=pr.status,
            confidence=confidence,
            error=pr.error,
            pages=pr.pages,  # ← preserved: typed CellValue precision intact
            entities=entities,
            parser_info=parser_info,
            trust=trust,
            provenance=provenance,
        )
