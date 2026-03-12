"""
PerceptionResultBuilder \u2014 Unified Construction Engine
=====================================================

Supersedes the legacy dual conversion chain:
    EnhancedResult.to_parser_output() → PerceptionResult

Single-step construction:
    PerceptionResultBuilder.build(base, enhanced) → PerceptionResult
"""
from __future__ import annotations


import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..entities.perception_result import ContentBlock, ContentBlockType, DocumentContent, ErrorDetail, KeyValueBlock, PerceptionResult, Provenance, ParserStep, ResultStatus, SourceInfo, TableBlock, TextBlock, TimingInfo, ValidationResult

logger = logging.getLogger(__name__)


def _map_block(block) -> ContentBlock:
    """Convert a domain.Block to a ContentBlock for PerceptionResult."""
    btype = block.block_type
    page = block.page

    if btype == "table":
        raw = block.raw_content
        if isinstance(raw, list) and raw:
            # Guard: if raw is a flat list of strings (e.g. ['客户', '人数上限']),
            # wrap it into a single-row 2D list so TableBlock validation succeeds.
            if raw and not isinstance(raw[0], (list, tuple)):
                raw = [raw]
            headers = raw[0] if raw else []
            rows = raw[1:] if len(raw) > 1 else []
        else:
            headers, rows = [], []

        bbox_raw = getattr(block, "bbox", None)
        has_bbox = bbox_raw and len(bbox_raw) == 4 and any(bbox_raw)
        bbox = tuple(bbox_raw) if has_bbox else None

        return ContentBlock(
            type=ContentBlockType.TABLE,
            page=page,
            table=TableBlock(
                headers=headers, rows=rows, page=page, bbox=bbox
            ),
        )

    elif btype == "key_value":
        pairs = {}
        if isinstance(block.raw_content, dict):
            pairs = block.raw_content
        return ContentBlock(
            type=ContentBlockType.KEY_VALUE,
            page=page,
            key_value=KeyValueBlock(pairs=pairs),
        )

    elif btype == "title":
        # Title blocks → HEADING type with hierarchy level
        content = ""
        if isinstance(block.raw_content, str):
            content = block.raw_content
        level = getattr(block, "heading_level", 1) or 1
        return ContentBlock(
            type=ContentBlockType.HEADING,
            page=page,
            text=TextBlock(content=content, level=level),
        )

    elif btype == "image":
        # Image blocks → IMAGE type with caption as content
        caption = getattr(block, "caption", "") or ""
        return ContentBlock(
            type=ContentBlockType.IMAGE,
            page=page,
            text=TextBlock(content=caption, level=0),
        )

    else:
        # text / footer / formula → TextBlock
        content = ""
        if isinstance(block.raw_content, str):
            content = block.raw_content
        level = getattr(block, "heading_level", 0) or 0
        return ContentBlock(
            type=ContentBlockType.TEXT,
            page=page,
            text=TextBlock(content=content, level=level),
        )






class PerceptionResultBuilder:
    """
    Unified constructor for PerceptionResult objects.

    Builds from BaseResult (+ optional EnhancedResult) in a single step.

    Usage::

        # Direct simple path (Non-PDF workflows)
        result = PerceptionResultBuilder.build(
            base_result, file_path="a.xlsx", file_type="excel"
        )

        # PDF Enhanced processing workflow
        result = PerceptionResultBuilder.build(
            base_result, enhanced=enhanced_result,
            file_path="a.pdf", file_type="pdf",
        )
    """

    @staticmethod
    def build(
        base_result,
        *,
        enhanced=None,
        file_path: str = "",
        file_type: str = "",
        file_size: int = 0,
        parser_name: str = "",
        elapsed_ms: float = 0.0,
        started_at: Optional[datetime] = None,
        mime_type: str = "",
        checksum: str = "",
        is_forged: Optional[bool] = None,
        forgery_reasons: Optional[List[str]] = None,
    ) -> PerceptionResult:
        """
        Rapid one-step compilation.

        Args:
            base_result: CoreExtractor's initial BaseResult output.
            enhanced:    Optional EnhancedResult with middleware enrichments.
            Other Params: Context payloads originating from the dispatcher.
        """
        meta = base_result.metadata if base_result else {}

        # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
        # 1. Envelope Layer
        # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
        if enhanced is not None:
            status_map = {
                "success": ResultStatus.SUCCESS,
                "partial": ResultStatus.PARTIAL,
                "failed": ResultStatus.FAILURE
            }
            result_status = status_map.get(
                enhanced.status, ResultStatus.FAILURE
            )
            confidence = (
                1.0 if enhanced.status == "success"
                else (0.5 if enhanced.status == "partial" else 0.0)
            )
            error_detail = None
            if enhanced.errors:
                error_detail = ErrorDetail(message="; ".join(enhanced.errors))
            p_name = parser_name or "DocMirror"
            p_elapsed = elapsed_ms or enhanced.processing_time
        else:
            result_status = ResultStatus.SUCCESS
            confidence = 1.0
            error_detail = None
            p_name = parser_name
            p_elapsed = elapsed_ms

        timing = TimingInfo(
            started_at=started_at, parser_name=p_name, elapsed_ms=p_elapsed
        )

        # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
        # 2. Content Layer
        # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
        content_blocks = []
        if base_result:
            content_blocks = [_map_block(b) for b in base_result.all_blocks]

        # Merge base optionally
        entities = {}
        if base_result:
            entities.update(base_result.entities)
        if enhanced is not None:
            entities.update(enhanced.enhanced_data.get("extracted_entities", {}))

        txt_fmt = "plain"
        if base_result and base_result.full_text.startswith("|"):
            txt_fmt = "markdown"

        final_text = base_result.full_text if base_result else ""

        content = DocumentContent(
            text=final_text,
            text_format=txt_fmt,
            blocks=content_blocks,
            entities={k: str(v) for k, v in entities.items()},
            page_count=base_result.page_count if base_result else 0,
        )

        # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
        # 3. Provenance Layer
        # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
        # Document properties
        document_properties = {}
        target_keys = (
            "format", "producer", "creator", "creationDate", "modDate",
            "title", "author", "subject", "keywords", "trapped", "encryption"
        )
        for k in target_keys:
            if k in meta:
                document_properties[k] = str(meta[k]) if meta[k] is not None else ""

        # Validation results
        validation = None
        if enhanced is not None:
            vr = enhanced.validation_result
            if vr:
                validation = ValidationResult(
                    l2_score=vr.get("total_score"),
                    l2_passed=vr.get("passed"),
                    l2_details=vr.get("details"),
                    image_quality=vr.get("image_quality"),
                )
            # L1 anomaly metrics
            if any(k in meta for k in (
                    "l1_anomaly_count", "l1_repaired_count")):
                if validation is None:
                    validation = ValidationResult()
                validation.l1_anomaly_count = meta.get("l1_anomaly_count", 0)
                validation.l1_repaired_count = meta.get("l1_repaired_count", 0)
                validation.l1_reverted_count = meta.get("l1_reverted_count", 0)
                validation.l1_llm_used = meta.get("l1_llm_used", False)
                validation.balance_truncation_repaired = meta.get(
                    "balance_truncation_repaired", 0
                )

        # Forgery detection
        if is_forged is not None:
            if validation is None:
                validation = ValidationResult()
            validation.is_forged = is_forged
            validation.forgery_reasons = forgery_reasons or []

        # Diagnostics
        from ._shared import build_diagnostics
        diagnostics = build_diagnostics(meta)

        p_chain = []
        if p_name:
            p_chain = [
                ParserStep(parser=p_name, action="parse", elapsed_ms=p_elapsed)
            ]

        provenance = Provenance(
            source=SourceInfo(
                file_path=file_path,
                file_size=file_size,
                file_type=file_type,
                mime_type=mime_type or None,
                checksum=checksum or None,
            ),
            parser_chain=p_chain,
            validation=validation,
            diagnostics=diagnostics,
            document_properties=document_properties,
        )

        # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
        # 4. Domain Mapping
        # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
        domain = None
        if enhanced is not None:
            cat = enhanced.scene
            if cat == "bank_statement":
                try:
                    from ..entities.domain_models import (
                        BankStatementData, DomainData
                    )
                    from ...configs.domain_registry import normalize_entity_keys
                    norm_ent = normalize_entity_keys(entities)
                    acc_h = norm_ent.get(
                        "Account name", norm_ent.get("Account name", "")
                    )
                    acc_n = norm_ent.get(
                        "Account number", norm_ent.get("Card number", "")
                    )
                    bs = BankStatementData(
                        account_holder=str(acc_h),
                        account_number=str(acc_n),
                        bank_name=str(norm_ent.get("bank_name", "")),
                        query_period=str(norm_ent.get("Query period", "")),
                        currency=str(norm_ent.get("Currency", "CNY")) or "CNY",
                    )
                    domain = DomainData(
                        document_type="bank_statement", bank_statement=bs
                    )
                except ImportError:
                    pass

        # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
        # 5. Assemble Final Payload
        # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
        pr = PerceptionResult(
            status=result_status,
            confidence=confidence,
            timing=timing,
            error=error_detail,
            content=content,
            domain=domain,
            scene=enhanced.scene if enhanced else "unknown",
            provenance=provenance,
        )

        # Attach enhanced result for backward-compatible access
        if enhanced is not None:
            pr._enhanced = enhanced

        return pr
