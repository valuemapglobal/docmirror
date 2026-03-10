"""
PerceptionResultBuilder — 统一构建器
======================================

取代原有的双重转换链:
    EnhancedResult.to_parser_output() → ParserOutput.to_perception_result()

改为单步构建:
    PerceptionResultBuilder.build(base_result, enhanced=...) → PerceptionResult
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..entities.perception_result import (
    ContentBlock,
    ContentBlockType,
    Diagnostics,
    DocumentContent,
    ErrorDetail,
    KeyValueBlock,
    PerceptionResult,
    Provenance,
    ParserStep,
    ResultStatus,
    SourceInfo,
    TableBlock,
    TextBlock,
    TimingInfo,
    ValidationResult,
)

logger = logging.getLogger(__name__)


def _map_block(block) -> ContentBlock:
    """将 domain.Block → schemas.ContentBlock (单步映射)。"""
    btype = block.block_type
    page = block.page

    if btype == "table":
        raw = block.raw_content
        if isinstance(raw, list) and raw:
            headers = raw[0] if raw else []
            rows = raw[1:] if len(raw) > 1 else []
        else:
            headers, rows = [], []

        bbox_raw = getattr(block, "bbox", None)
        bbox = tuple(bbox_raw) if bbox_raw and len(bbox_raw) == 4 and any(bbox_raw) else None

        return ContentBlock(
            type=ContentBlockType.TABLE,
            page=page,
            table=TableBlock(headers=headers, rows=rows, page=page, bbox=bbox),
        )

    elif btype == "key_value":
        pairs = block.raw_content if isinstance(block.raw_content, dict) else {}
        return ContentBlock(
            type=ContentBlockType.KEY_VALUE,
            page=page,
            key_value=KeyValueBlock(pairs=pairs),
        )

    else:
        # text / title / footer → TextBlock
        content = block.raw_content if isinstance(block.raw_content, str) else ""
        level = getattr(block, "heading_level", 0) or 0
        return ContentBlock(
            type=ContentBlockType.TEXT,
            page=page,
            text=TextBlock(content=content, level=level),
        )


def _overlay_standardized_tables(
    blocks: List[ContentBlock],
    std_tables: List[Dict[str, Any]],
) -> None:
    """
    将中间件标准化后的表格覆盖到 ContentBlock 中。

    策略: 用最大标准化表格替换第一个表格块。
    """
    if not std_tables or not blocks:
        return

    main = max(std_tables, key=lambda t: t.get("row_count", 0))
    for b in blocks:
        if b.type == ContentBlockType.TABLE and b.table is not None:
            b.table.headers = main.get("headers", [])
            b.table.rows = main.get("rows", [])
            break


class PerceptionResultBuilder:
    """
    统一构建 PerceptionResult — 从 BaseResult (+ 可选 EnhancedResult) 一步生成。

    使用方式::

        # 简单路径 (非PDF)
        result = PerceptionResultBuilder.build(base_result, file_path="a.xlsx", file_type="excel")

        # PDF 增强路径
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
        一步构建 PerceptionResult。

        Args:
            base_result: CoreExtractor 的 BaseResult 输出。
            enhanced:    可选的 EnhancedResult (PDF 路径)。
            其余参数:     文件上下文信息，由 dispatcher 传入。
        """
        meta = base_result.metadata if base_result else {}

        # ══════════════════════════════════════════════════════════════
        # 1. Envelope 信封层
        # ══════════════════════════════════════════════════════════════
        if enhanced is not None:
            status_map = {"success": ResultStatus.SUCCESS, "partial": ResultStatus.PARTIAL, "failed": ResultStatus.FAILURE}
            result_status = status_map.get(enhanced.status, ResultStatus.FAILURE)
            confidence = 1.0 if enhanced.status == "success" else (0.5 if enhanced.status == "partial" else 0.0)
            error_detail = ErrorDetail(message="; ".join(enhanced.errors)) if enhanced.errors else None
            p_name = parser_name or "DocMirror"
            p_elapsed = elapsed_ms or enhanced.processing_time
        else:
            result_status = ResultStatus.SUCCESS
            confidence = 1.0
            error_detail = None
            p_name = parser_name
            p_elapsed = elapsed_ms

        timing = TimingInfo(started_at=started_at, parser_name=p_name, elapsed_ms=p_elapsed)

        # ══════════════════════════════════════════════════════════════
        # 2. Content 内容层
        # ══════════════════════════════════════════════════════════════
        content_blocks = [_map_block(b) for b in base_result.all_blocks] if base_result else []

        # 如果有 EnhancedResult 的标准化表格, 覆盖原始表格
        if enhanced is not None:
            std_tables = enhanced.standardized_tables
            if std_tables:
                _overlay_standardized_tables(content_blocks, std_tables)

        # entities: 合并 base KV blocks + enhanced 提取
        entities = {}
        if base_result:
            entities.update(base_result.entities)
        if enhanced is not None:
            entities.update(meta.get("extracted_entities", {}))

        content = DocumentContent(
            text=base_result.full_text if base_result else "",
            text_format="markdown" if (base_result and base_result.full_text.startswith("|")) else "plain",
            blocks=content_blocks,
            entities={k: str(v) for k, v in entities.items()},
            page_count=base_result.page_count if base_result else 0,
        )

        # ══════════════════════════════════════════════════════════════
        # 3. Provenance 溯源层
        # ══════════════════════════════════════════════════════════════
        # PDF 属性
        pdf_props = {}
        target_keys = ("format", "producer", "creator", "creationDate", "modDate",
                       "title", "author", "subject", "keywords", "trapped", "encryption")
        for k in target_keys:
            if k in meta:
                pdf_props[k] = str(meta[k]) if meta[k] is not None else ""

        # 验证结果 (主要针对 PDF 管线)
        validation = None
        if enhanced is not None:
            vr = enhanced.validation_result
            if vr:
                validation = ValidationResult(
                    l2_score=vr.get("total_score"),
                    l2_passed=vr.get("passed"),
                    l2_details=vr.get("details"),
                )
            # L1 修复信息
            if any(k in meta for k in ("l1_anomaly_count", "l1_repaired_count")):
                if validation is None:
                    validation = ValidationResult()
                validation.l1_anomaly_count = meta.get("l1_anomaly_count", 0)
                validation.l1_repaired_count = meta.get("l1_repaired_count", 0)
                validation.l1_reverted_count = meta.get("l1_reverted_count", 0)
                validation.l1_llm_used = meta.get("l1_llm_used", False)
                validation.balance_truncation_repaired = meta.get("balance_truncation_repaired", 0)

        # 防伪信息
        if is_forged is not None:
            if validation is None:
                validation = ValidationResult()
            validation.is_forged = is_forged
            validation.forgery_reasons = forgery_reasons or []

        # Diagnostics
        diagnostics = None
        diag_data = meta.get("_diagnostics", {})
        if diag_data:
            diagnostics = Diagnostics(
                extraction_method=diag_data.get("extraction_method", ""),
                template_id=diag_data.get("template_id", ""),
                template_source=diag_data.get("template_source", ""),
                pages_processed=diag_data.get("pages_processed", 0),
                raw_rows_extracted=diag_data.get("raw_rows_extracted", 0),
                rows_after_cleaning=diag_data.get("rows_after_cleaning", 0),
                rows_final=diag_data.get("rows_final", 0),
                step_timing_ms=diag_data.get("step_timing_ms", {}),
                detected_columns=diag_data.get("detected_columns", []),
                missing_columns=diag_data.get("missing_columns", []),
                supplemented_columns=diag_data.get("supplemented_columns", []),
                failed_rows_sample=diag_data.get("failed_rows_sample", []),
                duplicate_rows_detected=diag_data.get("duplicate_rows_detected", 0),
                llm_usage=diag_data.get("llm_usage"),
            )

        provenance = Provenance(
            source=SourceInfo(
                file_path=file_path,
                file_size=file_size,
                file_type=file_type,
                mime_type=mime_type or None,
                checksum=checksum or None,
            ),
            parser_chain=[ParserStep(parser=p_name, action="parse", elapsed_ms=p_elapsed)] if p_name else [],
            validation=validation,
            diagnostics=diagnostics,
            pdf_properties=pdf_props,
        )

        # ══════════════════════════════════════════════════════════════
        # 4. Domain 领域层
        # ══════════════════════════════════════════════════════════════
        domain = None
        if enhanced is not None:
            cat = enhanced.scene
            if cat == "bank_statement":
                try:
                    from ..entities.domain_models import BankStatementData, DomainData
                    bs = BankStatementData(
                        account_holder=str(entities.get("户名", entities.get("账户名", ""))),
                        account_number=str(entities.get("账号", entities.get("卡号", ""))),
                        bank_name=str(entities.get("bank_name", "")),
                        query_period=str(entities.get("查询期间", "")),
                        currency=str(entities.get("币种", "CNY")) or "CNY",
                    )
                    domain = DomainData(document_type="bank_statement", bank_statement=bs)
                except ImportError:
                    pass

        # ══════════════════════════════════════════════════════════════
        # 5. 组装 PerceptionResult
        # ══════════════════════════════════════════════════════════════
        pr = PerceptionResult(
            status=result_status,
            confidence=confidence,
            timing=timing,
            error=error_detail,
            content=content,
            domain=domain,
            provenance=provenance,
        )

        # 附带 EnhancedResult 引用 (私有, 序列化时排除)
        if enhanced is not None:
            pr._enhanced = enhanced

        return pr
