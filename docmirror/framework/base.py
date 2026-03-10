
"""
多模态ParseContract layer (MultiModal Parsing Contract Layer)

本Moduledefine了多模态Parse系统的“DataContract”与“基准行为”。
It serves as the decoupling point between Dispatcher and parsers, ensuring consistency
in parsing flow and output format across different formats (PDF, Image, Office).

核心组件:
1. ParserStatus: Parsing lifecycle status enum.
2. ParserOutput: Standardized parser output model with backward-compatible API.
3. BaseParser: Abstract base class defining the parse() interface all parsers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
from pathlib import Path

# Import四 layer模型define (对外统一模型)
from docmirror.models.perception_result import (
    ContentBlock,
    ContentBlockType,
    Diagnostics,
    DocumentContent,
    ErrorDetail,
    KeyValueBlock,
    PerceptionResult,
    Provenance,
    ResultStatus,
    SourceInfo,
    TableBlock,
    TextBlock,
    TimingInfo,
    ValidationResult,
    ParserStep,
)
from docmirror.models.domain_models import (
    BankStatementData,
    DomainData,
)

class ParserStatus(str, Enum):
    """
    Parsing status enum marking the quality level of parser results.
    """
    SUCCESS = "success"             # Complete success
    PARTIAL_SUCCESS = "partial_success"     # Partial success (e.g., some tables failed but text exists)
    FAILURE = "failure"             # Core logic failure

class ParserOutput(BaseModel):
    """
    Standard parser output model.
    
    Design goals:
    1. Uniformity: Whether PDF or OCR, the returned data structure must be consistent.
    2. Compatibility: Seamlessly connects to legacy PerceptionResponse API via properties.
    3. Conversion: Provides to_perception_result() to map internal model to the 4-layer PerceptionResult.
    """
    metadata: Dict[str, Any] = Field(default_factory=dict, description="File metadata (author, creation date, page count, etc.)")
    structured_text: str = Field("", description="Reconstructed structured text (typically Markdown)")
    document_structure: List[Dict[str, Any]] = Field(default_factory=list, description="Document structure blocks (Headings, Paragraphs, Tables)")
    key_entities: Dict[str, Any] = Field(default_factory=dict, description="Business-relevant entity extraction (e.g., bank name, account)")
    status: ParserStatus = ParserStatus.SUCCESS
    error: Optional[str] = None
    confidence: float = Field(1.0, description="Overall parsing confidence score (0-1.0)")

    # ── Compatibility properties (for legacy PerceptionResponse callers) ──

    @property
    def success(self) -> bool:
        """Whether parsing is considered successful (includes partial success)."""
        return self.status in (ParserStatus.SUCCESS, ParserStatus.PARTIAL_SUCCESS)

    @property
    def coverage(self) -> float:
        """Alias for confidence, adapting legacy API."""
        return self.confidence

    @property
    def tables(self) -> List[List]:
        """
        Extract raw table data blocks from document_structure.
        Supports new format (headers + rows) and legacy format (data).
        """
        result = []
        for b in self.document_structure:
            if b.get("type") != "table":
                continue
            # New format: headers + rows
            if "headers" in b and "rows" in b:
                result.append([b["headers"]] + b["rows"])
            # Legacy format: data
            elif "data" in b:
                result.append(b["data"])
        return result

    @property
    def raw_response(self) -> Optional[Dict]:
        """Alias for metadata, mapping to legacy interface."""
        return self.metadata

    def to_perception_result(
        self,
        *,
        file_path: str = "",
        file_type: str = "",
        file_size: int = 0,
        parser_name: str = "",
        elapsed_ms: float = 0.0,
        started_at=None,
        mime_type: str = "",
        checksum: str = "",
        doc_info: Optional[Dict[str, str]] = None,
        is_forged: Optional[bool] = None,
        forgery_reasons: Optional[List[str]] = None,
        sanitize: bool = True,
    ) -> "PerceptionResult":
        """
        [Core Mapping Method]
        Converts Parser internal payload to the standardized 4-layer PerceptionResult model.

        Mapping logic:
        1. Envelope: Maps status, timing, and error.
        2. Content: Maps document_structure blocks to ContentBlocks (Table/Text/KV).
        3. Provenance: Maps source file info, PDF properties, and validation status.
        4. Domain: Maps domain-specific models based on category (e.g., bank statement).

        Args:
            file_path: Original file path.
            file_type: Detected file format (pdf, image...).
            file_size: File size (bytes).
            parser_name: Executed parser class name.
            elapsed_ms: Total processing time.
            doc_info: Business metadata from DigitalPDFParser.classify().
            is_forged: (Forgery detection) Whether the file is suspected forged.
            forgery_reasons: (Forgery detection) List of suspected forgery reasons.
        """
        
        # ── 1. Envelope: Status sync ──
        status_map = {
            ParserStatus.SUCCESS: ResultStatus.SUCCESS,
            ParserStatus.PARTIAL_SUCCESS: ResultStatus.PARTIAL,
            ParserStatus.FAILURE: ResultStatus.FAILURE,
        }
        result_status = status_map.get(self.status, ResultStatus.FAILURE)
        error_detail = ErrorDetail(message=self.error) if self.error else None
        timing = TimingInfo(started_at=started_at, parser_name=parser_name, elapsed_ms=elapsed_ms)

        # ── 2. Content: Expand document_structure into ContentBlocks ──
        blocks: list = []
        for b in self.document_structure:
            btype = b.get("type", "text")
            page = b.get("page")
            
            if btype == "table":
                # New format: headers + rows
                if "headers" in b and "rows" in b:
                    headers = b["headers"]
                    rows = b["rows"]
                # Legacy format: data (header = data[0], rows = data[1:])
                elif "data" in b:
                    raw_data = b["data"]
                    headers = raw_data[0] if raw_data else []
                    rows = raw_data[1:] if len(raw_data) > 1 else []
                else:
                    continue

                bbox_raw = b.get("bbox")
                bbox = tuple(bbox_raw) if bbox_raw and len(bbox_raw) == 4 else None
                
                blocks.append(ContentBlock(
                    type=ContentBlockType.TABLE,
                    page=page,
                    table=TableBlock(
                        headers=headers,
                        rows=rows,
                        page=page,
                        bbox=bbox,
                        markdown=b.get("markdown", ""),
                    ),
                ))
            elif btype == "key_value":
                # 新Format: pairs / 旧Format: pairs from entities
                pairs = b.get("pairs", {})
                if not pairs:
                    pairs = b.get("entities", {})
                if pairs:
                    blocks.append(ContentBlock(
                        type=ContentBlockType.KEY_VALUE,
                        page=page,
                        key_value=KeyValueBlock(pairs=pairs),
                    ))
            elif btype == "summary":
                # Legacy: summary → key_value
                pairs = b.get("entities", b.get("pairs", {}))
                if pairs:
                    blocks.append(ContentBlock(
                        type=ContentBlockType.KEY_VALUE,
                        page=page,
                        key_value=KeyValueBlock(pairs=pairs),
                    ))
            else:
                # title / footer / text → TextBlock
                blocks.append(ContentBlock(
                    type=ContentBlockType.TEXT,
                    page=page,
                    text=TextBlock(
                        content=b.get("content", b.get("text", "")),
                        level=b.get("level", 0),
                    ),
                ))

        content = DocumentContent(
            text=self.structured_text,
            text_format="markdown" if self.structured_text.startswith("|") else "plain",
            blocks=blocks,
            entities={k: str(v) for k, v in self.key_entities.items()},
            page_count=self.metadata.get("page_count", 0),
        )

        # ── 3. Provenance: Parse chain tracking & metadata ──
        # Extract key PDF properties
        pdf_props = {}
        target_keys = ("format", "producer", "creator", "creationDate", "modDate",
                      "title", "author", "subject", "keywords", "trapped", "encryption")
        for k in target_keys:
            if k in self.metadata:
                pdf_props[k] = str(self.metadata[k]) if self.metadata[k] is not None else ""

        # Extract validation scores (L1/L2 validation) and forgery detection results
        validation = None
        meta = self.metadata
        if any(k in meta for k in ("l2_score", "l2_passed", "l1_anomaly_count")) or (is_forged is not None):
            validation = ValidationResult(
                l1_anomaly_count=meta.get("l1_anomaly_count", 0),
                l1_repaired_count=meta.get("l1_repaired_count", 0),
                l1_reverted_count=meta.get("l1_reverted_count", 0),
                l1_llm_used=meta.get("l1_llm_used", False),
                l2_score=meta.get("l2_score"),
                l2_passed=meta.get("l2_passed"),
                l2_details=meta.get("l2_details"),
                l2_llm_used=meta.get("l2_llm_used", False),
                balance_truncation_repaired=meta.get("balance_truncation_repaired", 0),
                is_forged=is_forged,
                forgery_reasons=forgery_reasons or [],
            )

        provenance = Provenance(
            source=SourceInfo(
                file_path=file_path,
                file_size=file_size,
                file_type=file_type,
                mime_type=mime_type or None,
                checksum=checksum or None,
            ),
            # Record first hop of parse chain
            parser_chain=[ParserStep(parser=parser_name, action="parse", elapsed_ms=elapsed_ms)]
                if parser_name else [],
            validation=validation,
            diagnostics=self._build_diagnostics(meta),
            pdf_properties=pdf_props,
        )

        # ── 4. Domain: Plugin-based domain data abstraction ──
        domain = None
        cat = (doc_info or {}).get("category", "") or meta.get("_doc_category", "")
        if cat:
            from docmirror.plugins import registry as plugin_registry
            domain = plugin_registry.build_domain_data(
                cat, metadata=meta, entities=self.key_entities,
            )

        pr = PerceptionResult(
            status=result_status,
            confidence=self.confidence,
            timing=timing,
            error=error_detail,
            content=content,
            domain=domain,
            provenance=provenance,
        )

        if sanitize:
            pr.sanitize()

        return pr

    @staticmethod
    def _build_diagnostics(meta: Dict[str, Any]):
        """Extract debug diagnostics from metadata."""
        diag_data = meta.get("_diagnostics", {})
        if not diag_data:
            return None
        return Diagnostics(
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

class BaseParser(ABC):
    """
    Abstract base class for document parsers.

    New interface: ``perceive()`` → PerceptionResult (recommended)
    Legacy interface: ``parse()`` → ParserOutput (deprecated, kept for compatibility)
    """

    async def to_base_result(self, file_path: Path, **kwargs):
        """
        Extract file to BaseResult. Subclasses should implement this method.
        Not implemented by default; perceive() will fallback to parse().
        """
        raise NotImplementedError

    async def perceive(self, file_path: Path, **context) -> "PerceptionResult":
        """
        New unified interface: file → PerceptionResult (single step).

        Default impl: to_base_result() → Builder → PerceptionResult.
        If subclass doesn't implement to_base_result(), falls back to parse() → to_perception_result().
        """
        try:
            base_result = await self.to_base_result(file_path)
            from docmirror.models.builder import PerceptionResultBuilder
            return PerceptionResultBuilder.build(base_result, **context)
        except NotImplementedError:
            # fallback 到旧Interface
            result = await self.parse(file_path)
            return result.to_perception_result(**context)

    async def parse(self, file_path: Path, **kwargs) -> ParserOutput:
        """
        [DEPRECATED] Implement to_base_result() instead.
        This method is kept only for backward compatibility.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement parse(). Use perceive() instead."
        )

