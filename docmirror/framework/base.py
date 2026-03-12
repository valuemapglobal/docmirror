"""
MultiModal Parsing Contract Layer
=================================

This module defines the "Data Contract" and baseline behavior of the multi-modal
parsing system. It serves as the decoupling point between the Dispatcher and
individual parsers (adapters), ensuring consistency in the parsing flow and
output format across different document formats (PDF, Image, Office).

Core Components:
1. ParserStatus: Parsing lifecycle status enumeration.
2. ParserOutput: Standardized internal parser output model (backward-compatible).
3. BaseParser: Abstract base class defining the unified interface all parsers
   must implement.
"""
from __future__ import annotations

from abc import ABC
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from pathlib import Path

# Import 4-layer schema definitions (the unified external models)
from docmirror.models.perception_result import ContentBlock, ContentBlockType, DocumentContent, ErrorDetail, KeyValueBlock, PerceptionResult, Provenance, ResultStatus, SourceInfo, TableBlock, TextBlock, TimingInfo, ValidationResult, ParserStep

class ParserStatus(str, Enum):
    """
    Parsing status enumeration indicating the quality level of parser results.
    """
    SUCCESS = "success"             # Complete success
    PARTIAL_SUCCESS = "partial_success"  # Partial success (e.g., text exists but some tables failed)
    FAILURE = "failure"             # Core logic failure

class ParserOutput(BaseModel):
    """
    Standard internal parser output model.
    
    Design goals:
    1. Uniformity: Ensures returned data structure consistency across PDF, image, etc.
    2. Compatibility: Seamlessly connects to legacy `PerceptionResponse` API via properties.
    3. Conversion: Provides `to_perception_result()` to translate this internal
       payload into the standardized 4-layer `PerceptionResult` model.
    """
    metadata: Dict[str, Any] = Field(default_factory=dict, description="File metadata (author, creation date, page count, etc.)")
    structured_text: str = Field("", description="Reconstructed structured text (typically Markdown)")
    document_structure: List[Dict[str, Any]] = Field(default_factory=list, description="Document structure blocks (Headings, Paragraphs, Tables)")
    key_entities: Dict[str, Any] = Field(default_factory=dict, description="Business-relevant entity extraction (e.g., bank name, account)")
    status: ParserStatus = ParserStatus.SUCCESS
    error: Optional[str] = None
    confidence: float = Field(1.0, description="Overall parsing confidence score (0.0-1.0)")

    # ── Compatibility properties (for callers expecting the legacy PerceptionResponse API) ──

    @property
    def success(self) -> bool:
        """Whether parsing is considered successful (includes partial success)."""
        return self.status in (ParserStatus.SUCCESS, ParserStatus.PARTIAL_SUCCESS)

    @property
    def coverage(self) -> float:
        """Alias for confidence, adapting the legacy API."""
        return self.confidence

    @property
    def tables(self) -> List[List]:
        """
        Extract raw table data blocks from document_structure.
        Supports both the new format (`headers` + `rows`) and legacy format (`data`).
        """
        result = []
        for b in self.document_structure:
            if b.get("type") != "table":
                continue
            # New format: explicit headers and rows
            if "headers" in b and "rows" in b:
                result.append([b["headers"]] + b["rows"])
            # Legacy format: raw data array
            elif "data" in b:
                result.append(b["data"])
        return result

    @property
    def raw_response(self) -> Optional[Dict]:
        """Alias for metadata, mapping to the legacy interface."""
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
        Converts the Parser internal payload to the standardized 4-layer
        PerceptionResult model.

        Mapping Logic:
        1. Envelope: Maps execution status, timing, and error details.
        2. Content: Maps `document_structure` blocks into strongly-typed ContentBlocks (Table/Text/KV).
        3. Provenance: Maps source file info, PDF properties, and validation status.
        4. Domain: Maps domain-specific models based on category (e.g., BankStatementData).

        Args:
            file_path: Original file path.
            file_type: Detected file format (pdf, image...).
            file_size: File size in bytes.
            parser_name: Executed parser class name.
            elapsed_ms: Total processing time in milliseconds.
            doc_info: Business metadata classified by Adapters.
            is_forged: (Forgery detection) Flag indicating suspected forgery.
            forgery_reasons: (Forgery detection) List of reasons supporting the forgery suspicion.
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
                # Legacy format: data array (header = data[0], rows = data[1:])
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
                # New format: `pairs` / Legacy format: pairs from `entities`
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
                # Legacy: summary block mapped to key_value
                pairs = b.get("entities", b.get("pairs", {}))
                if pairs:
                    blocks.append(ContentBlock(
                        type=ContentBlockType.KEY_VALUE,
                        page=page,
                        key_value=KeyValueBlock(pairs=pairs),
                    ))
            elif btype == "title":
                # Title blocks → HEADING type
                blocks.append(ContentBlock(
                    type=ContentBlockType.HEADING,
                    page=page,
                    text=TextBlock(
                        content=b.get("content", b.get("text", "")),
                        level=b.get("level", 1) or 1,
                    ),
                ))
            elif btype == "image":
                # Image blocks → IMAGE type with caption
                blocks.append(ContentBlock(
                    type=ContentBlockType.IMAGE,
                    page=page,
                    text=TextBlock(
                        content=b.get("caption", ""),
                        level=0,
                    ),
                ))
            else:
                # Default text categories: footer / generic text → TextBlock
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

        # ── 3. Provenance: Parse chain tracking & metadata extraction ──
        # Extract key Document properties
        document_properties = {}
        target_keys = ("format", "producer", "creator", "creationDate", "modDate",
                      "title", "author", "subject", "keywords", "trapped", "encryption")
        for k in target_keys:
            if k in self.metadata:
                document_properties[k] = str(self.metadata[k]) if self.metadata[k] is not None else ""

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
            # Record the first hop of the parse chain
            parser_chain=[ParserStep(parser=parser_name, action="parse", elapsed_ms=elapsed_ms)]
                if parser_name else [],
            validation=validation,
            diagnostics=self._build_diagnostics(meta),
            document_properties=document_properties,
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
        from docmirror.models.construction._shared import build_diagnostics
        return build_diagnostics(meta)

class BaseParser(ABC):
    """
    Abstract base class for all document parsers.

    New Unified Interface: ``perceive()`` → PerceptionResult (highly recommended)
    Legacy Interface: ``parse()`` → ParserOutput (deprecated, retained for compatibility)
    """

    async def to_base_result(self, file_path: Path, **kwargs):
        """
        Extract the file into a BaseResult. 
        Subclasses should ideally implement this method. It is not implemented 
        by default; `perceive()` will fallback to `parse()` if this is missing.
        """
        raise NotImplementedError

    async def perceive(self, file_path: Path, **context) -> "PerceptionResult":
        """
        New unified interface mapping a file directly to a standardized `PerceptionResult` 
        in a single step.

        Execution pipeline:
        `to_base_result()` → Builder Pipeline → `PerceptionResult`.
        If the subclass does not implement `to_base_result()`, it falls back to 
        `parse()` → `to_perception_result()`.
        """
        try:
            base_result = await self.to_base_result(file_path)
            from docmirror.models.builder import PerceptionResultBuilder
            return PerceptionResultBuilder.build(base_result, **context)
        except NotImplementedError:
            # Fallback to the legacy interface
            result = await self.parse(file_path)
            return result.to_perception_result(**context)

    async def parse(self, file_path: Path, **kwargs) -> ParserOutput:
        """
        [DEPRECATED] Interface.
        Implement `to_base_result()` instead. This method is maintained solely for
        backward compatibility with older integrations.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement parse(). Use perceive() instead."
        )
