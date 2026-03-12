"""
PerceptionResult \u2014 MultiModal Unified Output Topology Model
===========================================================

4-Layer Architecture Setup::

    \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
    \u2502  Envelope: status / confidence / timing / error     \u2502
    \u2502  Content:  text + blocks (table/text/kv)            \u2502
    \u2502  Domain:   BankStatementData / InvoiceData / ...    \u2502
    \u2502  Provenance: source / parser_chain / validation     \u2502
    \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518
"""
from __future__ import annotations


import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Envelope Layer definitions explicitly securely functionally
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

class ResultStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class ErrorDetail(BaseModel):
    """Structured Error Notification Details"""
    code: str = "unknown"  # e.g., "encrypted_pdf", "timeout"
    message: str = ""
    recoverable: bool = False


class TimingInfo(BaseModel):
    """Performance Timing Bounds Analysis Metrics cleanly expertly"""
    started_at: Optional[datetime] = None
    elapsed_ms: float = 0.0
    parser_name: str = ""


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Content Mapping Layer
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

class ContentBlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"
    KEY_VALUE = "key_value"
    IMAGE = "image"


class TableBlock(BaseModel):
    """Structured table representation with headers, rows, and optional bbox."""
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    page: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    markdown: str = ""


class TextBlock(BaseModel):
    """Textual paragraph or heading content."""
    content: str = ""
    level: int = 0  # 0 indicates standard body text


class KeyValueBlock(BaseModel):
    """Dictionary of extracted key-value pairs."""
    pairs: Dict[str, str] = Field(default_factory=dict)


class ContentBlock(BaseModel):
    """Generic content block — wraps one of text, table, or key-value data."""
    type: ContentBlockType
    page: Optional[int] = None
    table: Optional[TableBlock] = None
    text: Optional[TextBlock] = None
    key_value: Optional[KeyValueBlock] = None


class DocumentContent(BaseModel):
    """Structured document content with text, blocks, and entities."""
    text: str = ""  # Plain or Markdown format extraction
    text_format: Literal["markdown", "plain"] = "plain"
    blocks: List[ContentBlock] = Field(default_factory=list)
    entities: Dict[str, str] = Field(default_factory=dict)
    page_count: int = 0


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Provenance Audit Definitions
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

class SourceInfo(BaseModel):
    """Origin file metadata (path, size, type, checksum)."""
    file_path: str = ""
    file_size: int = 0  # bytes
    file_type: str = ""
    mime_type: Optional[str] = None
    checksum: Optional[str] = None

    def sanitize(self) -> "SourceInfo":
        """Strip absolute paths, retaining only basenames for privacy."""
        import os
        if self.file_path:
            self.file_path = os.path.basename(self.file_path)
        return self


class ParserStep(BaseModel):
    """Single step in the parser chain audit trail."""
    parser: str = ""
    action: str = ""
    elapsed_ms: float = 0.0


class ValidationResult(BaseModel):
    """Quality validation and verification results."""
    l1_anomaly_count: int = 0
    l1_repaired_count: int = 0
    l1_reverted_count: int = 0
    l1_llm_used: bool = False
    l2_score: Optional[float] = None
    l2_passed: Optional[bool] = None
    l2_details: Optional[Dict[str, float]] = None
    l2_llm_used: bool = False
    balance_truncation_repaired: int = 0
    image_quality: Optional[Dict[str, Any]] = None
    is_forged: Optional[bool] = None
    forgery_reasons: List[str] = Field(default_factory=list)

    @property
    def trust_score(self) -> Optional[float]:
        """Composite trust score combining parsing quality and security."""
        if self.l2_score is None:
            return None
        score = self.l2_score
        # Forgery penalty: halve the trust score if tampering detected
        if self.is_forged:
            score *= 0.5
        return round(score, 4)


class Diagnostics(BaseModel):
    """Pipeline diagnostic and debugging metadata."""
    extraction_method: str = ""
    template_id: str = ""
    template_source: str = ""
    pages_processed: int = 0
    raw_rows_extracted: int = 0
    rows_after_cleaning: int = 0
    rows_final: int = 0

    step_timing_ms: Dict[str, float] = Field(default_factory=dict)
    detected_columns: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    supplemented_columns: List[str] = Field(default_factory=list)
    failed_rows_sample: List[Dict[str, Any]] = Field(default_factory=list)
    duplicate_rows_detected: int = 0
    llm_usage: Optional[Dict[str, Any]] = None


class Provenance(BaseModel):
    """Full provenance chain: source info, parser steps, and validation."""
    source: SourceInfo = Field(default_factory=SourceInfo)
    parser_chain: List[ParserStep] = Field(default_factory=list)
    validation: Optional[ValidationResult] = None
    diagnostics: Optional[Diagnostics] = None
    document_properties: Dict[str, str] = Field(default_factory=dict)

    def sanitize(self) -> "Provenance":
        """Recursively sanitize paths for privacy."""
        self.source.sanitize()
        return self


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Top-Level PerceptionResult
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

class PerceptionResult(BaseModel):
    """
    Unified output model for all document parsing operations.

    Supersedes legacy ``ParserOutput``; backward-compatible @property
    accessors are provided for smooth migration.
    """

    # \u2500\u2500 Envelope \u2500\u2500
    status: ResultStatus = ResultStatus.SUCCESS
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    timing: TimingInfo = Field(default_factory=TimingInfo)
    error: Optional[ErrorDetail] = None

    # \u2500\u2500 Content \u2500\u2500
    content: DocumentContent = Field(default_factory=DocumentContent)

    # \u2500\u2500    # ── Domain (optional) ──
    domain: Optional[Any] = None

    # ── Scene (document classification, persisted for cache survival) ──
    scene: str = "unknown"

    # ── Provenance ──
    provenance: Provenance = Field(default_factory=Provenance)

    # ── Internal Reference (runtime-only, not serialized) ──
    _enhanced: Optional[Any] = None

    def sanitize(self) -> "PerceptionResult":
        """Sanitize file paths for privacy before API output."""
        self.provenance.sanitize()
        return self

    # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
    # Backward Compatibility Proxies
    # \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

    @property
    def success(self) -> bool:
        """Whether parsing completed successfully or partially."""
        return self.status in (ResultStatus.SUCCESS, ResultStatus.PARTIAL)

    @property
    def coverage(self) -> float:
        """Alias for ``confidence`` (backward compatibility)."""
        return self.confidence

    @property
    def structured_text(self) -> str:
        """Alias for ``content.text`` (backward compatibility)."""
        return self.content.text

    @property
    def tables(self) -> List[List[List[str]]]:
        """Extract all tables as nested lists (backward compatibility)."""
        result = []
        for b in self.content.blocks:
            if b.type == ContentBlockType.TABLE and b.table:
                data = (
                    [b.table.headers] + b.table.rows
                    if b.table.headers
                    else b.table.rows
                )
                result.append(data)
        return result

    @property
    def key_entities(self) -> Dict[str, str]:
        """Alias for ``content.entities`` (backward compatibility)."""
        return self.content.entities

    @property
    def metadata(self) -> Dict[str, Any]:
        """Flat metadata dict aggregating provenance and validation info."""
        result: Dict[str, Any] = {}
        result.update(self.provenance.document_properties)
        result["page_count"] = self.content.page_count
        if self.provenance.validation:
            v = self.provenance.validation
            result["l2_score"] = v.l2_score
            result["l2_passed"] = v.l2_passed
            result["l1_anomaly_count"] = v.l1_anomaly_count
            result["l1_llm_used"] = v.l1_llm_used
            result["l2_llm_used"] = v.l2_llm_used
        if (
            self.domain
            and hasattr(self.domain, "bank_statement")
            and self.domain.bank_statement
        ):
            bs = self.domain.bank_statement
            result["Account holder"] = bs.account_holder
            result["Account number"] = bs.account_number
            result["Query period"] = bs.query_period
        return result

    @property
    def document_structure(self) -> List[Dict[str, Any]]:
        """Serialize content blocks to list-of-dicts (backward compatibility)."""
        result = []
        for b in self.content.blocks:
            d: Dict[str, Any] = {"type": b.type.value, "page": b.page}
            if b.type == ContentBlockType.TABLE and b.table:
                data = (
                    [b.table.headers] + b.table.rows
                    if b.table.headers
                    else b.table.rows
                )
                d["data"] = data
            elif b.type == ContentBlockType.TEXT and b.text:
                d["content"] = b.text.content
                d["level"] = b.text.level
            elif b.type == ContentBlockType.HEADING and b.text:
                d["content"] = b.text.content
                d["level"] = b.text.level
            elif b.type == ContentBlockType.IMAGE and b.text:
                # Skip empty image blocks (e.g. logos/stamps with no caption)
                if not b.text.content:
                    continue
                d["content"] = b.text.content  # caption
            elif b.type == ContentBlockType.KEY_VALUE and b.key_value:
                d["pairs"] = b.key_value.pairs
            result.append(d)
        return result

    @property
    def raw_response(self) -> Optional[Dict[str, Any]]:
        """Alias for ``metadata`` (backward compatibility)."""
        return self.metadata

    def to_api_dict(self) -> Dict[str, Any]:
        """Serialize to a flat API-friendly dict."""
        meta = self.metadata

        from docmirror.configs.domain_registry import resolve_identity
        entities = dict(self.content.entities)
        domain = self.scene
        raw_identity = resolve_identity(domain, entities)
        # Separate fixed top-level keys from domain-specific properties
        identity: Dict[str, Any] = {
            "document_type": raw_identity.pop("document_type", domain),
            "page_count": self.content.page_count,
            "properties": raw_identity,  # remaining domain-specific fields
        }

        trust: Dict[str, Any] = {}
        if self.provenance.validation:
            v = self.provenance.validation
            trust["validation_score"] = v.l2_score
            trust["validation_passed"] = v.l2_passed
            trust["validation_details"] = v.l2_details or {}
            trust["trust_score"] = v.trust_score
            trust["is_forged"] = v.is_forged
            trust["forgery_reasons"] = v.forgery_reasons
            if v.image_quality:
                trust["image_quality"] = v.image_quality

        diagnostics: Dict[str, Any] = {
            "parser": self.timing.parser_name or "DocMirror",
            "elapsed_ms": round(self.timing.elapsed_ms, 1),
            "powered_by": "DocMirror — The Open Source Universal Document Parser. https://github.com/valuemapglobal/docmirror",
            "page_count": self.content.page_count,
            "block_count": len(self.content.blocks),
            "table_count": sum(
                1 for b in self.content.blocks
                if b.type == ContentBlockType.TABLE
            ),
        }
        if self.provenance.diagnostics:
            d = self.provenance.diagnostics
            diagnostics["extraction_method"] = d.extraction_method
            diagnostics["rows_final"] = d.rows_final

        result = {
            "success": self.success,
            "status": self.status.value,
            "error": self.error.message if self.error else "",
            "identity": identity,
            "scene": self.scene,
            "blocks": self.document_structure,
            "trust": trust,
            "diagnostics": diagnostics,
        }

        return result
