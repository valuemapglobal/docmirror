"""
PerceptionResult — MultiModal 统一输出模型

4 层架构::

    ┌─────────────────────────────────────────────────────┐
    │  Envelope: status / confidence / timing / error     │
    │  Content:  text + blocks (table/text/kv)            │
    │  Domain:   BankStatementData / InvoiceData / ...    │
    │  Provenance: source / parser_chain / validation     │
    └─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Envelope 信封层
# ═══════════════════════════════════════════════════════════════════════════

class ResultStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class ErrorDetail(BaseModel):
    """结构化错误信息"""
    code: str = "unknown"              # "encrypted_pdf" | "parse_timeout" | ...
    message: str = ""
    recoverable: bool = False


class TimingInfo(BaseModel):
    """解析耗时信息"""
    started_at: Optional[datetime] = None
    elapsed_ms: float = 0.0
    parser_name: str = ""              # "CCBParser" | "DigitalPDFParser"


# ═══════════════════════════════════════════════════════════════════════════
# Content 内容层
# ═══════════════════════════════════════════════════════════════════════════

class ContentBlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"
    KEY_VALUE = "key_value"
    IMAGE = "image"


class TableBlock(BaseModel):
    """结构化表格"""
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    page: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    markdown: str = ""


class TextBlock(BaseModel):
    """文本段落"""
    content: str = ""
    level: int = 0                     # heading level (0=body, 1=h1 ...)


class KeyValueBlock(BaseModel):
    """键值对 (文档头信息等)"""
    pairs: Dict[str, str] = Field(default_factory=dict)


class ContentBlock(BaseModel):
    """
    通用内容块 — 文档由有序块序列组成。

    根据 ``type`` 字段, 对应的子对象 (table / text / key_value) 被填充。
    """
    type: ContentBlockType
    page: Optional[int] = None
    table: Optional[TableBlock] = None
    text: Optional[TextBlock] = None
    key_value: Optional[KeyValueBlock] = None


class DocumentContent(BaseModel):
    """文档提取的通用内容"""
    text: str = ""                     # 全文 Markdown / Plain
    text_format: Literal["markdown", "plain"] = "plain"
    blocks: List[ContentBlock] = Field(default_factory=list)
    entities: Dict[str, str] = Field(default_factory=dict)
    page_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Provenance 溯源层
# ═══════════════════════════════════════════════════════════════════════════

class SourceInfo(BaseModel):
    """文件来源"""
    file_path: str = ""
    file_size: int = 0                 # bytes
    file_type: str = ""                # "pdf" | "image" | "excel"
    mime_type: Optional[str] = None
    checksum: Optional[str] = None     # SHA256

    def sanitize(self) -> "SourceInfo":
        """
        脱敏处理: 隐藏绝对路径，仅保留文件名。
        """
        import os
        if self.file_path:
            self.file_path = os.path.basename(self.file_path)
        return self


class ParserStep(BaseModel):
    """解析链中的一步"""
    parser: str = ""                   # "CCBParser"
    action: str = ""                   # "classify" | "extract" | "validate"
    elapsed_ms: float = 0.0


class ValidationResult(BaseModel):
    """解析质量验证"""
    l1_anomaly_count: int = 0
    l1_repaired_count: int = 0
    l1_reverted_count: int = 0
    l1_llm_used: bool = False
    l2_score: Optional[float] = None
    l2_passed: Optional[bool] = None
    l2_details: Optional[Dict[str, float]] = None
    l2_llm_used: bool = False
    balance_truncation_repaired: int = 0
    is_forged: Optional[bool] = None
    forgery_reasons: List[str] = Field(default_factory=list)


class Diagnostics(BaseModel):
    """解析调试诊断信息"""
    # 提取路径诊断
    extraction_method: str = ""                    # coordinate / pdfplumber / ocr_fallback
    template_id: str = ""                          # 模板 ID
    template_source: str = ""                      # memory / yaml / llm_inferred
    pages_processed: int = 0
    raw_rows_extracted: int = 0                    # Step 2 原始行数
    rows_after_cleaning: int = 0                   # Step 4 清洗后行数
    rows_final: int = 0                            # 最终标准化行数

    # 各步骤耗时 (ms)
    step_timing_ms: Dict[str, float] = Field(default_factory=dict)

    # 列映射诊断
    detected_columns: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    supplemented_columns: List[str] = Field(default_factory=list)

    # 问题行样本 (最多 10 行)
    failed_rows_sample: List[Dict[str, Any]] = Field(default_factory=list)

    # 重复行检测
    duplicate_rows_detected: int = 0

    # LLM 调用成本
    llm_usage: Optional[Dict[str, Any]] = None


class Provenance(BaseModel):
    """解析溯源"""
    source: SourceInfo = Field(default_factory=SourceInfo)
    parser_chain: List[ParserStep] = Field(default_factory=list)
    validation: Optional[ValidationResult] = None
    diagnostics: Optional[Diagnostics] = None
    pdf_properties: Dict[str, str] = Field(default_factory=dict)

    def sanitize(self) -> "Provenance":
        """
        脱敏处理: 递归脱敏 SourceInfo。
        """
        self.source.sanitize()
        return self


# ═══════════════════════════════════════════════════════════════════════════
# PerceptionResult 顶层
# ═══════════════════════════════════════════════════════════════════════════

class PerceptionResult(BaseModel):
    """
    MultiModal 统一输出模型。

    所有 Parser 最终返回此类型, 替代原有 ``ParserOutput``。
    通过 ``@property`` 提供向后兼容字段。
    """

    # ── Envelope ──
    status: ResultStatus = ResultStatus.SUCCESS
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    timing: TimingInfo = Field(default_factory=TimingInfo)
    error: Optional[ErrorDetail] = None

    # ── Content ──
    content: DocumentContent = Field(default_factory=DocumentContent)

    # ── Domain (可选, 按文档类型填充) ──
    domain: Optional[Any] = None       # DomainData, 延迟导入避免循环

    # ── Provenance ──
    provenance: Provenance = Field(default_factory=Provenance)

    # ── 内部引用 (序列化时排除) ──
    _enhanced: Optional[Any] = None     # EnhancedResult 引用，供深度调用方访问

    def sanitize(self) -> "PerceptionResult":
        """
        显式脱敏: 移除内部路径与解析链细节, 供外部 API 安全返回。
        调用此方法后, 对象状态将被修改。
        """
        self.provenance.sanitize()
        return self

    # ═══════════════════════════════════════════════════════════════════════
    # 向后兼容属性 — 使旧代码无需修改即可使用
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def success(self) -> bool:
        return self.status in (ResultStatus.SUCCESS, ResultStatus.PARTIAL)

    @property
    def coverage(self) -> float:
        return self.confidence

    @property
    def structured_text(self) -> str:
        return self.content.text

    @property
    def tables(self) -> List[List[List[str]]]:
        """提取所有表格的 rows (含 headers 作为首行)"""
        result = []
        for b in self.content.blocks:
            if b.type == ContentBlockType.TABLE and b.table:
                data = [b.table.headers] + b.table.rows if b.table.headers else b.table.rows
                result.append(data)
        return result

    @property
    def key_entities(self) -> Dict[str, str]:
        return self.content.entities

    @property
    def metadata(self) -> Dict[str, Any]:
        """兼容旧调用方 — 合并溯源信息 + 领域数据为 flat dict"""
        result: Dict[str, Any] = {}
        result.update(self.provenance.pdf_properties)
        result["page_count"] = self.content.page_count
        if self.provenance.validation:
            v = self.provenance.validation
            result["l2_score"] = v.l2_score
            result["l2_passed"] = v.l2_passed
            result["l1_anomaly_count"] = v.l1_anomaly_count
            result["l1_llm_used"] = v.l1_llm_used
            result["l2_llm_used"] = v.l2_llm_used
        if self.domain and hasattr(self.domain, "bank_statement") and self.domain.bank_statement:
            bs = self.domain.bank_statement
            result["账户持有人"] = bs.account_holder
            result["账号"] = bs.account_number
            result["查询期间"] = bs.query_period
        return result

    @property
    def document_structure(self) -> List[Dict[str, Any]]:
        """兼容旧调用方 — 返回 block list as dicts"""
        result = []
        for b in self.content.blocks:
            d: Dict[str, Any] = {"type": b.type.value, "page": b.page}
            if b.type == ContentBlockType.TABLE and b.table:
                data = [b.table.headers] + b.table.rows if b.table.headers else b.table.rows
                d["data"] = data
                d["markdown"] = b.table.markdown
                d["bbox"] = b.table.bbox
            elif b.type == ContentBlockType.TEXT and b.text:
                d["content"] = b.text.content
                d["level"] = b.text.level
            elif b.type == ContentBlockType.KEY_VALUE and b.key_value:
                d["pairs"] = b.key_value.pairs
            result.append(d)
        return result

    @property
    def raw_response(self) -> Optional[Dict[str, Any]]:
        return self.metadata

    def to_api_dict(self, *, output_file: str = "") -> Dict[str, Any]:
        """
        扁平化为 API 友好的 dict — 替代 ParseResponse / ParseV2Response。

        前端消费的唯一输出格式。
        """
        enhanced = getattr(self, '_enhanced', None)
        meta = self.metadata

        # identity: 通过 Domain 注册表动态解析 (替代硬编码银行流水字段)
        from docmirror.configs.domain_registry import resolve_identity
        entities = dict(self.content.entities)
        domain = enhanced.scene if enhanced else (
            self.domain.document_type if self.domain else "unknown"
        )
        identity = resolve_identity(domain, entities)
        identity["page_count"] = self.content.page_count

        # trust: 从 provenance.validation 构建
        trust: Dict[str, Any] = {}
        if self.provenance.validation:
            v = self.provenance.validation
            trust["validation_score"] = v.l2_score
            trust["validation_passed"] = v.l2_passed
            trust["validation_details"] = v.l2_details or {}
            trust["is_forged"] = v.is_forged
            trust["forgery_reasons"] = v.forgery_reasons

        # diagnostics
        diagnostics: Dict[str, Any] = {
            "parser": self.timing.parser_name or "DocMirror",
            "elapsed_ms": round(self.timing.elapsed_ms, 1),
            "page_count": self.content.page_count,
            "block_count": len(self.content.blocks),
            "table_count": sum(1 for b in self.content.blocks if b.type == ContentBlockType.TABLE),
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
            "scene": enhanced.scene if enhanced else "unknown",
            "blocks": self.document_structure,
            "trust": trust,
            "diagnostics": diagnostics,
            "output_file": output_file,
        }

        # 增强信息 (仅 PDF 路径附带)
        if enhanced is not None:
            result["pre_analysis"] = meta.get("pre_analysis", {})
            result["mutations"] = enhanced.mutation_count

        return result

