
"""
多模态解析契约层 (MultiModal Parsing Contract Layer)

本模块定义了多模态解析系统的“数据契约”与“基准行为”。
它是 Dispatcher 与具体 Parser 之间的解耦点，确保了不同格式 (PDF, Image, Office) 
在解析流程和输出格式上的高度一致性。

核心组件:
1. ParserStatus: 解析生命周期的状态枚举。
2. ParserOutput: 解析器内部的标准化输出模型，具备向下兼容旧版 API 的能力。
3. BaseParser: 抽象基类，定义了所有解析器必须遵守的 parse() 接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
from pathlib import Path

# 导入四层模型定义 (对外统一模型)
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
    解析状态枚举。用于标识 Parser 阶段性的解析质量。
    """
    SUCCESS = "success"             # 完全成功
    PARTIAL_SUCCESS = "partial_success"     # 部分成功（如部分表格解析失败，但文本存在）
    FAILURE = "failure"             # 核心逻辑失败

class ParserOutput(BaseModel):
    """
    Parser 内部的标准输出模型。
    
    设计目标：
    1. 统一性：无论 PDF 还是 OCR，返回的数据结构必须一致。
    2. 兼容性：通过 property 完美衔接旧版的 PerceptionResponse 接口。
    3. 转换力：提供 self.to_perception_result() 一键将内部模型映射到对外的 PerceptionResult 四层模型。
    """
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文件元数据 (如作者、创建日期、页数等)")
    structured_text: str = Field("", description="重建的层级化文本 (通常为 Markdown 格式)")
    document_structure: List[Dict[str, Any]] = Field(default_factory=list, description="文档结构块列表 (Headings, Paragraphs, Tables)")
    key_entities: Dict[str, Any] = Field(default_factory=dict, description="业务强相关的实体提取 (如银行名、账户名等)")
    status: ParserStatus = ParserStatus.SUCCESS
    error: Optional[str] = None
    confidence: float = Field(1.0, description="解析整体置信度评分 (0-1.0)")

    # ── 兼容性属性区域 (针对旧版 PerceptionResponse 调用方) ──

    @property
    def success(self) -> bool:
        """解析是否算作成功 (包含部分成功)。"""
        return self.status in (ParserStatus.SUCCESS, ParserStatus.PARTIAL_SUCCESS)

    @property
    def coverage(self) -> float:
        """confidence 的别名，适配旧版 API。"""
        return self.confidence

    @property
    def tables(self) -> List[List]:
        """
        从 document_structure 中快捷提取表格原始数据块。
        支持新格式 (headers + rows) 和旧格式 (data)。
        """
        result = []
        for b in self.document_structure:
            if b.get("type") != "table":
                continue
            # 新格式: headers + rows
            if "headers" in b and "rows" in b:
                result.append([b["headers"]] + b["rows"])
            # 旧格式: data
            elif "data" in b:
                result.append(b["data"])
        return result

    @property
    def raw_response(self) -> Optional[Dict]:
        """metadata 的别名，映射旧版接口。"""
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
        [核心映射方法]
        将 Parser 的内部工作负载转换为标准化的 PerceptionResult 4 层模型。

        映射逻辑:
        1. Envelope: 映射状态、耗时与错误。
        2. Content: 将 document_structure 块级映射为 ContentBlocks (Table/Text/KV)。
        3. Provenance: 映射文件源信息及 PDF 特定属性、校验状态。
        4. Domain: 根据文品种识别 (category) 映射领域专属模型 (如银行流水)。

        Args:
            file_path: 原始文件路径。
            file_type: 识别出的文件格式 (pdf, image...)。
            file_size: 文件大小 (bytes)。
            parser_name: 最终执行的 Parser 类名。
            elapsed_ms: 总处理耗时。
            doc_info: 由 DigitalPDFParser.classify() 推出的业务元信息。
            is_forged: (篡改鉴定) 该文件是否疑似伪造篡改。
            forgery_reasons: (篡改鉴定) 疑似伪造的原因依据列表。
        """
        
        # ── 1. Envelope (外壳层): 状态同步 ──
        status_map = {
            ParserStatus.SUCCESS: ResultStatus.SUCCESS,
            ParserStatus.PARTIAL_SUCCESS: ResultStatus.PARTIAL,
            ParserStatus.FAILURE: ResultStatus.FAILURE,
        }
        result_status = status_map.get(self.status, ResultStatus.FAILURE)
        error_detail = ErrorDetail(message=self.error) if self.error else None
        timing = TimingInfo(started_at=started_at, parser_name=parser_name, elapsed_ms=elapsed_ms)

        # ── 2. Content (内容层): document_structure 展开为 ContentBlocks ──
        blocks: list = []
        for b in self.document_structure:
            btype = b.get("type", "text")
            page = b.get("page")
            
            if btype == "table":
                # 新格式: headers + rows
                if "headers" in b and "rows" in b:
                    headers = b["headers"]
                    rows = b["rows"]
                # 旧格式: data (header = data[0], rows = data[1:])
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
                # 新格式: pairs / 旧格式: pairs from entities
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

        # ── 3. Provenance (出处层): 解析链追溯与元数据 ──
        # 提取关键 PDF 属性
        pdf_props = {}
        target_keys = ("format", "producer", "creator", "creationDate", "modDate",
                      "title", "author", "subject", "keywords", "trapped", "encryption")
        for k in target_keys:
            if k in self.metadata:
                pdf_props[k] = str(self.metadata[k]) if self.metadata[k] is not None else ""

        # 提取校验过程评分 (主要针对银行流水 L1/L2 校验) 和防伪检测
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
            # 记录解析链路第一跳
            parser_chain=[ParserStep(parser=parser_name, action="parse", elapsed_ms=elapsed_ms)]
                if parser_name else [],
            validation=validation,
            diagnostics=self._build_diagnostics(meta),
            pdf_properties=pdf_props,
        )

        # ── 4. Domain (领域层): 场景相关的数据抽象 ──
        domain = None
        cat = (doc_info or {}).get("category", "") or meta.get("_doc_category", "")
        if cat == "bank_statement":
            bs = BankStatementData(
                account_holder=str(meta.get("账户持有人", "")),
                account_number=str(meta.get("账号", "")),
                bank_name=str(self.key_entities.get("bank_name", "")),
                query_period=str(meta.get("查询期间", "")),
                currency=str(meta.get("币种", "CNY")) or "CNY",
            )
            domain = DomainData(document_type="bank_statement", bank_statement=bs)

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
        """从 metadata 中提取调试诊断信息。"""
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
    文档解析器的抽象基类。

    新接口: ``perceive()`` → PerceptionResult (推荐)
    旧接口: ``parse()`` → ParserOutput (deprecated, 保留兼容)
    """

    async def to_base_result(self, file_path: Path, **kwargs):
        """
        提取文件为 BaseResult。子类应优先实现此方法。
        默认不实现，perceive() 会 fallback 到 parse()。
        """
        raise NotImplementedError

    async def perceive(self, file_path: Path, **context) -> "PerceptionResult":
        """
        新统一接口: 文件 → PerceptionResult (一步到位)。

        默认实现: to_base_result() → Builder → PerceptionResult。
        如果子类未实现 to_base_result(), 则 fallback 到 parse() → to_perception_result()。
        """
        try:
            base_result = await self.to_base_result(file_path)
            from docmirror.models.builder import PerceptionResultBuilder
            return PerceptionResultBuilder.build(base_result, **context)
        except NotImplementedError:
            # fallback 到旧接口
            result = await self.parse(file_path)
            return result.to_perception_result(**context)

    async def parse(self, file_path: Path, **kwargs) -> ParserOutput:
        """
        [DEPRECATED] 请实现 to_base_result() 替代。
        保留此方法仅为向后兼容。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement parse(). Use perceive() instead."
        )

