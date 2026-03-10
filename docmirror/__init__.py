"""
MultiModal: Perception Hub (Unified API)

重构后的目录结构:
- core/: 核心提取引擎 (CoreExtractor, Foundation, LayoutAnalysis, TableExtraction)
- models/: 数据模型 (BaseResult, EnhancedResult, Mutation)
- middlewares/: 中间件管线 (SceneDetector, ColumnMapper, Validator, Repairer, ...)
- configs/: 配置文件 (settings, hints.yaml, institution_registry.yaml)
- orchestrator.py: 全流程编排器
- engines/: Vision LLM (Qwen-VL), OCR, Seal 检测
- schemas/: 外部数据契约 (PerceptionResult, DocumentType)
- dispatcher.py: L0 文件类型路由
- base.py: ParserOutput + BaseParser 基类
- adapters/: 格式适配器 (PDF, Image, Office, Email, Web)

唯一公开入口: perceive_document()
"""

import logging
import warnings
from pathlib import Path
from typing import Literal, Optional

from docmirror.core.factory import perceive_document, PerceptionFactory
from docmirror.models.document_types import DocumentType
from docmirror.models.perception_result import PerceptionResult
from docmirror.models.domain_models import DomainData
from docmirror.framework.dispatcher import ParserDispatcher
from docmirror.framework.dispatcher import ParserDispatcher as DocumentProcessingOrchestrator  # compat
from docmirror.framework.base import ParserOutput
from docmirror.framework.orchestrator import Orchestrator
from docmirror.models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)

# backward-compat alias — callers importing PerceptionResponse get ParserOutput
PerceptionResponse = ParserOutput


async def parse_pdf_v2(
    file_path,
    enhance_mode: Literal["raw", "standard", "full"] = "standard",
    **kwargs,
) -> EnhancedResult:
    """
    [DEPRECATED] 请使用 perceive_document() 作为唯一入口。

    保留此函数仅为向后兼容。内部委托给 perceive_document。
    """
    warnings.warn(
        "parse_pdf_v2() is deprecated, use perceive_document() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = await perceive_document(file_path, DocumentType.OTHER)
    # 返回 EnhancedResult 以兼容旧调用方
    if hasattr(result, '_enhanced') and result._enhanced is not None:
        return result._enhanced
    # 如果没有 _enhanced (非 PDF 路径), 构造一个最小 EnhancedResult
    from docmirror.models.domain import BaseResult
    base = BaseResult(document_id=str(file_path), full_text=result.content.text, pages=(), metadata={})
    return EnhancedResult.from_base_result(base)


__all__ = [
    "perceive_document",
    "PerceptionFactory",
    "PerceptionResult",
    "PerceptionResponse",
    "DocumentType",
    "DomainData",
    "DocumentProcessingOrchestrator",
    "ParserOutput",
    "Orchestrator",
    "parse_pdf_v2",
]

