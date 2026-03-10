"""
PDF Adapter — PDF → PerceptionResult

主路径: Orchestrator → EnhancedResult → PerceptionResultBuilder (一步直达)。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from docmirror.framework.base import BaseParser, ParserOutput
from docmirror.models.domain import BaseResult

logger = logging.getLogger(__name__)

# ── Orchestrator 单例 ──
_orchestrator = None

def _get_shared_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from docmirror.framework.orchestrator import Orchestrator
        _orchestrator = Orchestrator()
    return _orchestrator


class PDFAdapter(BaseParser):
    """
    PDF 格式适配器。

    通过共享 Orchestrator 单例完成全流程，
    使用 PerceptionResultBuilder 一步生成 PerceptionResult。
    """

    def __init__(self, enhance_mode: str = "standard", **kwargs):
        self._enhance_mode = enhance_mode

    async def to_base_result(self, file_path: Path, **kwargs) -> BaseResult:
        """PDF → BaseResult (仅核心提取, 不走中间件)。"""
        from docmirror.core.extractor import CoreExtractor
        extractor = CoreExtractor()
        return await extractor.extract(file_path)

    async def perceive(self, file_path: Path, **context):
        """PDF → PerceptionResult (完整管线, 一步直达)。"""
        from docmirror.models.builder import PerceptionResultBuilder

        orchestrator = _get_shared_orchestrator()
        enhanced = await orchestrator.run_pipeline(
            file_path=file_path,
            enhance_mode=self._enhance_mode,
        )

        return PerceptionResultBuilder.build(
            enhanced.base_result,
            enhanced=enhanced,
            **context,
        )

    async def parse(self, file_path: Path, **kwargs) -> ParserOutput:
        """[DEPRECATED] 保留旧接口兼容。"""
        orchestrator = _get_shared_orchestrator()
        enhanced = await orchestrator.run_pipeline(
            file_path=file_path,
            enhance_mode=self._enhance_mode,
            **kwargs,
        )
        output = enhanced.to_parser_output()
        output._enhanced = enhanced
        return output


