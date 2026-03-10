"""
MultiModal Perception Factory
"""

from pathlib import Path
from typing import Union, Optional, TYPE_CHECKING
from docmirror.framework.dispatcher import ParserDispatcher
from docmirror.models.document_types import DocumentType

if TYPE_CHECKING:
    from docmirror.models.perception_result import PerceptionResult

class PerceptionFactory:
    """
    感知工厂，自动根据文件类型及业务需求分发解析任务。
    ParserDispatcher 以类级缓存单例方式管理, 避免重复初始化。
    """
    _dispatcher: Optional[ParserDispatcher] = None

    @classmethod
    def get_dispatcher(cls) -> ParserDispatcher:
        if cls._dispatcher is None:
            cls._dispatcher = ParserDispatcher()
        return cls._dispatcher

    # backward-compat alias
    get_orchestrator = get_dispatcher

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._dispatcher = None

import logging

logger = logging.getLogger(__name__)

# 便捷入口
async def perceive_document(
    file_path: Union[str, Path],
    document_type: DocumentType = DocumentType.OTHER
) -> "PerceptionResult":
    logger.info(f"[PerceptionFactory] ▶ perceive_document | file_path={file_path} | document_type={document_type}")
    dispatcher = PerceptionFactory.get_dispatcher()
    return await dispatcher.process(str(file_path), document_type=document_type)
