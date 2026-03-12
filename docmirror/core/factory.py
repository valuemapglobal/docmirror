"""
MultiModal Perception Factory
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Union, Optional, TYPE_CHECKING
from docmirror.framework.dispatcher import ParserDispatcher
from docmirror.models.document_types import DocumentType

if TYPE_CHECKING:
    from docmirror.models.perception_result import PerceptionResult

class PerceptionFactory:
    """
    Perception factory that auto-dispatches parsing tasks by file type.
    ParserDispatcher is managed as a class-level cached singleton to avoid re-initialization.
    """
    _dispatcher: Optional[ParserDispatcher] = None
    _lock = threading.Lock()

    @classmethod
    def get_dispatcher(cls) -> ParserDispatcher:
        if cls._dispatcher is None:
            with cls._lock:
                if cls._dispatcher is None:  # double-check
                    cls._dispatcher = ParserDispatcher()
        return cls._dispatcher

    # backward-compat alias
    get_orchestrator = get_dispatcher

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._dispatcher = None

logger = logging.getLogger(__name__)

# Convenience entry point
async def perceive_document(
    file_path: Union[str, Path],
    document_type: DocumentType = DocumentType.OTHER,
    skip_cache: bool = False,
) -> "PerceptionResult":
    logger.info(f"[PerceptionFactory] ▶ perceive_document | file_path={file_path} | document_type={document_type}")
    dispatcher = PerceptionFactory.get_dispatcher()
    return await dispatcher.process(str(file_path), document_type=document_type, skip_cache=skip_cache)
