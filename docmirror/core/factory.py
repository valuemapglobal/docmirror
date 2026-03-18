# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
MultiModal Perception Factory
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from docmirror.framework.dispatcher import ParserDispatcher
from docmirror.models.entities.document_types import DocumentType

if TYPE_CHECKING:
    from docmirror.models.entities.parse_result import ParseResult


class PerceptionFactory:
    """
    Perception factory that auto-dispatches parsing tasks by file type.
    ParserDispatcher is managed as a class-level cached singleton to avoid re-initialization.
    """

    _dispatcher: ParserDispatcher | None = None
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
    file_path: str | Path,
    document_type: DocumentType = DocumentType.OTHER,
    skip_cache: bool = False,
) -> ParseResult:
    logger.info(f"[PerceptionFactory] ▶ perceive_document | file_path={file_path} | document_type={document_type}")
    dispatcher = PerceptionFactory.get_dispatcher()
    return await dispatcher.process(str(file_path), document_type=document_type, skip_cache=skip_cache)
