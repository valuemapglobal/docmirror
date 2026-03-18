# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
MultiModal Parsing Contract Layer
=================================

Core Components:
1. ParserStatus: Parsing lifecycle status enumeration.
2. BaseParser: Abstract base class — all adapters implement to_parse_result().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path


class ParserStatus(str, Enum):
    """Parsing status enumeration."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


class BaseParser(ABC):
    """
    Abstract base class for all document parsers.

    All adapters MUST implement ``to_parse_result()``.
    The ``perceive()`` pipeline handles middleware enhancement automatically.
    """

    @abstractmethod
    async def to_parse_result(self, file_path: Path, **kwargs) -> ParseResult:
        """
        Extract the file into a ParseResult.

        Each adapter controls how it extracts content and which
        ParseResult types it uses (CellValue, TextBlock, etc.).
        """
        ...

    async def perceive(self, file_path: Path, **context) -> ParseResult:
        """
        Unified pipeline: file → ParseResult → middleware → enhanced ParseResult.

        Pipeline:
            1. ``to_parse_result()`` → ParseResult
            2. Fill provenance (Zone 5) — automatic for all adapters
            3. ``Orchestrator.enhance()`` → middleware enrichment in-place
        """
        from docmirror.framework.orchestrator import Orchestrator

        pr = await self.to_parse_result(file_path)

        # ── Fill provenance (Zone 5) for all adapters ──
        if pr.provenance is None:
            import hashlib
            import mimetypes

            from docmirror.models.entities.parse_result import ProvenanceInfo

            file_stat = file_path.stat()
            with open(file_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            suffix = file_path.suffix.lstrip(".").lower()
            pr.provenance = ProvenanceInfo(
                file_type=suffix,
                file_size=file_stat.st_size,
                checksum=checksum,
                mime_type=mime_type,
            )

        # ── Fill parser_version if empty ──
        if not pr.parser_info.parser_version:
            import docmirror

            pr.parser_info.parser_version = getattr(docmirror, "__version__", "2.1.0")

        orchestrator = Orchestrator()
        file_type = (context.get("file_type") or pr.provenance.file_type or "unknown").lower()
        enhance_mode = context.get("enhance_mode", "standard")

        return await orchestrator.enhance(
            pr,
            enhance_mode=enhance_mode,
            file_type=file_type,
        )
