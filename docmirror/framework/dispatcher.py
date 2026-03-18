# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
MultiModal Parser Dispatcher
============================

This module is the L0 routing layer of the document parsing system.
It adheres strictly to the "Single Responsibility Principle":
1. L0: I/O validation, file-type detection → dispatch to corresponding parser.
2. L1 & L2: Deep PDF internal business category classification and institution
   identification are delegated explicitly to the `DigitalPDFParser` (or Adapters).

Architecture model:
    - Dispatcher (L0): Rapid file-format routing (PDF, Image, Office...).
    - MultiModalParser/Adapters (L1/L2): Document structural inference.
    - InstitutionPlugin: Deep domain-specific extraction.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

import filetype

# Type alias for progress callbacks.
# Signature: (step_number, total_steps, step_name, detail_message) -> None
ProgressCallback = Callable[[int, int, str, str], None]

from docmirror.framework.base import BaseParser

# Initialize logger
logger = logging.getLogger(__name__)


class ParserDispatcher:
    """
    ParserDispatcher is the entry point for the parsing pipeline.

    Responsibilities:
    - I/O-level file validation (existence, file size safeguards).
    - Fast file-type identification (Magic Number combined with Extension).
    - Optimal parser selection based on detected file type.
    - Handling fallback strategies upon primary parser failure.
    - Delegating security checks (forgery detection).
    """

    async def process(
        self,
        file_path: str | Path,
        fallback: bool = True,
        document_type=None,
        skip_cache: bool = False,
        on_progress: ProgressCallback | None = None,
        **kwargs,
    ) -> Any:
        """
        Main entry point for document parsing.

        Invokes `adapter.perceive()` to directly return a `ParseResult`.
        """
        _t0 = time.time()
        path = Path(file_path)
        _total_steps = 5  # validation, cache, detect, security, parse
        _step_timings: dict[str, float] = {}  # per-step timing breakdown

        def _emit(step: int, name: str, detail: str = ""):
            if on_progress:
                on_progress(step, _total_steps, name, detail)

        def _mark(label: str, since: float) -> float:
            """Record elapsed time for a pipeline step and return current time."""
            now = time.time()
            _step_timings[label] = (now - since) * 1000
            return now

        # ── 1. File context & signature preparation ──
        _emit(1, "Validating file", f"{path.name}")
        _ts = time.time()
        _file_checksum = ""
        _file_mime = ""
        file_size = 0
        _file_mtime = 0.0
        if path.exists():
            try:
                stat = path.stat()
                file_size = stat.st_size
                _file_mtime = stat.st_mtime
                # Fast cache key: (mtime, size, partial_hash) — avoids full-file SHA256
                # but includes first 4KB hash for collision resistance (C2 fix)
                with open(path, "rb") as _f:
                    _head = _f.read(4096)
                _partial = hashlib.md5(_head).hexdigest()[:8]
                _file_checksum = f"fast:{file_size}:{_file_mtime}:{_partial}"
            except OSError as exc:
                logger.debug(f"[Dispatcher] File stat failed: {exc}")
            _ft = filetype.guess(str(path))
            _file_mime = _ft.mime if _ft else ""
        _ts = _mark("1_validation_checksum", _ts)

        # ── 2. Cache lookup (Performance bypass) ──
        _emit(2, "Checking cache", "")
        if not skip_cache and _file_checksum:
            from docmirror.framework.cache import parse_cache

            try:
                cached_json = await parse_cache.get(_file_checksum, document_type or "")
                if cached_json:
                    from docmirror.models.entities.parse_result import ParseResult

                    logger.info(f"[Dispatcher] \u26a1 Cache HIT for {path.name}")
                    _emit(2, "Checking cache", "Cache HIT \u26a1")
                    return ParseResult.model_validate_json(cached_json)
            except Exception as e:
                logger.debug(f"[Dispatcher] Cache lookup error (non-fatal): {e}")

        # ── 3. Physical validation (Constructor Theory Guards) ──
        _file_type = ""
        _is_forged: bool | None = None
        _forgery_reasons: list[str] = []

        if not path.exists():
            return self._build_failure("FILE_NOT_FOUND", f"File not found: {file_path}", _t0, str(path))

        # Guard: file too small for meaningful extraction (Constructor Theory)
        from docmirror.configs.settings import default_settings

        if file_size < default_settings.min_file_size:
            return self._build_failure(
                "FILE_TOO_SMALL",
                f"File too small ({file_size} bytes < {default_settings.min_file_size}B minimum). "
                "No document can encode meaningful content at this size.",
                _t0,
                str(path),
            )

        # Guard: file too large — prevents OOM
        if file_size > default_settings.max_file_size:
            return self._build_failure(
                "FILE_TOO_LARGE",
                f"File too large: {file_size / 1024 / 1024:.1f}MB "
                f"(max {default_settings.max_file_size / 1024 / 1024:.0f}MB)",
                _t0,
                str(path),
            )
        if file_size == 0:
            return self._build_failure("FILE_EMPTY", "File is empty (0 bytes)", _t0, str(path))

        # ── 4. L0 Format Routing ──
        file_type = self._detect_file_type(path, known_mime=_file_mime)
        _file_type = file_type
        _emit(3, "Detecting file type", f"{file_type} ({file_size:,} bytes)")
        logger.info(
            f"[Dispatcher] \u25b6 process | file={path.name} | size={file_size}B | fallback={fallback} | doc_type={document_type}"
        )
        logger.info(f"[Dispatcher] L0 detected file_type={file_type}")

        if document_type:
            kwargs["document_type"] = document_type

        parser = self._get_parser_for_type(file_type)
        if not parser:
            return self._build_failure(
                "UNSUPPORTED_FORMAT", f"Unsupported format: {file_type}", _t0, str(path), file_type=_file_type
            )

        # ── 5. Forgery / Tampering detection (Security pass) ──
        _emit(4, "Security scan", "Checking forgery / tampering")
        try:
            if file_type == "pdf":
                from docmirror.core.security.forgery_detector import detect_pdf_forgery

                _is_forged, _forgery_reasons = detect_pdf_forgery(path)
            elif file_type == "image":
                from docmirror.core.security.forgery_detector import detect_image_forgery

                _is_forged, _forgery_reasons = detect_image_forgery(path)
            _emit(4, "Security scan", "Forged" if _is_forged else "Clean")
        except Exception as e:
            logger.warning(f"[Dispatcher] Forgery Detection Engine error: {e}")
        _ts = _mark("4_security_scan", _ts)

        # ── 6. Parse dispatch \u2014 invoke `perceive()` ──
        pname = parser.__class__.__name__
        # Build standard context for perceive / builder injection
        context = {
            "file_path": str(path),
            "file_type": _file_type,
            "file_size": file_size,
            "parser_name": pname,
            "started_at": datetime.fromtimestamp(_t0),
            "mime_type": _file_mime,
            "checksum": _file_checksum,
            "is_forged": _is_forged,
            "forgery_reasons": _forgery_reasons,
        }
        try:
            _emit(5, "Extracting content", f"via {pname}")
            logger.info(f"[Dispatcher] Dispatching to {pname}")
            # The first positional arg of perceive() is file_path.
            # To avoid "got multiple values for argument 'file_path'",
            # exclude it before expanding **context kwargs.
            perceive_ctx = {k: v for k, v in context.items() if k != "file_path"}
            perception = await parser.perceive(path, **perceive_ctx)

            # ── 7. Fallback error handling (Resilience) ──
            if fallback and (not perception.success or not perception.full_text.strip()):
                fallback_parser = self._get_fallback_parser(file_type)
                if fallback_parser and fallback_parser.__class__ != parser.__class__:
                    logger.info(
                        f"[Dispatcher] Primary {pname} failed/empty, attempting fallback: {fallback_parser.__class__.__name__}"
                    )
                    context["parser_name"] = fallback_parser.__class__.__name__
                    fb_ctx = {k: v for k, v in context.items() if k != "file_path"}
                    fb_perception = await fallback_parser.perceive(path, **fb_ctx)
                    if fb_perception.success and fb_perception.full_text.strip():
                        _elapsed = int((time.time() - _t0) * 1000)
                        fb_perception.parser_info.elapsed_ms = _elapsed
                        logger.info(
                            f"[Dispatcher] \u25c0 process | parser={fallback_parser.__class__.__name__}(fallback) | status={fb_perception.status} | elapsed={_elapsed}ms"
                        )
                        return fb_perception

            # ── 8. Timing + Metric logging ──
            _elapsed = int((time.time() - _t0) * 1000)
            perception.parser_info.elapsed_ms = _elapsed
            _emit(
                5,
                "Extracting content",
                f"{len(perception.full_text):,} chars, {perception.total_tables} tables ({_elapsed}ms)",
            )
            logger.info(
                f"[Dispatcher] \u25c0 process | parser={pname} | status={perception.status} | confidence={perception.confidence:.4f} | text_len={len(perception.full_text)} | tables={perception.total_tables} | forged={_is_forged} | elapsed={_elapsed}ms"
            )

            # ── 9. Write caching (Success paths only) ──
            if _file_checksum and perception.success:
                try:
                    from docmirror.framework.cache import parse_cache

                    await parse_cache.set(
                        _file_checksum,
                        document_type or "",
                        perception.model_dump_json(),
                    )
                except Exception as e:
                    logger.debug(f"[Dispatcher] Cache write error (non-fatal): {e}")

            return perception

        except Exception as e:
            logger.error(f"[Dispatcher] Critical orchestration error: {e}", exc_info=True)
            return self._build_failure(
                "ORCHESTRATION_FAILURE",
                f"Orchestration failure: {str(e)}",
                _t0,
                str(path),
                file_type=_file_type,
                is_forged=_is_forged,
                forgery_reasons=_forgery_reasons,
            )

    @staticmethod
    def _build_failure(
        error_code: str,
        error_msg: str,
        t0: float,
        file_path: str = "",
        file_type: str = "",
        is_forged: bool | None = None,
        forgery_reasons: list[str] | None = None,
    ):
        """Build a failure ParseResult with unified error code (see docmirror.models.errors)."""
        from docmirror.models.errors import build_failure_result

        return build_failure_result(
            code=error_code,
            message=error_msg,
            file_path=file_path,
            file_type=file_type,
            is_forged=is_forged,
            forgery_reasons=forgery_reasons,
            t0=t0,
        )

    def _detect_file_type(self, path: Path, known_mime: str = "") -> str:
        """
        Uses Magic Numbers (`filetype` lib) backed by file-extension combinations
        for robust composite type detection.

        Args:
            path: File path to detect.
            known_mime: Pre-detected MIME type to avoid redundant filetype.guess() call.
        """
        mime = known_mime
        if not mime:
            try:
                kind = filetype.guess(str(path))
                if kind:
                    mime = kind.mime
            except Exception as exc:
                logger.debug(f"_detect_file_type: suppressed {exc}")

        if mime:
            if mime == "application/pdf":
                return "pdf"
            if mime.startswith("image/"):
                return "image"

        ext = path.suffix.lower()
        mapping = {
            ".pdf": "pdf",
            ".doc": "word",
            ".docx": "word",
            ".xlsx": "excel",
            ".pptx": "ppt",
            ".png": "image",
            ".jpg": "image",
            ".jpeg": "image",
            ".tiff": "image",
            ".bmp": "image",
            ".json": "structured",
            ".xml": "structured",
            ".csv": "structured",
            ".eml": "email",
            ".msg": "email",
            ".html": "web",
            ".htm": "web",
        }
        return mapping.get(ext, "unknown")

    def _get_parser_for_type(self, file_type: str) -> BaseParser | None:
        """L0 static mapping lookup table — prefers the Adapter pattern layer for decoupling.

        PDF resolution uses the unified MultiModal Pipeline (via PDFAdapter routing).
        """
        import os

        # Dict Dispatch: file type → (adapter_module_path, class_name, kwargs)
        _PARSER_REGISTRY = {
            "pdf": (
                "docmirror.adapters",
                "PDFAdapter",
                {"enhance_mode": os.environ.get("DOCMIRROR_ENHANCE_MODE", "standard")},
            ),
            "image": (
                "docmirror.adapters",
                "PDFAdapter",
                {"enhance_mode": os.environ.get("DOCMIRROR_ENHANCE_MODE", "standard")},
            ),
            "word": ("docmirror.adapters", "WordAdapter", {}),
            "excel": ("docmirror.adapters", "ExcelAdapter", {}),
            "ppt": ("docmirror.adapters", "PPTAdapter", {}),
            "email": ("docmirror.adapters", "EmailAdapter", {}),
            "structured": ("docmirror.adapters", "StructuredAdapter", {}),
            "web": ("docmirror.adapters", "WebAdapter", {}),
        }

        entry = _PARSER_REGISTRY.get(file_type)
        if entry is None:
            return None

        module_path, class_name, kwargs = entry
        import importlib

        mod = importlib.import_module(module_path)
        adapter_cls = getattr(mod, class_name)

        if file_type in ("pdf", "image"):
            logger.info(
                "[Dispatcher] Using PDFAdapter (unified pipeline for %s)",
                file_type,
            )
        return adapter_cls(**kwargs)

    def _get_fallback_parser(self, file_type: str) -> BaseParser | None:
        """
        Defines fallback parser strategy when primary orchestrator fails.

        Current policy:
        For PDF, the MultiModal pipeline internally integrates all OCR/vLM/Heuristic mechanisms.
        We no longer downgrade to legacy scanned-document parsers at the dispatcher level.
        If MultiModal yields a terminal failure, we return None.
        """
        if file_type == "pdf":
            # MultiModal internally cascades fallbacks; do not attempt L0 downgrade.
            return None
        return None
