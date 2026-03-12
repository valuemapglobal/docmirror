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

import logging
import filetype
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union, Any, List

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
        file_path: Union[str, Path], 
        fallback: bool = True,
        document_type=None,
        skip_cache: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        **kwargs
    ) -> Any:
        """
        Main entry point for document parsing.

        Invokes `adapter.perceive()` to directly return a `PerceptionResult` (new path),
        bypassing the legacy `parse()` interface.
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
        import hashlib
        _file_checksum = ""
        _file_mime = ""
        file_size = 0
        if path.exists():
            try:
                # Memory Optimization: chunked read for large files instead of loading all bytes
                file_size = path.stat().st_size
                if hasattr(hashlib, "file_digest"):
                    # Python 3.11+
                    with open(path, "rb") as f:
                        _file_checksum = hashlib.file_digest(f, "sha256").hexdigest()
                else:
                    # Python < 3.11 chunked fallback
                    sha256_hash = hashlib.sha256()
                    with open(path, "rb") as f:
                        for byte_block in iter(lambda: f.read(4096 * 64), b""):
                            sha256_hash.update(byte_block)
                    _file_checksum = sha256_hash.hexdigest()
            except OSError:
                pass
            _ft = filetype.guess(str(path))
            _file_mime = _ft.mime if _ft else ""
        _ts = _mark('1_validation_checksum', _ts)

        # ── 2. Cache lookup (Performance bypass) ──
        _emit(2, "Checking cache", "")
        if not skip_cache and _file_checksum:
            from docmirror.framework.cache import parse_cache
            try:
                cached_json = await parse_cache.get(_file_checksum, document_type or "")
                if cached_json:
                    from docmirror.models.perception_result import PerceptionResult
                    logger.info(f"[Dispatcher] \u26a1 Cache HIT for {path.name}")
                    _emit(2, "Checking cache", "Cache HIT ⚡")
                    return PerceptionResult.model_validate_json(cached_json)
            except Exception as e:
                logger.debug(f"[Dispatcher] Cache lookup error (non-fatal): {e}")

        # ── 3. Physical validation (Constructor Theory Guards) ──
        _file_type = ""
        _is_forged: Optional[bool] = None
        _forgery_reasons: List[str] = []

        if not path.exists():
            return self._build_failure(f"File not found: {file_path}", _t0, str(path))

        # Guard: file too small for meaningful extraction (Constructor Theory)
        from docmirror.configs.settings import default_settings
        if file_size < default_settings.min_file_size:
            return self._build_failure(
                f"File too small ({file_size} bytes < {default_settings.min_file_size}B minimum). "
                "No document can encode meaningful content at this size.",
                _t0, str(path),
            )

        # Guard: file too large — prevents OOM
        if file_size > default_settings.max_file_size:
            return self._build_failure(
                f"File too large: {file_size / 1024 / 1024:.1f}MB "
                f"(max {default_settings.max_file_size / 1024 / 1024:.0f}MB)",
                _t0, str(path),
            )
        if file_size == 0:
            return self._build_failure("File is empty (0 bytes)", _t0, str(path))

        # ── 4. L0 Format Routing ──
        file_type = self._detect_file_type(path)
        _file_type = file_type
        _emit(3, "Detecting file type", f"{file_type} ({file_size:,} bytes)")
        logger.info(f"[Dispatcher] \u25b6 process | file={path.name} | size={file_size}B | fallback={fallback} | doc_type={document_type}")
        logger.info(f"[Dispatcher] L0 detected file_type={file_type}")

        if document_type:
            kwargs["document_type"] = document_type

        parser = self._get_parser_for_type(file_type)
        if not parser:
            return self._build_failure(f"Unsupported format: {file_type}", _t0, str(path), file_type=_file_type)

        # ── 5. Forgery / Tampering detection (Security pass) ──
        _emit(4, "Security scan", "Checking forgery / tampering")
        try:
            if file_type == 'pdf':
                from docmirror.core.security.forgery_detector import detect_pdf_forgery
                _is_forged, _forgery_reasons = detect_pdf_forgery(path)
            elif file_type == 'image':
                from docmirror.core.security.forgery_detector import detect_image_forgery
                _is_forged, _forgery_reasons = detect_image_forgery(path)
            _emit(4, "Security scan", "Forged" if _is_forged else "Clean")
        except Exception as e:
            logger.warning(f"Forgery Detection Engine error: {e}")
        _ts = _mark('4_security_scan', _ts)

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
            logger.info(f"Dispatching to {pname}")
            # The first positional arg of perceive() is file_path.
            # To avoid "got multiple values for argument 'file_path'",
            # exclude it before expanding **context kwargs.
            perceive_ctx = {k: v for k, v in context.items() if k != "file_path"}
            perception = await parser.perceive(path, **perceive_ctx)

            # ── 7. Fallback error handling (Resilience) ──
            if fallback and (not perception.success or not perception.content.text.strip()):
                fallback_parser = self._get_fallback_parser(file_type)
                if fallback_parser and fallback_parser.__class__ != parser.__class__:
                    logger.info(f"Primary {pname} failed/empty, attempting fallback: {fallback_parser.__class__.__name__}")
                    context["parser_name"] = fallback_parser.__class__.__name__
                    fb_ctx = {k: v for k, v in context.items() if k != "file_path"}
                    fb_perception = await fallback_parser.perceive(path, **fb_ctx)
                    if fb_perception.success and fb_perception.content.text.strip():
                        _elapsed = int((time.time() - _t0) * 1000)
                        fb_perception.timing.elapsed_ms = _elapsed
                        logger.info(f"[Dispatcher] \u25c0 process | parser={fallback_parser.__class__.__name__}(fallback) | status={fb_perception.status} | elapsed={_elapsed}ms")
                        return fb_perception

            # ── 8. Timing + Metric logging ──
            _elapsed = int((time.time() - _t0) * 1000)
            perception.timing.elapsed_ms = _elapsed
            _emit(5, "Extracting content", f"{len(perception.content.text):,} chars, {len(perception.tables)} tables ({_elapsed}ms)")
            logger.info(f"[Dispatcher] \u25c0 process | parser={pname} | status={perception.status} | confidence={perception.confidence:.4f} | text_len={len(perception.content.text)} | tables={len(perception.tables)} | forged={_is_forged} | elapsed={_elapsed}ms")

            # ── 9. Write caching (Success paths only) ──
            if _file_checksum and perception.success:
                try:
                    from docmirror.framework.cache import parse_cache
                    await parse_cache.set(
                        _file_checksum, document_type or "",
                        perception.model_dump_json(),
                    )
                except Exception as e:
                    logger.debug(f"[Dispatcher] Cache write error (non-fatal): {e}")

            return perception

        except Exception as e:
            logger.error(f"Critical orchestration error: {e}", exc_info=True)
            return self._build_failure(f"Orchestration failure: {str(e)}", _t0, str(path),
                                       file_type=_file_type, is_forged=_is_forged, forgery_reasons=_forgery_reasons)

    @staticmethod
    def _build_failure(
        error_msg: str,
        t0: float,
        file_path: str = "",
        file_type: str = "",
        is_forged: Optional[bool] = None,
        forgery_reasons: Optional[List[str]] = None,
    ):
        """Build an elegant failure PerceptionResult (replaces legacy tuple/ParserOutput wrapping)."""
        from docmirror.models.perception_result import (
            PerceptionResult, ResultStatus, ErrorDetail, TimingInfo, Provenance, SourceInfo,
            DocumentContent, ValidationResult,
        )
        elapsed = (time.time() - t0) * 1000
        validation = None
        if is_forged is not None:
            validation = ValidationResult(is_forged=is_forged, forgery_reasons=forgery_reasons or [])
        return PerceptionResult(
            status=ResultStatus.FAILURE,
            confidence=0.0,
            timing=TimingInfo(elapsed_ms=elapsed),
            error=ErrorDetail(message=error_msg),
            content=DocumentContent(),
            provenance=Provenance(
                source=SourceInfo(file_path=file_path, file_type=file_type),
                validation=validation,
            ),
        )

    def _detect_file_type(self, path: Path) -> str:
        """
        Uses Magic Numbers (`filetype` lib) backed by file-extension combinations 
        for robust composite type detection.
        
        Rationale: 
        - Extensions can be mocked or accidentally tampered with.
        - Magic Numbers can map ambiguously in certain older office formats (e.g. DOC vs structured XML).
        Combining both guarantees safety.
        """
        try:
            kind = filetype.guess(str(path))
            if kind:
                mime = kind.mime
                if mime == 'application/pdf':
                    return 'pdf'
                if mime.startswith('image/'):
                    return 'image'
        except Exception as exc:
            logger.debug(f"_detect_file_type: suppressed {exc}")

        ext = path.suffix.lower()
        mapping = {
            '.pdf': 'pdf',
            '.docx': 'word',
            '.xlsx': 'excel',
            '.pptx': 'ppt',
            '.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.tiff': 'image', '.bmp': 'image',
            '.json': 'structured', '.xml': 'structured', '.csv': 'structured',
            '.eml': 'email', '.msg': 'email',
            '.html': 'web', '.htm': 'web'
        }
        return mapping.get(ext, 'unknown')

    def _get_parser_for_type(self, file_type: str) -> Optional[BaseParser]:
        """
        L0 static mapping lookup table \u2014 prefers the Adapter pattern layer for decoupling.
        PDF resolution uses the unified MultiModal Pipeline (via PDFAdapter routing).
        """
        if file_type == 'pdf':
            import os
            from docmirror.adapters import PDFAdapter
            logger.info("[Dispatcher] Using PDFAdapter (promoted)")
            return PDFAdapter(enhance_mode=os.environ.get("DOCMIRROR_ENHANCE_MODE", "standard"))
        elif file_type == 'image':
            from docmirror.adapters import ImageAdapter
            return ImageAdapter()
        elif file_type == 'word':
            from docmirror.adapters import WordAdapter
            return WordAdapter()
        elif file_type == 'excel':
            from docmirror.adapters import ExcelAdapter
            return ExcelAdapter()
        elif file_type == 'ppt':
            from docmirror.adapters import PPTAdapter
            return PPTAdapter()
        elif file_type == 'email':
            from docmirror.adapters import EmailAdapter
            return EmailAdapter()
        elif file_type == 'structured':
            from docmirror.adapters import StructuredAdapter
            return StructuredAdapter()
        elif file_type == 'web':
            from docmirror.adapters import WebAdapter
            return WebAdapter()
        return None

    def _get_fallback_parser(self, file_type: str) -> Optional[BaseParser]:
        """
        Defines fallback parser strategy when primary orchestrator fails.
        
        Current policy:
        For PDF, the MultiModal pipeline internally integrates all OCR/vLM/Heuristic mechanisms.
        We no longer downgrade to legacy scanned-document parsers at the dispatcher level.
        If MultiModal yields a terminal failure, we return None.
        """
        if file_type == 'pdf':
            # MultiModal internally cascades fallbacks; do not attempt L0 downgrade.
            return None
        return None
