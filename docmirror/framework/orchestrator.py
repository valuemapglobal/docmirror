# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Orchestration Layer
====================

The "Brain" of DocMirror — orchestrates the middleware pipeline:
    1. Receives a ``ParseResult`` from the adapter.
    2. Dynamically builds a ``MiddlewarePipeline`` based on ``enhance_mode``.
    3. Executes the pipeline, enriching the ParseResult in-place.
    4. Returns the enhanced ParseResult.

Three Enhancement Modes:
    - ``raw``:      No enrichment.
    - ``standard``: SceneDetector + EntityExtractor + InstitutionDetector + Validator.

Exception Downgrade Strategy:
    - If a middleware fails, the pipeline skips it and continues.
    - Guarantees a valid payload even under catastrophic halts.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

from ..configs.settings import DocMirrorSettings
from ..middlewares import (
    BaseMiddleware,
    EntityExtractor,
    GenericEntityExtractor,
    InstitutionDetector,
    LanguageDetector,
    MiddlewarePipeline,
    SceneDetector,
    Validator,
)
from ..models.entities.parse_result import ParseResult, ResultStatus

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Middleware Registry — Open/Closed Principle
# ═══════════════════════════════════════════════════════════════════════════════

MIDDLEWARE_REGISTRY: dict[str, type[BaseMiddleware]] = {
    "SceneDetector": SceneDetector,
    "EntityExtractor": EntityExtractor,
    "InstitutionDetector": InstitutionDetector,
    "Validator": Validator,
    # ── Cross-format generic middlewares ──
    "LanguageDetector": LanguageDetector,
    "GenericEntityExtractor": GenericEntityExtractor,
}


class Orchestrator:
    """
    DocMirror Orchestrator — manages the middleware enhancement pipeline.

    Usage::

        orchestrator = Orchestrator()
        result = await orchestrator.enhance(parse_result, enhance_mode="standard")
    """

    def __init__(
        self,
        settings: DocMirrorSettings | None = None,
        config: dict[str, Any] | None = None,
        fail_strategy: str | None = None,
        seal_detector_fn: Callable | None = None,
    ):
        self.settings = settings or DocMirrorSettings.from_env()
        self.config = config or self.settings.to_dict()
        self.pipeline = MiddlewarePipeline(fail_strategy=fail_strategy or self.settings.fail_strategy)

    async def enhance(
        self,
        result: ParseResult,
        enhance_mode: Literal["raw", "standard", "full"] = "standard",
        file_type: str = "unknown",
        **kwargs,
    ) -> ParseResult:
        """
        Run the middleware pipeline on a ParseResult.

        Args:
            result:       ParseResult from adapter's to_parse_result().
            enhance_mode: Depth of enhancements (raw/standard/full).
            file_type:    Document type hint (pdf, image, word, excel, ...).

        Returns:
            ParseResult: The same object, enriched in-place with entities,
                         trust scores, scene detection, etc.
        """
        t0 = time.time()

        logger.info(
            f"[Orchestrator] Pipeline ▶ mode={enhance_mode} | file_type={file_type} | pages={result.page_count}"
        )

        # ═══ Step 1: Validate extraction baseline ═══
        if not result.pages:
            logger.warning("[Orchestrator] Empty ParseResult — no pages")
            result.status = ResultStatus.FAILURE
            result.add_error("Empty extraction result")
            return result

        # ═══ Step 2: Middleware Pipeline ═══
        effective_mode = enhance_mode

        if effective_mode == "raw":
            logger.info("[Orchestrator] Raw mode — skipping middleware pipeline")
        else:
            middlewares = self._build_middlewares(effective_mode, file_type)
            result = self.pipeline.execute(middlewares, result)

        # ═══ Step 3: Trace Instrumentation ═══
        elapsed = (time.time() - t0) * 1000
        result.processing_time = elapsed
        result.parser_info.elapsed_ms = elapsed

        # ═══ Step 4: Mutation Auditing ═══
        if result.mutations:
            try:
                from ..middlewares import MutationAnalyzer

                analyzer = MutationAnalyzer()
                analysis = analyzer.analyze(result.mutations)
                result.entities.domain_specific["mutation_analysis"] = analysis.to_dict()
            except Exception as e:
                logger.debug(f"[Orchestrator] MutationAnalyzer error bypass: {e}")

        logger.info(
            f"[Orchestrator] Pipeline ◀ status={result.status.value} | "
            f"scene={result.entities.document_type} | "
            f"mutations={result.mutation_count} | "
            f"elapsed={elapsed:.0f}ms"
        )

        return result

    # Legacy alias for backward compatibility
    async def run_pipeline(
        self,
        file_path: Path,
        enhance_mode: Literal["raw", "standard", "full"] = "standard",
        file_type: str = "pdf",
        *,
        pre_extracted: Any | None = None,
        **kwargs,
    ) -> ParseResult:
        """Legacy wrapper — delegates to enhance()."""
        if pre_extracted is not None and isinstance(pre_extracted, ParseResult):
            return await self.enhance(pre_extracted, enhance_mode, file_type)
        # If pre_extracted is a BaseResult, convert first
        if pre_extracted is not None:
            from ..models.construction.parse_result_bridge import ParseResultBridge

            pr = ParseResultBridge.from_base_result(pre_extracted)
            return await self.enhance(pr, enhance_mode, file_type)
        # Fallback: extract from file
        from ..core.extraction.extractor import CoreExtractor

        extractor = CoreExtractor()
        base_result = await extractor.extract(file_path)
        from ..models.construction.parse_result_bridge import ParseResultBridge

        pr = ParseResultBridge.from_base_result(base_result)
        return await self.enhance(pr, enhance_mode, file_type)

    def _build_middlewares(
        self,
        enhance_mode: str,
        file_type: str = "pdf",
    ) -> list[BaseMiddleware]:
        """Build middleware list from pipeline config."""
        from ..configs.pipeline_registry import get_pipeline_config

        middleware_names = get_pipeline_config(file_type, enhance_mode)
        middlewares = []

        for name in middleware_names:
            cls = MIDDLEWARE_REGISTRY.get(name)
            if cls is None:
                logger.warning(f"[Orchestrator] Unresolved middleware: {name}")
                continue
            mw_config = self.config.get(name, {})
            middlewares.append(cls(config=mw_config))

        return middlewares
