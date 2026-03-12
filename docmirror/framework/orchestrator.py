"""
Orchestration Layer
====================

The "Brain" of the MultiModal system — responsible for orchestrating the entire
extract-and-enhance pipeline:
    1. Invokes ``CoreExtractor`` to generate a baseline ``BaseResult``.
    2. Dynamically builds a ``MiddlewarePipeline`` based on ``enhance_mode``.
    3. Executes the pipeline, sequentially enriching and validating the result.
    4. Bridges the final output into an API-compatible data structure.

Three Enhancement Modes:
    - ``raw``:      Base extraction only, no enrichment.
    - ``standard``: SceneDetector + EntityExtractor + InstitutionDetector + Validator.

Exception Downgrade Strategy:
    - If a middleware fails, by default the pipeline skips it and continues executing.
    - Guarantees the return of a valid payload even under catastrophic internal
      halts (returning ``status="partial"``).
"""
from __future__ import annotations


import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from ..core.extraction.extractor import CoreExtractor
from ..middlewares.base import BaseMiddleware, MiddlewarePipeline
from ..middlewares.scene_detector import SceneDetector
from ..middlewares.institution_detector import InstitutionDetector
from ..middlewares.entity_extractor import EntityExtractor
from ..middlewares.validator import Validator

from ..middlewares.language_detector import LanguageDetector
from ..middlewares.generic_entity_extractor import GenericEntityExtractor
from ..models.enhanced import EnhancedResult
from ..configs.settings import DocMirrorSettings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Middleware Registry — Conforms to Open/Closed Principle
# ═══════════════════════════════════════════════════════════════════════════════

MIDDLEWARE_REGISTRY: Dict[str, Type[BaseMiddleware]] = {
    "SceneDetector": SceneDetector,
    "EntityExtractor": EntityExtractor,
    "InstitutionDetector": InstitutionDetector,
    "Validator": Validator,
    # ── Cross-format generic middlewares ──
    "LanguageDetector": LanguageDetector,
    "GenericEntityExtractor": GenericEntityExtractor,
}




# ═══════════════════════════════════════════════════════════════════════════════
# hints.yaml File Cache (based on mtime checks)
# ═══════════════════════════════════════════════════════════════════════════════

_hints_cache: Optional[Dict[str, Any]] = None
_hints_mtime: float = 0.0


def _load_hints_cached() -> Dict[str, Any]:
    """Load configuration from `hints.yaml` (utilizing mtime-based caching)."""
    global _hints_cache, _hints_mtime
    try:
        import yaml
        hints_path = Path(__file__).resolve().parent.parent / "configs" / "hints.yaml"
        if hints_path.exists():
            mtime = hints_path.stat().st_mtime
            if mtime != _hints_mtime or _hints_cache is None:
                with open(hints_path, "r", encoding="utf-8") as f:
                    _hints_cache = yaml.safe_load(f) or {}
                _hints_mtime = mtime
                logger.debug("[DocMirror] hints.yaml reloaded (mtime changed)")
    except Exception as e:
        logger.debug(f"[DocMirror] Failed to load hints.yaml: {e}")
    return _hints_cache or {}


class Orchestrator:
    """
    MultiModal Orchestrator — Manages the full extraction lifecycle.

    Usage::

        orchestrator = Orchestrator()
        result = await orchestrator.run_pipeline(
            file_path=Path("bank_statement.pdf"),
            enhance_mode="full",
        )

        # Retrieve a backwards-compatible output envelope
        parser_output = result.to_parser_output()
    """

    def __init__(
        self,
        settings: Optional[DocMirrorSettings] = None,
        config: Optional[Dict[str, Any]] = None,
        fail_strategy: Optional[str] = None,
        seal_detector_fn: Optional[Callable] = None,
    ):
        self.settings = settings or DocMirrorSettings.from_env()
        self.config = config or self.settings.to_dict()
        self.extractor = CoreExtractor(seal_detector_fn=seal_detector_fn)
        self.pipeline = MiddlewarePipeline(
            fail_strategy=fail_strategy or self.settings.fail_strategy
        )

    async def run_pipeline(
        self,
        file_path: Path,
        enhance_mode: Literal["raw", "standard", "full"] = "standard",
        file_type: str = "pdf",
        **kwargs,
    ) -> EnhancedResult:
        """
        Execute the complete document processing pipeline.

        Args:
            file_path:    Valid path to the target document.
            enhance_mode: Depth of enhancements applied (raw/standard/full).
            file_type:    Injected document type (pdf, image, word).

        Returns:
            EnhancedResult: Aggregated payload containing the underlying
                            BaseResult + injected domain data + applied mutations.
        """
        t0 = time.time()

        logger.info(
            f"[DocMirror] Orchestrator \u25b6 "
            f"file={Path(file_path).name} | mode={enhance_mode}"
        )

        # \u2550\u2550\u2550 Step 1: Core Physical Extraction \u2192 Returns BaseResult \u2550\u2550\u2550
        base_result = await self.extractor.extract(file_path)

        # Validate extraction baseline viability
        if not base_result.pages and not base_result.full_text:
            error_msg = base_result.metadata.get("error", "Empty extraction result")
            logger.warning(f"[DocMirror] Extraction failed: {error_msg}")
            result = EnhancedResult.from_base_result(base_result)
            result.status = "failed"
            result.add_error(error_msg)
            result.enhanced_data["enhance_mode"] = enhance_mode
            return result

        # \u2550\u2550\u2550 Step 2: Initialize Enrichment Lifecycle Wrapper \u2550\u2550\u2550
        result = EnhancedResult.from_base_result(base_result)
        result.enhanced_data["enhance_mode"] = enhance_mode

        # \u2550\u2550\u2550 Step 2.5: Adaptive Strategy Downgrades (driven by PreAnalyzer) \u2550\u2550\u2550
        pre_analysis = base_result.metadata.get("pre_analysis", {})
        recommended = pre_analysis.get("recommended_strategy", "standard")
        strategy_params = pre_analysis.get("strategy_params", {})
        result.enhanced_data["pre_analysis"] = pre_analysis

        # `fast` Strategy overrides: forcibly downgrade target execution mode
        effective_mode = enhance_mode
        if recommended == "fast" and enhance_mode == "full":
            effective_mode = "standard"
            logger.info("[DocMirror] PreAnalyzer: 'fast' strategy engaged \u2192 downgraded full\u2192standard")
            
        # Optional LLM Escalation: conditionally inject LLM flags dynamically
        if strategy_params.get("enable_llm", False):
            self.config.setdefault("SceneDetector", {})["enable_llm"] = True
            logger.info("[DocMirror] PreAnalyzer: 'deep' strategy engaged \u2192 LLM execution unblocked")

        # \u2550\u2550\u2550 Step 3: Middleware Pipeline Compilation & Execution \u2550\u2550\u2550
        if effective_mode == "raw":
            logger.info("[DocMirror] Raw mode selected \u2014 skipping middleware pipeline entirely")
        else:
            middlewares = self._build_middlewares(effective_mode, file_type)
            result = self.pipeline.execute(middlewares, result)

        # \u2550\u2550\u2550 Step 4: Trace Instrumentation & Status Seal \u2550\u2550\u2550
        elapsed = (time.time() - t0) * 1000
        result.processing_time = elapsed

        # \u2550\u2550\u2550 Step 4.5: Analytical Mutation Auditing (Self-Correcting Loop Feedback) \u2550\u2550\u2550
        if result.mutations:
            try:
                from .middlewares.mutation_analyzer import MutationAnalyzer
                analyzer = MutationAnalyzer()
                analysis = analyzer.analyze(result.mutations)
                result.enhanced_data["mutation_analysis"] = analysis.to_dict()
            except Exception as e:
                logger.debug(f"[DocMirror] MutationAnalyzer error bypass: {e}")

        # Safety Fallback: Ensure structurally bound blocks were identified
        if not base_result.table_blocks:
            if result.status == "success":
                result.status = "partial"
                result.add_error("No tables found in document layout")

        logger.info(
            f"[DocMirror] Orchestrator \u25c0 status={result.status} | "
            f"scene={result.scene} | "
            f"mutations={result.mutation_count} | "
            f"elapsed={elapsed:.0f}ms"
        )

        return result

    def _build_middlewares(
        self, enhance_mode: str, file_type: str = "pdf",
    ) -> List[BaseMiddleware]:
        """
        Builds an active List of Middleware instances mapping to the target 
        format and execution mode criteria.

        Extensibility Contract:
        To append a new Middleware into the pipeline simply:
          1. Register its Class mapping within `MIDDLEWARE_REGISTRY`.
          2. Splice its identifier explicitly into `configs/pipeline_registry.py` definitions.
        """
        from ..configs.pipeline_registry import get_pipeline_config
        middleware_names = get_pipeline_config(file_type, enhance_mode)
        middlewares = []

        for name in middleware_names:
            cls = MIDDLEWARE_REGISTRY.get(name)
            if cls is None:
                logger.warning(f"[DocMirror] Unresolved middleware configuration request: {name}")
                continue

            mw_config = self.config.get(name, {})
            middlewares.append(cls(config=mw_config))

        return middlewares
