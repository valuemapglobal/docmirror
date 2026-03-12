"""
Middleware Base Class and Pipeline Executor
===========================================

Design principles:
    - Each Middleware is an independent, composable Python class.
    - Unified ``process(EnhancedResult) -> EnhancedResult`` interface.
    - PipelineExecutor provides per-middleware exception isolation.
    - All data transformations are recorded via Mutations, without
      directly modifying the BaseResult.
"""
from __future__ import annotations


import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models.enhanced import EnhancedResult
from ..core.exceptions import MiddlewareError

logger = logging.getLogger(__name__)


class BaseMiddleware(ABC):
    """
    Abstract Base Class for Middlewares.

    All Middlewares must implement the ``process()`` method.

    Causal Dependency Protocol (Deutsch V5: 'hard to vary' ordering):
        - ``DEPENDS_ON``: List of middleware class names that MUST run before this one.
        - ``PROVIDES``:   List of data keys this middleware contributes to the result.
        These declarations make the pipeline execution order *causally justified*
        rather than an arbitrary convention. The Orchestrator can topologically
        sort middlewares based on these declarations.

    Conventions:
        - Receives an EnhancedResult, returns the modified EnhancedResult.
        - Records all transformations via result.record_mutation().
        - Upon failure, should use add_error() rather than raising exceptions.
    """

    # ── Causal Dependency Declarations ──
    DEPENDS_ON: List[str] = []   # Middleware names that must run before this one
    PROVIDES: List[str] = []     # Data keys this middleware contributes

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._name = self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    def should_skip(self, result: EnhancedResult) -> bool:
        """
        Conditional Skip: If True is returned, the Middleware is skipped.

        Subclasses can override this method to implement conditional logic.
        Default implementation: Checks the ``skip_scenes`` list in config.

        Example::

            class BankSpecificMiddleware(BaseMiddleware):
                def should_skip(self, result):
                    return result.scene not in ('bank_statement', 'unknown')
        """
        skip_scenes = self.config.get("skip_scenes")
        if skip_scenes and hasattr(result, "scene"):
            return result.scene in skip_scenes
        return False

    @abstractmethod
    def process(self, result: EnhancedResult) -> EnhancedResult:
        """Processes the EnhancedResult and returns the augmented Result."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}>"


class MiddlewarePipeline:
    """
    Middleware Pipeline Executor.

    Responsibilities:
        1. Sequentially executes the provided list of Middlewares.
        2. Implements per-middleware try/except exception isolation.
        3. Decides whether to [Skip] or [Abort] based on the strategy.
        4. Records the execution time of each Middleware.

    Usage::

        pipeline = MiddlewarePipeline()
        result = pipeline.execute(
            middlewares=[SceneDetector(), EntityExtractor(), Validator()],
            result=initial_result,
        )
    """

    def __init__(
        self,
        fail_strategy: str = "skip",  # "skip" | "abort"
    ):
        self.fail_strategy = fail_strategy

    def execute(
        self,
        middlewares: List[BaseMiddleware],
        result: EnhancedResult,
    ) -> EnhancedResult:
        """
        Sequentially executes the Middleware Pipeline.

        Args:
            middlewares: Ordered list of Middlewares.
            result: Initial EnhancedResult.

        Returns:
            The processed EnhancedResult.
        """
        logger.info(
            f"[DocMirror] Pipeline \u25b6 {len(middlewares)} middlewares: "
            f"{[m.name for m in middlewares]}"
        )

        # ── Validate causal ordering (Deutsch V5) ──
        seen: set = set()
        for mw in middlewares:
            for dep in mw.DEPENDS_ON:
                if dep not in seen:
                    logger.warning(
                        f"[DocMirror] ⚠ Causal violation: {mw.name} depends on "
                        f"{dep}, but {dep} has not run yet in this pipeline."
                    )
            seen.add(mw.name)

        step_timings: Dict[str, float] = {}

        for mw in middlewares:
            # \u2500\u2500\u2500 Conditional Skip Check \u2500\u2500\u2500
            if mw.should_skip(result):
                logger.info(f"[DocMirror] {mw.name} \u23ed skipped")
                step_timings[mw.name] = 0.0
                continue

            t0 = time.time()
            try:
                logger.debug(f"[DocMirror] Running {mw.name}...")
                result = mw.process(result)
                elapsed = (time.time() - t0) * 1000
                step_timings[mw.name] = round(elapsed, 1)
                num_mutations = sum(
                    1 for m in result.mutations if m.middleware_name == mw.name
                )
                logger.info(
                    f"[DocMirror] {mw.name} \u25c0 {elapsed:.0f}ms | "
                    f"mutations=+{num_mutations}"
                )

            except Exception as e:
                elapsed = (time.time() - t0) * 1000
                step_timings[mw.name] = round(elapsed, 1)
                mw_error = MiddlewareError(
                    str(e), middleware_name=mw.name
                )
                logger.warning(f"[DocMirror] {mw_error}", exc_info=True)
                result.add_error(str(mw_error))

                if self.fail_strategy == "abort":
                    logger.warning(
                        f"[DocMirror] Pipeline aborted at {mw.name}"
                    )
                    result.status = "failed"
                    break
                else:
                    logger.info(
                        f"[DocMirror] Skipping {mw.name}, continuing"
                    )

        # Record total execution timings
        result.enhanced_data["step_timings"] = step_timings

        total_mutations = result.mutation_count
        logger.info(
            f"[DocMirror] Pipeline \u25c0 status={result.status} | "
            f"total_mutations={total_mutations}"
        )

        return result
