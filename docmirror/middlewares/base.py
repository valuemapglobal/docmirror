# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Middleware Base Class and Pipeline Executor
===========================================

Design principles:
    - Each Middleware is an independent, composable Python class.
    - Unified ``process(ParseResult) -> ParseResult`` interface.
    - PipelineExecutor provides per-middleware exception isolation.
    - All data transformations are recorded via Mutations on ParseResult.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.exceptions import MiddlewareError
from ..models.entities.parse_result import ParseResult

logger = logging.getLogger(__name__)


class BaseMiddleware(ABC):
    """
    Abstract Base Class for Middlewares.

    All Middlewares must implement the ``process()`` method.

    Causal Dependency Protocol:
        - ``DEPENDS_ON``: List of middleware class names that MUST run before this one.
        - ``PROVIDES``:   List of data keys this middleware contributes to the result.

    Conventions:
        - Receives a ParseResult, returns the modified ParseResult.
        - Records all transformations via result.record_mutation().
        - Upon failure, should use add_error() rather than raising exceptions.
    """

    # ── Causal Dependency Declarations ──
    DEPENDS_ON: list[str] = []  # Middleware names that must run before this one
    PROVIDES: list[str] = []  # Data keys this middleware contributes

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._name = self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    def should_skip(self, result: ParseResult) -> bool:
        """
        Conditional Skip: If True is returned, the Middleware is skipped.

        Subclasses can override this method to implement conditional logic.
        Default implementation: Checks the ``skip_scenes`` list in config.
        """
        skip_scenes = self.config.get("skip_scenes")
        if skip_scenes:
            return result.entities.document_type in skip_scenes
        return False

    @abstractmethod
    def process(self, result: ParseResult) -> ParseResult:
        """Processes the ParseResult and returns the augmented Result."""
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
    """

    def __init__(
        self,
        fail_strategy: str = "skip",  # "skip" | "abort"
    ):
        self.fail_strategy = fail_strategy

    def execute(
        self,
        middlewares: list[BaseMiddleware],
        result: ParseResult,
    ) -> ParseResult:
        """
        Executes the Middleware Pipeline with parallel batching.

        T4-1: Middlewares with no DEPENDS_ON (or whose dependencies are
        already satisfied) are grouped into parallel batches and executed
        concurrently.  Middlewares with unsatisfied dependencies wait for
        their batch predecessors to complete first.

        Args:
            middlewares: Ordered list of Middlewares.
            result: Initial ParseResult.

        Returns:
            The processed ParseResult.
        """
        logger.info(f"[Middleware] Pipeline \u25b6 {len(middlewares)} middlewares: {[m.name for m in middlewares]}")

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

        step_timings: dict[str, float] = {}

        # ── T4-1: Group into parallel batches ──
        # Batch = set of middlewares whose DEPENDS_ON are all in `completed` set
        completed: set = set()
        remaining = list(middlewares)

        while remaining:
            # Find all middlewares whose dependencies are satisfied
            batch = []
            deferred = []
            for mw in remaining:
                deps_satisfied = all(dep in completed for dep in mw.DEPENDS_ON)
                if deps_satisfied:
                    batch.append(mw)
                else:
                    deferred.append(mw)
            remaining = deferred

            if not batch:
                # Circular dependency or bug — run remaining sequentially
                logger.warning(
                    f"[DocMirror] ⚠ Pipeline: unsatisfied deps for {[m.name for m in remaining]}, running sequentially"
                )
                batch = remaining
                remaining = []

            if len(batch) == 1:
                # Single middleware — run directly (no thread overhead)
                mw = batch[0]
                self._run_single(mw, result, step_timings)
                completed.add(mw.name)
            else:
                # Multiple independent middlewares — run in parallel
                self._run_parallel_batch(batch, result, step_timings, remaining)

                for mw in batch:
                    completed.add(mw.name)

        # Record step timings as structured data (not string in warnings)
        result.entities.domain_specific["step_timings"] = step_timings

        total_mutations = result.mutation_count
        logger.info(
            f"[Middleware] Pipeline ◀ status={result.status.value} | "
            f"total_mutations={total_mutations} | timings={step_timings}"
        )

        return result

    def _run_single(
        self,
        mw: BaseMiddleware,
        result: ParseResult,
        step_timings: dict[str, float],
    ) -> None:
        """Execute a single middleware sequentially (original path)."""
        if mw.should_skip(result):
            logger.info(f"[Middleware] {mw.name} \u23ed skipped")
            step_timings[mw.name] = 0.0
            return

        t0 = time.time()
        try:
            logger.debug(f"[DocMirror] Running {mw.name}...")
            result_new = mw.process(result)
            # Some middlewares may return a new result; merge key fields
            if result_new is not result:
                for attr in ("status",):
                    if hasattr(result_new, attr):
                        setattr(result, attr, getattr(result_new, attr))
                result.entities = result_new.entities
            elapsed = (time.time() - t0) * 1000
            step_timings[mw.name] = round(elapsed, 1)
            num_mutations = sum(1 for m in result.mutations if m.middleware_name == mw.name)
            logger.info(f"[Middleware] {mw.name} \u25c0 {elapsed:.0f}ms | mutations=+{num_mutations}")
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            step_timings[mw.name] = round(elapsed, 1)
            from ..core.exceptions import MiddlewareError

            mw_error = MiddlewareError(str(e), middleware_name=mw.name)
            logger.warning(f"[Middleware] {mw_error}", exc_info=True)
            result.add_error(str(mw_error))
            if self.fail_strategy == "abort":
                logger.warning(f"[Middleware] Pipeline aborted at {mw.name}")
                from ..models.entities.parse_result import ResultStatus

                result.status = ResultStatus.FAILURE

    def _run_parallel_batch(
        self,
        batch: list[BaseMiddleware],
        result: ParseResult,
        step_timings: dict[str, float],
        remaining: list,
    ) -> None:
        """Execute a batch of independent middlewares in parallel threads.

        Thread-safety: each middleware runs process() with a shared lock
        protecting mutation/error list access.
        """
        import threading

        logger.debug(f"[Middleware] T4-1: parallel batch: {[m.name for m in batch]}")

        _lock = threading.Lock()

        def _run_mw(m):
            if m.should_skip(result):
                return (m.name, 0, 0.0, None)
            t0 = time.time()
            try:
                # Middleware calls result.record_mutation() / result.add_error()
                # which append to result.mutations / result.errors lists.
                # We need to serialize these appends.
                # Since Pydantic doesn't allow monkey-patching, we'll
                # use the lock at the process boundary level.
                with _lock:
                    m.process(result)
                elapsed = (time.time() - t0) * 1000
                with _lock:
                    num_mut = sum(1 for mut in result.mutations if mut.middleware_name == m.name)
                return (m.name, num_mut, elapsed, None)
            except Exception as e:
                elapsed = (time.time() - t0) * 1000
                return (m.name, 0, elapsed, e)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {executor.submit(_run_mw, mw): mw for mw in batch}
                for future in concurrent.futures.as_completed(futures):
                    name, num_mut, elapsed, error = future.result()
                    step_timings[name] = round(elapsed, 1)
                    if error:
                        from ..core.exceptions import MiddlewareError

                        mw_error = MiddlewareError(str(error), middleware_name=name)
                        logger.warning(f"[Middleware] {mw_error}", exc_info=True)
                        result.add_error(str(mw_error))
                        if self.fail_strategy == "abort":
                            logger.warning(f"[Middleware] Pipeline aborted at {name}")
                            from ..models.entities.parse_result import ResultStatus

                            result.status = ResultStatus.FAILURE
                            remaining.clear()
                            break
                    elif elapsed > 0:
                        logger.info(f"[Middleware] {name} ◀ {elapsed:.0f}ms | mutations=+{num_mut}")
                    else:
                        logger.info(f"[Middleware] {name} ⏭ skipped")
        except Exception as e:
            logger.warning(f"[Middleware] Parallel batch error: {e}", exc_info=True)
