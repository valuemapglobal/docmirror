"""
中间件基类与管线执行器 (Middleware Base & Pipeline)
====================================================

设计原则:
    - 每个中间件是独立的、可组合的 Python 类
    - 统一 ``process(EnhancedResult) -> EnhancedResult`` 接口
    - 管线执行器提供 per-middleware 异常隔离和降级策略
    - 所有数据变换通过 Mutation 记录，不直接修改 BaseResult
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
    中间件抽象基类。

    所有中间件必须实现 ``process()`` 方法。
    约定:
        - 接收 EnhancedResult，返回修改后的 EnhancedResult
        - 通过 result.record_mutation() 记录所有变换
        - 失败时应 add_error() 而非抛出异常
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._name = self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    def should_skip(self, result: EnhancedResult) -> bool:
        """条件跳过: 返回 True 时整个中间件不执行。

        子类可覆写此方法实现条件跳过逻辑。
        默认实现: 检查 config 中的 ``skip_scenes`` 列表。

        示例::

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
        """处理 EnhancedResult 并返回增强后的结果。"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}>"


class MiddlewarePipeline:
    """
    中间件管线执行器。

    职责:
        1. 顺序执行中间件列表
        2. Per-middleware try/except 异常隔离
        3. 根据策略决定 [跳过失败中间件] 或 [终止管线]
        4. 记录每个中间件的耗时

    使用方式::

        pipeline = MiddlewarePipeline()
        result = pipeline.execute(
            middlewares=[SceneDetector(), ColumnMapper(), Validator()],
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
        顺序执行中间件管线。

        Args:
            middlewares: 有序中间件列表。
            result: 初始 EnhancedResult。

        Returns:
            处理后的 EnhancedResult。
        """
        logger.info(
            f"[DocMirror] Pipeline ▶ {len(middlewares)} middlewares: "
            f"{[m.name for m in middlewares]}"
        )

        step_timings: Dict[str, float] = {}

        for mw in middlewares:
            # ── 条件跳过检查 ──
            if mw.should_skip(result):
                logger.info(f"[DocMirror] {mw.name} ⏭ skipped (should_skip=True)")
                step_timings[mw.name] = 0.0
                continue

            t0 = time.time()
            try:
                logger.debug(f"[DocMirror] Running {mw.name}...")
                result = mw.process(result)
                elapsed = (time.time() - t0) * 1000
                step_timings[mw.name] = round(elapsed, 1)
                logger.info(
                    f"[DocMirror] {mw.name} ◀ {elapsed:.0f}ms | "
                    f"mutations=+{sum(1 for m in result.mutations if m.middleware_name == mw.name)}"
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
                    logger.warning(f"[DocMirror] Pipeline aborted at {mw.name}")
                    result.status = "failed"
                    break
                else:
                    logger.info(f"[DocMirror] Skipping {mw.name}, continuing pipeline")

        # 记录总耗时
        result.enhanced_data["step_timings"] = step_timings

        total_mutations = result.mutation_count
        logger.info(
            f"[DocMirror] Pipeline ◀ status={result.status} | "
            f"total_mutations={total_mutations}"
        )

        return result
