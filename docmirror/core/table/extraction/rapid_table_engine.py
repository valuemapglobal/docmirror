"""RapidTable 单例引擎 — 线程安全的表格结构识别。

使用方式::

    from .rapid_table_engine import get_rapid_table_engine
    engine = get_rapid_table_engine()
    result = engine(img_np)  # -> RapidTableOutput
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from rapid_table import RapidTable, RapidTableInput, RapidTableOutput
    HAS_RAPID_TABLE = True
except ImportError:
    HAS_RAPID_TABLE = False


class RapidTableEngine:
    """线程安全的 RapidTable 单例。

    首次调用时懒加载模型 (~1-3s), 后续调用复用同一实例。
    """

    _instance: Optional["RapidTableEngine"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._engine = None
        self._available = HAS_RAPID_TABLE

    @classmethod
    def get_instance(cls) -> "RapidTableEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_engine(self) -> bool:
        """懒加载引擎, 返回是否可用。"""
        if not self._available:
            return False
        if self._engine is not None:
            return True
        with self._lock:
            if self._engine is not None:
                return True
            try:
                logger.info("[RapidTable] Initializing model...")
                self._engine = RapidTable()
                logger.info("[RapidTable] Model loaded.")
                return True
            except Exception as e:
                logger.warning(f"[RapidTable] Init failed: {e}")
                self._available = False
                return False

    def __call__(self, img) -> Optional["RapidTableOutput"]:
        """运行表格结构识别。

        Args:
            img: numpy ndarray (RGB/BGR), PIL Image, 或图片路径。

        Returns:
            RapidTableOutput 或 None (不可用/失败时)。
        """
        if not self._ensure_engine():
            return None
        try:
            return self._engine(img)
        except Exception as e:
            logger.debug(f"[RapidTable] Inference error: {e}")
            return None

    @property
    def is_available(self) -> bool:
        return self._available


def get_rapid_table_engine() -> RapidTableEngine:
    """获取全局 RapidTable 引擎单例。"""
    return RapidTableEngine.get_instance()
