"""
RapidTable Singleton Engine — Thread-safe table structure recognition.

Usage::

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
    """Thread-safe RapidTable singleton.

    The model is lazily loaded on first invocation (~1–3 s); subsequent
    calls reuse the same instance.
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
        """Lazily load the engine.  Returns whether it is available."""
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
        """Run table structure recognition.

        Args:
            img: numpy ndarray (RGB/BGR), PIL Image, or image path.

        Returns:
            ``RapidTableOutput``, or ``None`` if unavailable / on failure.
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
    """Return the global RapidTable engine singleton."""
    return RapidTableEngine.get_instance()
