"""
MultiModal 数据模型层。

核心三件套:
    - domain.py:   BaseResult (frozen) — 不可变提取结果
    - mutation.py: Mutation            — 变换血缘记录
    - enhanced.py: EnhancedResult      — 增强后的最终结果
"""

from .entities.domain import Style, TextSpan, Block, PageLayout, BaseResult
from .tracking.mutation import Mutation
from .entities.enhanced import EnhancedResult

__all__ = [
    "Style", "TextSpan", "Block", "PageLayout", "BaseResult",
    "Mutation", "EnhancedResult",
]

# ── Backward-compatible shims ──────────────────────────────────────────
# Allow  ``from ...models.domain import ...``  etc. to keep working.
import importlib as _il, sys as _sys

_SHIM_MAP = {
    f"{__name__}.domain":             f"{__name__}.entities.domain",
    f"{__name__}.domain_models":      f"{__name__}.entities.domain_models",
    f"{__name__}.document_types":     f"{__name__}.entities.document_types",
    f"{__name__}.enhanced":           f"{__name__}.entities.enhanced",
    f"{__name__}.perception_result":  f"{__name__}.entities.perception_result",
    f"{__name__}.mutation":           f"{__name__}.tracking.mutation",
    f"{__name__}.builder":            f"{__name__}.construction.builder",
}
for _old, _new in _SHIM_MAP.items():
    if _old not in _sys.modules:
        _sys.modules[_old] = _il.import_module(_new)
