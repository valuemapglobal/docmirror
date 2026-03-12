"""
MultiModal Data Model Layer.

Core Trinity:
    - domain.py:   BaseResult (frozen) \u2014 Immutable Extraction Result
    - mutation.py: Mutation            \u2014 Data Lineage Tracker
    - enhanced.py: EnhancedResult      \u2014 Final Enhanced Output
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
from docmirror._compat import register_shims as _register_shims

_register_shims({
    f"{__name__}.domain":             f"{__name__}.entities.domain",
    f"{__name__}.domain_models":      f"{__name__}.entities.domain_models",
    f"{__name__}.document_types":     f"{__name__}.entities.document_types",
    f"{__name__}.enhanced":           f"{__name__}.entities.enhanced",
    f"{__name__}.perception_result":  f"{__name__}.entities.perception_result",
    f"{__name__}.mutation":           f"{__name__}.tracking.mutation",
    f"{__name__}.builder":            f"{__name__}.construction.builder",
})

