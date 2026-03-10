"""MultiModal 中间件管线。"""

from .base import BaseMiddleware, MiddlewarePipeline
from .detection.scene_detector import SceneDetector
from .detection.institution_detector import InstitutionDetector
from .detection.language_detector import LanguageDetector
from .extraction.entity_extractor import EntityExtractor
from .extraction.generic_entity_extractor import GenericEntityExtractor
from .alignment.column_mapper import ColumnMapper
from .alignment.repairer import Repairer
from .validation.validator import Validator
from .validation.entropy_monitor import EntropyMonitor
from .validation.mutation_analyzer import MutationAnalyzer

__all__ = [
    "BaseMiddleware", "MiddlewarePipeline",
    "SceneDetector", "InstitutionDetector", "LanguageDetector",
    "EntityExtractor", "GenericEntityExtractor",
    "ColumnMapper", "Repairer",
    "Validator", "EntropyMonitor", "MutationAnalyzer",
]

# ── Backward-compatible shims ──────────────────────────────────────────
# External code may still use  ``from ..middlewares.scene_detector import ...``
# These module-level re-exports keep those import paths working.
import importlib as _il, sys as _sys

_SHIM_MAP = {
    f"{__name__}.scene_detector":       f"{__name__}.detection.scene_detector",
    f"{__name__}.language_detector":    f"{__name__}.detection.language_detector",
    f"{__name__}.institution_detector": f"{__name__}.detection.institution_detector",
    f"{__name__}.entity_extractor":     f"{__name__}.extraction.entity_extractor",
    f"{__name__}.generic_entity_extractor": f"{__name__}.extraction.generic_entity_extractor",
    f"{__name__}.column_mapper":        f"{__name__}.alignment.column_mapper",
    f"{__name__}.header_alignment":     f"{__name__}.alignment.header_alignment",
    f"{__name__}.amount_splitter":      f"{__name__}.alignment.amount_splitter",
    f"{__name__}.repairer":             f"{__name__}.alignment.repairer",
    f"{__name__}.validator":            f"{__name__}.validation.validator",
    f"{__name__}.entropy_monitor":      f"{__name__}.validation.entropy_monitor",
    f"{__name__}.mutation_analyzer":    f"{__name__}.validation.mutation_analyzer",
}
for _old, _new in _SHIM_MAP.items():
    if _old not in _sys.modules:
        _sys.modules[_old] = _il.import_module(_new)
