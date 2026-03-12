"""MultiModal MiddlewarePipeline。"""

from .base import BaseMiddleware, MiddlewarePipeline
from .detection.scene_detector import SceneDetector
from .detection.institution_detector import InstitutionDetector
from .detection.language_detector import LanguageDetector
from .extraction.entity_extractor import EntityExtractor
from .extraction.generic_entity_extractor import GenericEntityExtractor
from .validation.validator import Validator
from .validation.mutation_analyzer import MutationAnalyzer

__all__ = [
    "BaseMiddleware", "MiddlewarePipeline",
    "SceneDetector", "InstitutionDetector", "LanguageDetector",
    "EntityExtractor", "GenericEntityExtractor",
    "Validator", "MutationAnalyzer",
]

# ── Backward-compatible shims ──────────────────────────────────────────
# External code may still use  ``from ..middlewares.scene_detector import ...``
# These module-level re-exports keep those import paths working.
from docmirror._compat import register_shims as _register_shims

_register_shims({
    f"{__name__}.scene_detector":       f"{__name__}.detection.scene_detector",
    f"{__name__}.language_detector":    f"{__name__}.detection.language_detector",
    f"{__name__}.institution_detector": f"{__name__}.detection.institution_detector",
    f"{__name__}.entity_extractor":     f"{__name__}.extraction.entity_extractor",
    f"{__name__}.generic_entity_extractor": f"{__name__}.extraction.generic_entity_extractor",
    f"{__name__}.header_alignment":     f"{__name__}.alignment.header_alignment",
    f"{__name__}.amount_splitter":      f"{__name__}.alignment.amount_splitter",
    f"{__name__}.validator":            f"{__name__}.validation.validator",
    f"{__name__}.mutation_analyzer":    f"{__name__}.validation.mutation_analyzer",
})

