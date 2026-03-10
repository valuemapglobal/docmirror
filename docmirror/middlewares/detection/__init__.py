"""Detection middlewares – scene, language, institution."""

from .scene_detector import SceneDetector
from .language_detector import LanguageDetector
from .institution_detector import InstitutionDetector

__all__ = ["SceneDetector", "LanguageDetector", "InstitutionDetector"]
