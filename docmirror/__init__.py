"""
DocMirror: Universal Document Parsing Engine

Directory structure:
- core/: Core extraction engines (CoreExtractor, LayoutAnalysis, TableExtraction)
- models/: Data models (BaseResult, EnhancedResult, PerceptionResult)
- middlewares/: Middleware pipeline (SceneDetector, EntityExtractor, Validator, ...)
- configs/: Configuration (settings, pipeline_registry, institution_registry)
- framework/: Pipeline orchestration (dispatcher, orchestrator, cache)
- adapters/: Format adapters (PDF, Image, Office, Email, Web)
- plugins/: Domain plugins (bank_statement, ...)

Single public entry point: perceive_document()
"""

__version__ = "0.2.0"

import logging

from docmirror.core.factory import perceive_document, PerceptionFactory
from docmirror.models.document_types import DocumentType
from docmirror.models.perception_result import PerceptionResult
from docmirror.models.domain_models import DomainData
from docmirror.framework.dispatcher import ParserDispatcher
from docmirror.framework.dispatcher import ParserDispatcher as DocumentProcessingOrchestrator  # compat
from docmirror.framework.base import ParserOutput
from docmirror.framework.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

# backward-compat alias — callers importing PerceptionResponse get ParserOutput
PerceptionResponse = ParserOutput


__all__ = [
    "perceive_document",
    "PerceptionFactory",
    "PerceptionResult",
    "PerceptionResponse",
    "DocumentType",
    "DomainData",
    "ParserDispatcher",
    "DocumentProcessingOrchestrator",
    "ParserOutput",
    "Orchestrator",
]
