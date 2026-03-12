"""
Pipeline Registry — Format-specific middleware pipeline composition.
=====================================================================

Defines which middlewares run (and in what order) for each combination
of file format and enhancement mode. This is the central place to
control the processing pipeline without modifying orchestrator logic.

Structure:
    FORMAT_PIPELINES[file_type][enhance_mode] → list of middleware names

Enhancement modes:
    - ``raw``:      No middlewares — returns the raw extraction result.
    - ``standard``: Default production pipeline with scene detection,
                    entity extraction, institution detection, and validation.

The wildcard entry ``"*"`` serves as a fallback for unregistered formats.
If the requested enhance_mode is not found within a format's config,
it falls back to the ``"standard"`` pipeline for that format.

To add a new format:
    Add a new key to FORMAT_PIPELINES with its middleware lists.

To add a new middleware to an existing pipeline:
    Append the middleware class name to the appropriate list(s).

Usage::

    from docmirror.configs.pipeline_registry import get_pipeline_config

    middlewares = get_pipeline_config("pdf", "full")
    # ['SceneDetector', 'EntityExtractor', 'InstitutionDetector',
    #  'Validator']
"""
from __future__ import annotations

from typing import Dict, List


# File format → { enhance_mode → ordered list of middleware class names }
FORMAT_PIPELINES: Dict[str, Dict[str, List[str]]] = {
    "pdf": {
        "raw": [],
        "standard": [
            "SceneDetector",
            "EntityExtractor",
            "InstitutionDetector",
            "Validator",
        ],
        "full": [
            "SceneDetector",
            "EntityExtractor",
            "InstitutionDetector",
            "Validator",
        ],
    },
    "image": {
        "raw": [],
        "standard": ["LanguageDetector", "GenericEntityExtractor"],
    },
    "excel": {
        "raw": [],
        "standard": ["GenericEntityExtractor"],
    },
    "word": {
        "raw": [],
        "standard": ["LanguageDetector", "GenericEntityExtractor"],
    },
    # Wildcard fallback — used for any file type not explicitly registered
    "*": {
        "raw": [],
        "standard": ["LanguageDetector"],
    },
}


def get_pipeline_config(file_type: str, enhance_mode: str = "standard") -> List[str]:
    """
    Get the ordered middleware list for a given format and enhancement mode.

    Lookup priority:
        1. Exact match: FORMAT_PIPELINES[file_type][enhance_mode]
        2. Mode fallback: FORMAT_PIPELINES[file_type]["standard"]
        3. Format fallback: FORMAT_PIPELINES["*"][enhance_mode]
        4. Double fallback: FORMAT_PIPELINES["*"]["standard"]

    Args:
        file_type:    File format identifier (e.g., "pdf", "image", "excel").
        enhance_mode: Enhancement level (e.g., "raw", "standard", "full").

    Returns:
        Ordered list of middleware class names to execute in the pipeline.
    """
    fmt_config = FORMAT_PIPELINES.get(file_type, FORMAT_PIPELINES.get("*", {}))
    return fmt_config.get(enhance_mode, fmt_config.get("standard", []))
