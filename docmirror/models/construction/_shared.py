"""
Shared Utilities — Common helpers for PerceptionResult construction.
===================================================================

Functions used by both ``framework/base.py`` (the legacy ``ParserOutput``
converter) and ``models/construction/builder.py`` (the canonical builder)
to avoid code duplication.
"""
from __future__ import annotations


from typing import Any, Dict, Optional

from docmirror.models.entities.perception_result import Diagnostics


def build_diagnostics(meta: Dict[str, Any]) -> Optional[Diagnostics]:
    """Extract pipeline diagnostics from a metadata dict.

    Returns ``None`` if the metadata contains no ``_diagnostics`` key.
    """
    diag_data = meta.get("_diagnostics", {})
    if not diag_data:
        return None
    return Diagnostics(
        extraction_method=diag_data.get("extraction_method", ""),
        template_id=diag_data.get("template_id", ""),
        template_source=diag_data.get("template_source", ""),
        pages_processed=diag_data.get("pages_processed", 0),
        raw_rows_extracted=diag_data.get("raw_rows_extracted", 0),
        rows_after_cleaning=diag_data.get("rows_after_cleaning", 0),
        rows_final=diag_data.get("rows_final", 0),
        step_timing_ms=diag_data.get("step_timing_ms", {}),
        detected_columns=diag_data.get("detected_columns", []),
        missing_columns=diag_data.get("missing_columns", []),
        supplemented_columns=diag_data.get("supplemented_columns", []),
        failed_rows_sample=diag_data.get("failed_rows_sample", []),
        duplicate_rows_detected=diag_data.get("duplicate_rows_detected", 0),
        llm_usage=diag_data.get("llm_usage"),
    )
