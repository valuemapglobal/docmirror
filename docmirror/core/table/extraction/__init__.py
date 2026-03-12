"""
table_extraction — Table extraction subpackage
================================================

Split from the original ``table_extraction.py`` (1,759 lines) into
6 focused modules:

- ``engine``              — Main entry point ``extract_tables_layered`` (6+1 layers)
- ``pipe_strategy``       — Layer 0.5: pipe-delimited table extraction
- ``pdfplumber_strategy`` — Layer 1: header recovery when pdfplumber fails
- ``classifier``          — Pre-classification + confidence scoring + validation gates
- ``char_strategy``       — Layer 2: character-level extraction strategies
- ``utils``               — Shared low-level utility functions

Public API (backward-compatible)::

    from docmirror.core.table_extraction import extract_tables_layered
"""

# ── Public API re-exports ──
from .engine import extract_tables_layered
from .classifier import (
    TABLE_SETTINGS,
    TABLE_SETTINGS_LINES,
    get_last_layer_timings,
    _quick_classify,
    _compute_table_confidence,
    _tables_look_valid,
    _cell_is_stuffed,
)
from .pipe_strategy import _extract_by_pipe_delimited, _merge_pipe_continuation_rows
from .pdfplumber_strategy import _recover_header_from_zone
from .char_strategy import (
    _extract_by_hline_columns,
    _extract_by_rect_columns,
    detect_columns_by_header_anchors,
    detect_columns_by_whitespace_projection,
    detect_columns_by_clustering,
    detect_columns_by_word_anchors,
    detect_columns_by_data_voting,
)
from .utils import (
    _group_chars_into_rows,
    _cluster_x_positions,
    _assign_chars_to_columns,
    _chars_to_text,
)

__all__ = [
    "extract_tables_layered",
    "get_last_layer_timings",
    "TABLE_SETTINGS",
    "TABLE_SETTINGS_LINES",
]
