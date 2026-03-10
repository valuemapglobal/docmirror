"""
table_extraction — 表格提取子包
================================

从原 ``table_extraction.py`` (1,759 行) 拆分为 6 个模块:

- ``engine``              — 主入口 ``extract_tables_layered`` (6+1 层)
- ``pipe_strategy``       — Layer 0.5 管道分隔符提取
- ``pdfplumber_strategy`` — Layer 1 header recovery
- ``classifier``          — 预分类 + 置信度 + 验证门控
- ``char_strategy``       — Layer 2 字符级策略集
- ``utils``               — 共享低层工具函数

Public API (向后兼容):
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
