"""Auto-split from table_extraction.py"""

from __future__ import annotations

import concurrent.futures
import contextvars
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...utils.text_utils import _is_cjk_char, _smart_join, normalize_table
from ...utils.vocabulary import (
    KNOWN_HEADER_WORDS,
    PIPE_CHARS,
    HLINE_CHARS,
    _ALL_BORDER_CHARS,
    _is_header_row,
    _normalize_for_vocab,
    _score_header_by_vocabulary,
    _RE_IS_DATE,
    _RE_IS_AMOUNT,
)

logger = logging.getLogger(__name__)

TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "lines",
    "snap_x_tolerance": 5,
    "join_x_tolerance": 5,
    "snap_y_tolerance": 3,
}

TABLE_SETTINGS_LINES = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
}

# ── contextvars: 线程/异步安全的 layer timings ──
_layer_timings_var: contextvars.ContextVar[Dict[str, float]] = contextvars.ContextVar(
    'layer_timings', default={}
)


def get_last_layer_timings() -> Dict[str, float]:
    """返回当前上下文中最近一次 extract_tables_layered 的各层耗时 (ms)。"""
    return dict(_layer_timings_var.get({}))


def _quick_classify(work_page) -> str:
    """表格预分类: 根据快速特征建议起始 Layer, 跳过不太可能命中的层。

    返回建议的起始 Layer 标签:
      - 'pipe'   : 管道符 >= 10 → 直接从 L0.5 开始 (默认)
      - 'lines'  : 有 >= 3 条线 → 从 L1 开始 (默认)
      - 'text'   : 有线但不多, x 分散 → 从 L1b 开始, 跳过 L1
      - 'char'   : 无线条, 无管道 → 直接跳到 L2 char-level
    """
    chars = work_page.chars or []
    lines = work_page.lines or []

    # 特征1: 管道符数量
    pipe_count = sum(1 for c in chars if c.get("text") in PIPE_CHARS)
    if pipe_count >= 10:
        return "pipe"

    # 特征2: 线条数量
    h_lines = [l for l in lines if abs(l.get("top", 0) - l.get("bottom", 0)) < 1]
    v_lines = [l for l in lines if abs(l.get("x0", 0) - l.get("x1", 0)) < 1]
    total_lines = len(h_lines) + len(v_lines)

    if total_lines >= 6:  # 有足够的线条 -> 走线条路径
        return "lines"

    # 特征3: x 坐标分散度 (区分文本对齐表 vs 纯文本)
    if chars:
        x_positions = set(round(c["x0"] / 10) * 10 for c in chars)
        if len(x_positions) >= 5:  # x 分散 -> 可能是无线表格
            if total_lines >= 3:    # 有一些横线 -> text 策略可能更好
                return "text"
            return "char"          # 无线条 -> 直接 char-level

    return "pipe"  # 默认: 从头开始


def _compute_table_confidence(
    tables: List[List[List[str]]],
    layer: str,
) -> float:
    """计算表格提取结果的置信度 (0.0 ~ 1.0)。

    综合考量:
      - vocab_score: 表头匹配词表的命中数 (权重最高)
      - row_count: 行数越多越可信
      - col_consistency: 各行列数一致性
      - layer_bonus: 前置层天然置信度更高
    """
    if not tables or not tables[0]:
        return 0.0

    tbl = tables[0]  # 主表
    if len(tbl) < 1:
        return 0.0

    # 1. vocab_score (0~1, 线性映射: 0->0, 3->0.6, 5+->1.0)
    header = tbl[0]
    vocab = _score_header_by_vocabulary(header)
    vocab_norm = min(1.0, vocab / 5.0)

    # 2. row_count (0~1, 对数映射: 2->0.3, 10->0.7, 50+->1.0)
    row_count = len(tbl)
    row_norm = min(1.0, math.log2(max(2, row_count)) / math.log2(50))

    # 3. col_consistency (0~1, 各行列数一致的比例)
    if len(tbl) >= 2:
        expected_cols = len(tbl[0])
        consistent = sum(1 for row in tbl if len(row) == expected_cols)
        col_norm = consistent / len(tbl)
    else:
        col_norm = 0.5

    # 4. layer_bonus (前置层天然更可信)
    _LAYER_BONUS = {
        "pipe_delimited": 0.15, "lines": 0.15, "hline_columns": 0.10,
        "rect_columns": 0.10, "text": 0.10, "docling_tableformer": 0.10,
        "rapid_table": 0.12,  # 视觉模型: 高于 char-level 策略
        "header_anchors": 0.05, "word_anchors": 0.05,
        "data_voting": 0.05, "whitespace_projection": 0.05,
        "x_clustering": 0.0, "fallback": -0.10,
    }
    bonus = _LAYER_BONUS.get(layer, 0.0)

    # 加权求和
    confidence = (vocab_norm * 0.40 + row_norm * 0.20 + col_norm * 0.25) + bonus
    return round(max(0.0, min(1.0, confidence)), 3)

def _cell_is_stuffed(cell: str) -> bool:
    """检测单元格是否塞入了多行数据 (行被合并的症状)。

    正常单元格不会同时包含多个日期或多个金额。
    若检测到以下任一情况, 说明上层把多条记录压进了同一格子:

      - 单格内 ≥2 个日期模式  (如 '2025-09-21...2025-10-27...')
      - 单格内 ≥4 个金额        (如 '3000000.00 600000.00 160000.00 3760000.00')

    注意:
      - 日期用 (?:19|20)\\d{6} 而非 \\d{8}, 避免将电话号码 (如 13883435811)
        的连续8位误识别为日期。
      - 金额用词边界正则, 避免将大数字 (如 3000000.00) 的子串 (000.00) 重复计数。
      - 不要求 '\\n': hline_columns 层用空格拼接单元格内文字, 没有换行符。
    """
    if not cell or len(cell) < 10:
        return False
        
    # Ignore preamble common text containing dates
    if "至" in cell and re.search(r"时间|期限|期间", cell):
        return False

    # 条件1: 单格内出现 ≥2 个日期
    # 注: (?:19|20)\d{6} 仅匹配以 19/20 开头的8位日期, 防止电话号码误匹配
    dates = re.findall(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:19|20)\d{6}', cell)
    if len(dates) >= 2:
        return True
    # 条件2: 单格内出现 ≥4 个金额 (词边界匹配, 防止子串重复计数)
    amounts = re.findall(r'(?<!\d)\d[\d,]*\.\d{2}(?!\d)', cell)
    if len(amounts) >= 4:
        return True
    return False


def _tables_look_valid(tables: list, min_rows: int = 2, has_borders: bool = False) -> bool:
    """检查表格是否有效 (含行密度和单格塞行质量检测)。

    Args:
        has_borders: True 表示页面有纵线边框, 跳过 stuffed cell 检测。
                     有边框的表格列结构由物理线条保证, 单格内多行内容是合法的。
    """
    if not tables:
        return False
    for tbl in tables:
        if tbl and len(tbl) >= min_rows:
            col_count = len(tbl[0])
            if 2 <= col_count <= 30:
                # ── 检测1: 平均行字符数异常 (所有行合并成一行时字符暴增) ──
                total_chars = sum(
                    len(str(c or "")) for row in tbl for c in row
                )
                avg_chars_per_row = total_chars / len(tbl)
                if avg_chars_per_row > 500:
                    logger.warning(
                        f"[v2] table rejected: avg {avg_chars_per_row:.0f} "
                        f"chars/row > 500 → fallback to char-level"
                    )
                    return False
                # ── F-1: 增强采样检测 (首4 + 中2 + 末2) ──
                # 有边框表格跳过 stuffed cell 检测 (边框保证列结构正确)
                if not has_borders:
                    n = len(tbl)
                    sample_indices = list(range(min(4, n)))
                    if n > 8:
                        mid = n // 2
                        sample_indices.extend([mid - 1, mid])
                    if n > 4:
                        sample_indices.extend([n - 2, n - 1])
                    sample_indices = sorted(set(i for i in sample_indices if 0 <= i < n))

                    for idx in sample_indices:
                        for cell in tbl[idx]:
                            if _cell_is_stuffed(str(cell or "")):
                                logger.warning(
                                    f"[v2] table rejected: stuffed cell at row {idx} "
                                    f"(cell={str(cell or '')[:40]!r}…) → fallback"
                                )
                                return False
                return True
    return False

