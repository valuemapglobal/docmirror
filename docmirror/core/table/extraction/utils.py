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



# ── 共享工具函数 ──

def _adaptive_row_tolerance(chars: List[Dict]) -> float:
    """F-1: 计算行分组的自适应 y 容差。

    基于字符中位高度动态计算，防止:
      - 小字号 PDF 的 3pt 固定容差导致多行合并
      - 大字号 PDF 的 3pt 容差导致同行字符被拆分

    Returns:
        自适应容差值 (通常 1.5 ~ 5.0pt)
    """
    if not chars or len(chars) < 5:
        return 3.0

    heights = [c["bottom"] - c["top"] for c in chars
               if c.get("bottom", 0) > c.get("top", 0) and c["bottom"] - c["top"] < 30]
    if not heights:
        return 3.0

    heights.sort()
    median_h = heights[len(heights) // 2]
    # 容差 = 中位字符高度 × 0.6, 限制在 [1.5, 5.0] 范围
    tol = max(1.5, min(5.0, median_h * 0.6))
    return tol


def _group_chars_into_rows(
    chars: List[Dict], y_tolerance: float = 3.0
) -> List[Tuple[float, List[Dict]]]:
    """按 y 坐标将字符分组到行。

    F-1 增强: 当 y_tolerance <= 0 时自动使用 _adaptive_row_tolerance。
    """
    if not chars:
        return []

    # F-1: 自适应容差
    if y_tolerance <= 0:
        y_tolerance = _adaptive_row_tolerance(chars)

    sorted_chars = sorted(chars, key=lambda c: c["top"])
    rows: List[Tuple[float, List[Dict]]] = []
    current_row: List[Dict] = [sorted_chars[0]]
    current_y = sorted_chars[0]["top"]

    for c in sorted_chars[1:]:
        if abs(c["top"] - current_y) <= y_tolerance:
            current_row.append(c)
        else:
            y_mid = sum(ch["top"] for ch in current_row) / len(current_row)
            rows.append((y_mid, sorted(current_row, key=lambda x: x["x0"])))
            current_row = [c]
            current_y = c["top"]

    if current_row:
        y_mid = sum(ch["top"] for ch in current_row) / len(current_row)
        rows.append((y_mid, sorted(current_row, key=lambda x: x["x0"])))

    return rows


def _cluster_x_positions(
    x_coords: List[float], gap_multiplier: float = 2.0, min_col_width: float = 10.0
) -> List[Tuple[float, float]]:
    """x 坐标聚类: 找列边界。

    优化3: 使用 IQR (Tukey Fence) 自适应阈值替代 median × multiplier,
    对窄间距列更鲁棒。当 gap 数量不足 (< 4) 时退回原逻辑。
    """
    if not x_coords:
        return []

    sorted_x = sorted(set(round(x, 1) for x in x_coords))
    if len(sorted_x) < 2:
        return [(sorted_x[0], sorted_x[0] + 50)]

    gaps = [sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x) - 1)]
    non_zero_gaps = sorted(g for g in gaps if g > 0.5)

    if not non_zero_gaps:
        return [(sorted_x[0], sorted_x[-1])]

    # ── 优化3: 自适应阈值 (Natural Break) ──
    # 列间 gap 通常呈双峰分布 (小 gap = 同列字符间距, 大 gap = 列间距)
    # 找 sorted gaps 中最大的跳变点, 在该点设置阈值
    median_gap = non_zero_gaps[len(non_zero_gaps) // 2]

    if len(non_zero_gaps) >= 4:
        # 找排序后 gaps 中最大的相邻跳变
        max_jump = 0
        jump_idx = -1
        for j in range(len(non_zero_gaps) - 1):
            jump = non_zero_gaps[j + 1] - non_zero_gaps[j]
            if jump > max_jump:
                max_jump = jump
                jump_idx = j

        if max_jump > median_gap * 2 and jump_idx >= 0:
            # 明显的双峰分布: 阈值 = 跳变点中位
            threshold = (non_zero_gaps[jump_idx] + non_zero_gaps[jump_idx + 1]) / 2
        else:
            # 连续分布: 退回 median × multiplier
            threshold = median_gap * gap_multiplier
    else:
        # 数据点太少, 退回原逻辑
        threshold = median_gap * gap_multiplier

    col_bounds: List[Tuple[float, float]] = []
    col_start = sorted_x[0]

    for i, gap in enumerate(gaps):
        if gap > threshold:
            col_end = sorted_x[i]
            if col_end - col_start >= min_col_width:
                col_bounds.append((col_start, col_end))
            col_start = sorted_x[i + 1]

    col_end = sorted_x[-1]
    if col_end - col_start >= min_col_width:
        col_bounds.append((col_start, col_end))

    return col_bounds


def _assign_chars_to_columns(
    row_chars: List[Dict], col_bounds: List[Tuple[float, float]]
) -> List[str]:
    """将一行中的字符按列分割线归箱。

    分割线 = 相邻列之间的中点。
    比固定容差更精确 (不受列宽不均影响),
    比最近中心更稳定 (不受宽窄列不对称吸引)。
    """
    if not col_bounds:
        return []

    cells = ["" for _ in col_bounds]

    # 计算相邻列之间的分割线
    dividers = [col_bounds[0][0] - 10]  # 左边界
    for i in range(len(col_bounds) - 1):
        mid = (col_bounds[i][1] + col_bounds[i + 1][0]) / 2
        dividers.append(mid)
    dividers.append(col_bounds[-1][1] + 10)  # 右边界

    # 将相邻字符合并成 word (避免单词从中间被切断)
    sorted_chars = sorted(row_chars, key=lambda x: x["x0"])
    words = []
    curr_word = None
    def is_cjk(ch):
        return _is_cjk_char(ch)

    for c in sorted_chars:
        if not str(c.get("text", "")).strip():
            continue
        if not curr_word:
            curr_word = {"x0": c["x0"], "x1": c.get("x1", c["x0"]), "text": c["text"]}
        else:
            gap = c["x0"] - curr_word["x1"]
            is_prev_cjk = _is_cjk_char(curr_word["text"][-1]) if curr_word["text"] else False
            is_curr_cjk = _is_cjk_char(c["text"])
            threshold = 5.0 if (is_prev_cjk or is_curr_cjk) else 2.5
            
            if gap < threshold:
                curr_word["x1"] = max(curr_word["x1"], c.get("x1", c["x0"]))
                curr_word["text"] += c["text"]
            else:
                words.append(curr_word)
                curr_word = {"x0": c["x0"], "x1": c.get("x1", c["x0"]), "text": c["text"]}
    if curr_word:
        words.append(curr_word)

    for w in words:
        char_x = (w["x0"] + w["x1"]) / 2
        # 二分查找所属列
        col_idx = len(col_bounds) - 1
        for i in range(len(dividers) - 1):
            if dividers[i] <= char_x < dividers[i + 1]:
                col_idx = i
                break
        if 0 <= col_idx < len(cells):
            if cells[col_idx]:
                existing = cells[col_idx].strip()
                val = w.get("text", "").strip()
                if existing and val:
                    cells[col_idx] = _smart_join(existing, val)
                else:
                    cells[col_idx] += w.get("text", "")
            else:
                cells[col_idx] += w.get("text", "")

    return [cell.strip() for cell in cells]


def _chars_to_text(chars: List[Dict]) -> str:
    """将字符列表合并为文本。"""
    if not chars:
        return ""
    sorted_c = sorted(chars, key=lambda c: c["x0"])
    parts = [sorted_c[0].get("text", "")]
    for i in range(1, len(sorted_c)):
        gap = sorted_c[i]["x0"] - sorted_c[i-1].get("x1", sorted_c[i-1]["x0"])
        if gap > 3:
            parts.append(" ")
        parts.append(sorted_c[i].get("text", ""))
    return "".join(parts)

