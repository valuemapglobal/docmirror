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

def _extract_by_pipe_delimited(
    page_plum,
) -> Optional[List[List[str]]]:
    """管道分隔符表格提取 (Layer 0.5) — Grid Consistency 算法。

    专门处理大机 mainframe 生成的 ASCII 画线 PDF:
    - 垂直分隔: | │ 等 PIPE_CHARS
    - 水平分隔: ─ ━ 等 HLINE_CHARS
    - PDF 无任何绘图原语 (lines/rects = 0)

    安全门控:
      G1: pdfplumber lines = 0 且 rects = 0 (仅在无绘图原语时启用)
      G2: ≥3 个一致的垂直网格线 (出现率 ≥ 70% 的数据行)
      G3: ≥3 行数据行 (排除水平分隔线后)
      G4: 每条网格线的 x 坐标标准差 ≤ 3pt
    """
    # ── G1: 只在无 PDF 绘图原语时启用 ──
    pdf_lines = page_plum.lines or []
    pdf_rects = page_plum.rects or []
    if pdf_lines or pdf_rects:
        return None

    chars = page_plum.chars
    if not chars:
        return None

    # ── Step 1: 按 y 坐标分行 ──
    y_groups: Dict[int, List[dict]] = defaultdict(list)
    for c in chars:
        y_key = round(c["top"] / 3) * 3
        y_groups[y_key].append(c)

    if len(y_groups) < 3:
        return None

    # ── Step 2: 分类行 — 数据行 vs 水平分隔行 ──
    data_rows_ys: List[int] = []      # 含 pipe 的数据行的 y_key
    hline_rows_ys: List[int] = []     # 纯水平线行
    all_pipe_x_by_row: Dict[int, List[float]] = {}  # y_key → pipe x 坐标列表

    for y_key in sorted(y_groups.keys()):
        row_chars = y_groups[y_key]
        row_text = "".join(c["text"] for c in sorted(row_chars, key=lambda c: c["x0"]))

        # 检测水平分隔行: 大部分字符是 HLINE_CHARS 或 PIPE_CHARS 或空格
        non_space = [c for c in row_text if c.strip()]
        if non_space:
            border_ratio = sum(1 for c in non_space if c in _ALL_BORDER_CHARS) / len(non_space)
            if border_ratio >= 0.8:
                hline_rows_ys.append(y_key)
                continue

        # 收集 pipe 字符的 x 坐标
        pipe_xs = [
            round(c["x0"], 1)
            for c in row_chars
            if c.get("text") in PIPE_CHARS
        ]
        if len(pipe_xs) >= 2:
            data_rows_ys.append(y_key)
            all_pipe_x_by_row[y_key] = sorted(pipe_xs)

    # ── G3: 至少 3 行数据行 ──
    if len(data_rows_ys) < 3:
        return None

    # ── Step 3: 对 pipe x 坐标做聚类, 找垂直网格线 ──
    # 收集所有 pipe x 坐标
    all_pipe_xs: List[float] = []
    for xs in all_pipe_x_by_row.values():
        all_pipe_xs.extend(xs)

    if not all_pipe_xs:
        return None

    # 聚类: snap 到 5pt 网格
    SNAP = 5.0
    x_clusters: Dict[float, List[float]] = defaultdict(list)
    for x in sorted(all_pipe_xs):
        snapped = round(x / SNAP) * SNAP
        x_clusters[snapped].append(x)

    # 合并相近的聚类 (间距 < 8pt)
    sorted_centers = sorted(x_clusters.keys())
    merged_clusters: List[List[float]] = []
    for center in sorted_centers:
        if merged_clusters and center - sum(merged_clusters[-1]) / len(merged_clusters[-1]) < 8:
            merged_clusters[-1].extend(x_clusters[center])
        else:
            merged_clusters.append(list(x_clusters[center]))

    # ── G2 + G4: 检查网格一致性 ──
    n_data_rows = len(data_rows_ys)
    consistent_grid_lines: List[float] = []  # 通过一致性检查的网格线 x 中心

    for cluster in merged_clusters:
        # 出现率: 这个 x 聚类出现在多少行中
        rows_with_this_x = set()
        for y_key, pipe_xs in all_pipe_x_by_row.items():
            if any(abs(px - sum(cluster) / len(cluster)) < 8 for px in pipe_xs):
                rows_with_this_x.add(y_key)

        presence_ratio = len(rows_with_this_x) / n_data_rows
        if presence_ratio < 0.7:
            continue

        # G4: x 坐标标准差
        mean_x = sum(cluster) / len(cluster)
        variance = sum((x - mean_x) ** 2 for x in cluster) / len(cluster)
        std_x = variance ** 0.5
        if std_x > 3.0:
            continue

        consistent_grid_lines.append(mean_x)

    # G2: 至少 3 条一致的垂直网格线
    if len(consistent_grid_lines) < 3:
        return None

    consistent_grid_lines.sort()
    n_cols = len(consistent_grid_lines) - 1  # pipe 之间的列数
    if n_cols < 2:
        return None

    logger.info(
        f"[v2] pipe_delimited: detected {len(consistent_grid_lines)} grid lines, "
        f"{n_cols} cols, {n_data_rows} data rows, "
        f"{len(hline_rows_ys)} hline rows"
    )

    # ── Step 4: 用网格线分列, 构建二维表格 ──
    # 列区间: (left_pipe_x, right_pipe_x)
    col_intervals = [
        (consistent_grid_lines[i], consistent_grid_lines[i + 1])
        for i in range(n_cols)
    ]

    table: List[List[str]] = []
    for y_key in sorted(data_rows_ys):
        row_chars = sorted(y_groups[y_key], key=lambda c: c["x0"])
        # 过滤掉 pipe 字符本身
        content_chars = [c for c in row_chars if c.get("text") not in PIPE_CHARS]

        cells = [""] * n_cols
        for c in content_chars:
            cx = c["x0"]
            # 找到所属的列
            assigned = False
            for col_idx, (left, right) in enumerate(col_intervals):
                if left - 3 <= cx < right + 3:
                    cells[col_idx] += c["text"]
                    assigned = True
                    break
            if not assigned:
                # 超出网格范围: 分配到最近的列
                distances = [abs(cx - (l + r) / 2) for l, r in col_intervals]
                nearest = distances.index(min(distances))
                cells[nearest] += c["text"]

        table.append([cell.strip() for cell in cells])

    if len(table) < 3:
        return None

    # ── Step 5: 合并续行 (mainframe 一条记录可能跨多行) ──
    table = _merge_pipe_continuation_rows(table)

    logger.info(
        f"[v2] pipe_delimited: extracted {len(table)} rows x {n_cols} cols"
    )
    return table


def _merge_pipe_continuation_rows(table: List[List[str]]) -> List[List[str]]:
    """合并 pipe 表格中的续行。

    Mainframe 格式中, 一条记录可能拆成多行:
      Row N:   | 1  |251209|251209|实时缴税|    |19077378/2025120964867670 重庆...|   894.34|         |   9,143.21|...
      Row N+1: |    |      |      |        |    |限公司长沙分公司 91430100...     |         |         |           |...

    规则: 如果首列 (序号列) 为空, 则视为上一行的续行, 将内容追加到上一行。
    """
    if not table or len(table) < 2:
        return table

    merged: List[List[str]] = [table[0]]
    for row in table[1:]:
        first_cell = row[0].strip() if row else ""
        # 续行: 首列 (序号) 为空, 且行中有非空内容
        has_content = any(c.strip() for c in row[1:])
        if not first_cell and has_content and merged:
            # 追加到上一行
            prev = merged[-1]
            for i in range(len(row)):
                if i < len(prev):
                    cell_text = row[i].strip()
                    if cell_text:
                        if prev[i].strip():
                            prev[i] = prev[i].strip() + cell_text
                        else:
                            prev[i] = cell_text
        else:
            merged.append(row)

    return merged

