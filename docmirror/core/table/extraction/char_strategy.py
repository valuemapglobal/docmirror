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
    _is_header_cell,
    _normalize_for_vocab,
    _score_header_by_vocabulary,
    _RE_IS_DATE,
    _RE_IS_AMOUNT,
)
from ...utils.watermark import is_watermark_char
from ..postprocess import _find_vocab_words_in_string

logger = logging.getLogger(__name__)


from .utils import _group_chars_into_rows, _cluster_x_positions, _assign_chars_to_columns, _chars_to_text

def _extract_by_hline_columns(page_plum) -> Optional[List[List[str]]]:
    """
    水平线列边界法 — 有横线无纵线时, 用横线 x 位置推断列边界。

    专门处理"只有水平分隔线、无垂直线"的 PDF (如招商银行交易流水)。
    横线的 x 断点定义了列边界, 数据行用 word y 坐标聚类。

    触发条件: ≥3 条水平线, 0 条垂直线。
    """
    lines = page_plum.lines or []
    if not lines:
        return None

    # 分类线条: 水平 vs 垂直
    h_lines = [l for l in lines if abs(l["top"] - l["bottom"]) < 1]
    v_lines = [l for l in lines if abs(l["x0"] - l["x1"]) < 1]

    # 触发条件: 有足够的水平线, 无垂直线
    if len(h_lines) < 3 or len(v_lines) > 0:
        return None

    # ── 从水平线 x 坐标提取列边界 ──
    raw_x = sorted(set(
        round(v, 1)
        for l in h_lines
        for v in [l["x0"], l["x1"]]
    ))
    # 合并相近的 x (snapping, 阈值 10pt — 避免微小间隙产生空列)
    x_positions = [raw_x[0]]
    for x in raw_x[1:]:
        if x - x_positions[-1] > 10:
            x_positions.append(x)

    if len(x_positions) < 3:
        return None  # 列太少, 不像表格

    # ── 确定列区间 ──
    col_count = len(x_positions) - 1
    intervals = [(x_positions[i], x_positions[i + 1]) for i in range(col_count)]

    # ── 确定表头区域 (取水平线的 y 范围) ──
    h_y_values = sorted(set(round(l["top"], 1) for l in h_lines))
    # 表头在最上面两条线之间, 数据从第二条线开始
    if len(h_y_values) < 2:
        return None
    header_top = h_y_values[0]
    data_start_y = h_y_values[1]

    # ── 提取 words ──
    try:
        words = page_plum.extract_words(keep_blank_chars=True)
    except Exception:
        return None
    if not words:
        return None

    # ── 按 y 坐标聚类 words 为行 ──
    ROW_TOLERANCE = 5  # 同一行 y 差 < 5pt
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    rows_words = []  # list of (y, [words])
    current_y = -999
    current_row = []
    for w in sorted_words:
        if w["top"] - current_y > ROW_TOLERANCE:
            if current_row:
                rows_words.append((current_y, current_row))
            current_y = w["top"]
            current_row = [w]
        else:
            current_row.append(w)
    if current_row:
        rows_words.append((current_y, current_row))

    # ── 只取表头行(header_top ~ data_start_y 之间) + 数据行(之后) ──
    # 用 _is_header_row 验证: 裁剪后 header 区间可能包含数据行
    # (如原始第1条 hline 被裁掉, data_start_y 后移, 数据行落入 header 区间)
    header_rows = []
    data_rows = []
    for y, rw in rows_words:
        if header_top - 2 <= y < data_start_y:
            # 用词表验证: 含日期/金额/长数字 → 数据行, 不是表头
            texts = [w["text"].strip() for w in rw if w["text"].strip()]
            if _is_header_row(texts):
                header_rows.append(rw)
            else:
                data_rows.append(rw)
        elif y >= data_start_y:
            data_rows.append(rw)

    if not data_rows:
        return None

    # ── 将 words 分配到列 ──
    def _words_to_row(row_words):
        cells = [""] * col_count
        for w in sorted(row_words, key=lambda w: w["x0"]):
            wx = w["x0"]
            assigned = False
            for ci, (x0, x1) in enumerate(intervals):
                if x0 - 5 <= wx < x1 + 5:
                    if cells[ci]:
                        cells[ci] += " " + w["text"]
                    else:
                        cells[ci] = w["text"]
                    assigned = True
                    break
            if not assigned and col_count > 0:
                # 超出右边界的归到最后一列
                if wx >= x_positions[-1] - 5:
                    if cells[-1]:
                        cells[-1] += " " + w["text"]
                    else:
                        cells[-1] = w["text"]
        return cells

    # 构建表头
    header_cells = [""] * col_count
    for rw in header_rows:
        merged = _words_to_row(rw)
        for ci in range(col_count):
            if merged[ci]:
                if header_cells[ci]:
                    header_cells[ci] += " " + merged[ci]
                else:
                    header_cells[ci] = merged[ci]

    # ── 计算表头锚点 (crop-immune) ──
    # 从 data_start_y 之前的所有 words 中, 为每个列区间找最近的 word center
    # 不依赖 header_rows 是否被正确识别, 完全不受引擎裁剪影响
    pre_data_words = [w for w in words if w["top"] < data_start_y]
    header_anchors = []
    for ci in range(col_count):
        interval_mid = (intervals[ci][0] + intervals[ci][1]) / 2
        if pre_data_words:
            # 找 x 中心最接近区间中点的 pre-data word
            best_w = min(
                pre_data_words,
                key=lambda w: abs((w["x0"] + w.get("x1", w["x0"] + 10)) / 2 - interval_mid)
            )
            anchor = (best_w["x0"] + best_w.get("x1", best_w["x0"] + 10)) / 2
            # 只采纳距离区间中点 < 半区间宽度的 word (防止误匹配)
            interval_half = (intervals[ci][1] - intervals[ci][0]) / 2
            if abs(anchor - interval_mid) < interval_half:
                header_anchors.append(anchor)
                continue
        # fallback: 区间中点
        header_anchors.append(interval_mid)

    logger.debug(
        f"[v2] hline-columns: anchors={[f'{a:.1f}' for a in header_anchors]}"
    )

    # ── 数据行: 最近邻锚点分配 ──
    def _words_to_row_nn(row_words):
        cells = [""] * col_count
        for w in sorted(row_words, key=lambda w: w["x0"]):
            w_center = (w["x0"] + w.get("x1", w["x0"] + 5)) / 2
            best_ci = min(
                range(col_count),
                key=lambda ci: abs(w_center - header_anchors[ci])
            )
            if cells[best_ci]:
                cells[best_ci] += " " + w["text"]
            else:
                cells[best_ci] = w["text"]
        return cells

    # 构建数据行 (最近邻)
    table = [header_cells]
    for rw in data_rows:
        table.append(_words_to_row_nn(rw))

    # 验证: 数据行太少或列太少 → 不是有效表格
    if len(table) < 2 or col_count < 2:
        return None

    logger.info(
        f"[v2] hline-columns: {len(table)-1} data rows, "
        f"{col_count} cols from {len(h_lines)} h-lines"
    )
    return table


def _extract_by_rect_columns(page_plum) -> Optional[List[List[str]]]:
    """矩形列边界法。"""
    rects = page_plum.rects
    if not rects or len(rects) < 3:
        return None

    y_groups = defaultdict(list)
    for r in rects:
        y_key = round(r["top"] / 3) * 3
        y_groups[y_key].append(r)

    best_group = max(y_groups.values(), key=len)
    if len(best_group) < 3:
        return None

    raw_x = sorted(set(
        round(v, 1) for r in best_group
        for v in [r["x0"], r["x1"]]
    ))
    x_positions = [0.0]
    for x in raw_x:
        if x - x_positions[-1] > 2:
            x_positions.append(x)
    x_positions.append(page_plum.width)

    if len(x_positions) < 4:
        return None

    header_top = min(r["top"] for r in best_group) - 2
    header_bottom = max(r["bottom"] for r in best_group) + 1

    try:
        cropped = page_plum.crop((
            0, header_top,
            page_plum.width, page_plum.height,
        ))
        chars = cropped.chars
        if not chars:
            return None

        col_count = len(x_positions) - 1
        intervals = [(x_positions[i], x_positions[i + 1]) for i in range(col_count)]

        def _chars_to_row(row_chars):
            cells = [""] * col_count
            for c in sorted(row_chars, key=lambda c: c["x0"]):
                for ci, (x0, x1) in enumerate(intervals):
                    if x0 - 2 <= c["x0"] < x1 + 2:
                        cells[ci] += c["text"]
                        break
            return [cell.strip() for cell in cells]

        header_chars = [c for c in chars if c["top"] < header_bottom]
        data_chars = [c for c in chars if c["top"] >= header_bottom]

        table = []
        if header_chars:
            table.append(_chars_to_row(header_chars))

        row_groups = defaultdict(list)
        for c in data_chars:
            y_key = round(c["top"] / 3) * 3
            row_groups[y_key].append(c)

        for y_key in sorted(row_groups.keys()):
            row = _chars_to_row(row_groups[y_key])
            if any(cell for cell in row):
                table.append(row)

        while table and all(not row[0] for row in table):
            table = [row[1:] for row in table]
        while table and all(not row[-1] for row in table):
            table = [row[:-1] for row in table]

        if len(table) >= 3:
            return table

    except Exception as e:
        logger.debug(f"[v2] rect columns failed: {e}")

    return None


def detect_columns_by_header_anchors(page_plum) -> Optional[List[List[str]]]:
    """表头锚点法。"""
    chars = page_plum.chars
    if not chars or len(chars) < 10:
        return None

    chars = [c for c in chars if not is_watermark_char(c)]
    if not chars:
        return None

    rows_by_y = _group_chars_into_rows(chars)
    if len(rows_by_y) < 2:
        return None

    header_row_idx = -1
    best_vocab_score = 0
    # 扫描前 15 行: 优先 vocab 匹配, 兼容 KV 元数据行在表头前的场景
    for i, (y_mid, row_chars) in enumerate(rows_by_y[:15]):
        row_text = _chars_to_text(row_chars)
        cells = [t.strip() for t in row_text.split("  ") if t.strip()]
        if len(cells) < 2:
            continue
        vs = _score_header_by_vocabulary(cells)
        if vs > best_vocab_score:
            best_vocab_score = vs
            header_row_idx = i
        elif vs == 0 and header_row_idx == -1:
            # Fallback: 结构启发式 (无 vocab 匹配时)
            if all(_is_header_cell(c) for c in cells[:4]):
                header_row_idx = i

    if header_row_idx == -1:
        return None

    header_chars = rows_by_y[header_row_idx][1]
    col_bounds = _cluster_x_positions([c["x0"] for c in header_chars])

    if len(col_bounds) < 2:
        return None

    result: List[List[str]] = []
    for y_mid, row_chars in rows_by_y[header_row_idx:]:
        row = _assign_chars_to_columns(row_chars, col_bounds)
        result.append(row)

    return result if len(result) >= 2 else None


def _adjust_boundaries_by_vocab(
    col_boundaries: List[float],
    header_chars: List[dict],
) -> List[float]:
    """词表引导的列边界校正: 如果边界落在已知表头词内部, 移动到词的后方。

    算法:
        1. 从 header chars 中提取 non-space 字符, 拼接为完整表头文本
        2. 用 _find_vocab_words_in_string 找到所有 vocab 匹配
        3. 对每个匹配, 用 char 的 x 坐标确定词的 x 范围
        4. 如果任何列边界落在某个词的 x 范围内, 将边界移到该词之后
    """
    text_chars = [c for c in header_chars if c["text"].strip()]
    if not text_chars:
        return col_boundaries

    full_text = "".join(c["text"] for c in text_chars)
    found = _find_vocab_words_in_string(full_text)
    if not found:
        return col_boundaries

    adjusted = list(col_boundaries)
    modified = False

    for word, start_idx, end_idx in found:
        if end_idx > len(text_chars):
            continue
        word_x0 = text_chars[start_idx]["x0"]
        word_x1 = text_chars[end_idx - 1]["x1"]

        for bi in range(1, len(adjusted) - 1):
            bx = adjusted[bi]
            if word_x0 + 1 < bx < word_x1 - 1:
                # 边界落在 vocab word 内部 → 移到词的后方
                new_bx = word_x1 + 0.5
                logger.debug(
                    f"[v2] vocab boundary fix: {bx:.1f}→{new_bx:.1f} "
                    f"to preserve '{word}'"
                )
                adjusted[bi] = new_bx
                modified = True
                break

    if modified:
        adjusted.sort()

    return adjusted


def detect_columns_by_whitespace_projection(
    page_plum,
) -> Optional[List[List[str]]]:
    """垂直空白投影法 — 对所有行投影 x 坐标, 用空白带检测列边界。

    算法:
        1. 收集所有 non-space 字符, 按 y 分行
        2. 对每个 x 位置 (1pt 分辨率), 统计有多少行在该位置有文字
        3. 投影值 ≤ 10% 行数的位置视为"空白"
        4. 宽度 ≥ 3pt 的连续空白带 → 列边界 (取中点)
        5. 根据列边界, 将每行字符切分为 cells

    适用场景: 无线条 borderless 表格, 列对齐靠空格/间距
    """
    chars = page_plum.chars
    if not chars or len(chars) < 20:
        return None

    # F-1: 自适应行分组容差
    from .utils import _adaptive_row_tolerance
    row_tol = _adaptive_row_tolerance(chars)

    # 收集 non-space chars, 按 y 分行 (使用自适应容差)
    text_chars = [c for c in chars if c["text"].strip()]
    if not text_chars:
        return None
    sorted_chars = sorted(text_chars, key=lambda c: c["top"])
    y_rows: Dict[int, List] = {}
    current_yk = round(sorted_chars[0]["top"] / row_tol) * row_tol
    y_rows[current_yk] = [sorted_chars[0]]
    for c in sorted_chars[1:]:
        ck = round(c["top"] / row_tol) * row_tol
        if abs(c["top"] - current_yk) <= row_tol:
            y_rows.setdefault(current_yk, []).append(c)
        else:
            current_yk = ck
            y_rows.setdefault(current_yk, []).append(c)

    if len(y_rows) < 3:
        return None

    # x 坐标范围
    all_text_chars = [c for row in y_rows.values() for c in row]
    x_min = min(c["x0"] for c in all_text_chars)
    x_max = max(c["x1"] for c in all_text_chars)
    width = int(x_max - x_min) + 2
    if width < 20:
        return None

    # 构建 x 轴投影直方图
    row_count = len(y_rows)
    projection = [0] * width

    for row_chars in y_rows.values():
        marked = set()
        for c in row_chars:
            c_x0 = max(0, int(c["x0"] - x_min))
            c_x1 = min(width - 1, int(c["x1"] - x_min))
            for x in range(c_x0, c_x1 + 1):
                marked.add(x)
        for x in marked:
            projection[x] += 1

    # F-3: 动态列间距阈值 (基于平均字符宽度)
    avg_char_w = sum(c["x1"] - c["x0"] for c in all_text_chars) / len(all_text_chars)
    min_gap_width = max(2.0, avg_char_w * 0.5)  # 最小 2pt 或半个字符宽

    # 找白带: 投影值 ≤ 10% 行数的连续区间
    threshold = row_count * 0.10
    gaps: List[Tuple[float, float, int]] = []
    in_gap = False
    gap_start = 0

    for x in range(width):
        if projection[x] <= threshold:
            if not in_gap:
                gap_start = x
                in_gap = True
        else:
            if in_gap:
                gap_width = x - gap_start
                if gap_width >= min_gap_width:  # F-3: 使用动态阈值
                    gaps.append((gap_start + x_min, x - 1 + x_min, gap_width))
                in_gap = False
    # 处理末尾的 gap
    if in_gap:
        gap_width = width - gap_start
        if gap_width >= 3:
            gaps.append((gap_start + x_min, width - 1 + x_min, gap_width))

    if len(gaps) < 2:
        return None  # 至少需要 2 个间隔才能划分 3+ 列

    # 列边界 = [x_min, gap1_mid, gap2_mid, ..., x_max]
    col_boundaries = [x_min]
    for g_start, g_end, _ in gaps:
        col_boundaries.append((g_start + g_end) / 2)
    col_boundaries.append(x_max + 1)

    n_cols = len(col_boundaries) - 1
    if n_cols < 3 or n_cols > 20:
        return None

    # 词表引导的边界校正: 避免列边界切割已知表头词
    first_yk = sorted(y_rows.keys())[0]
    header_chars = sorted(y_rows[first_yk], key=lambda c: c["x0"])
    col_boundaries = _adjust_boundaries_by_vocab(col_boundaries, header_chars)
    n_cols = len(col_boundaries) - 1  # 边界数可能不变, 但位置调整

    # 按列边界切分每行
    result: List[List[str]] = []
    for yk in sorted(y_rows.keys()):
        row_chars = sorted(y_rows[yk], key=lambda c: c["x0"])
        
        # 1. 将相邻字符合并成 word (避免单词从中间被竖线切断)
        words = []
        curr_word = None
        for c in row_chars:
            if not str(c.get("text", "")).strip():
                continue
            if not curr_word:
                curr_word = {"x0": c["x0"], "x1": c.get("x1", c["x0"]), "text": c["text"]}
            else:
                gap = c["x0"] - curr_word["x1"]
                if gap < 2.5:
                    curr_word["x1"] = max(curr_word["x1"], c.get("x1", c["x0"]))
                    curr_word["text"] += c["text"]
                else:
                    words.append(curr_word)
                    curr_word = {"x0": c["x0"], "x1": c.get("x1", c["x0"]), "text": c["text"]}
        if curr_word:
            words.append(curr_word)

        cells: List[str] = []
        for i in range(n_cols):
            left = col_boundaries[i]
            right = col_boundaries[i + 1]
            # 以 word 的中心点为基准分配列
            cell_words = [
                w for w in words
                if (w["x0"] + w["x1"]) / 2 >= left - 1
                and (w["x0"] + w["x1"]) / 2 < right + 1
            ]
            cell_text = " ".join(w["text"] for w in cell_words).strip()
            cells.append(cell_text)
        result.append(cells)

    logger.debug(
        f"[v2] whitespace_projection: {len(result)} rows, "
        f"{n_cols} cols from {len(gaps)} gaps"
    )

    if len(result) < 2:
        return None

    # ── vocab 扫描: 找到真正的表头行, 跳过 KV 元数据行 ──
    # 有些 PDF 的 table zone 包含 KV 元数据行 (如 "户名:xxx"),
    # 这些行在表头之前, 需要跳过才能得到正确的表格
    best_header_idx = 0
    best_header_vs = 0
    scan_limit = min(15, len(result))
    for ri in range(scan_limit):
        vs = _score_header_by_vocabulary(result[ri])
        if vs > best_header_vs:
            best_header_vs = vs
            best_header_idx = ri

    if best_header_vs >= 3 and best_header_idx > 0:
        logger.info(
            f"[v2] whitespace_projection: header found at row {best_header_idx} "
            f"(vocab={best_header_vs}), skipping {best_header_idx} preamble rows"
        )
        result = result[best_header_idx:]

    return result if len(result) >= 2 else None


def detect_columns_by_clustering(page_plum) -> Optional[List[List[str]]]:
    """x 坐标聚类法。"""
    chars = page_plum.chars
    if not chars or len(chars) < 10:
        return None

    chars = [c for c in chars if not is_watermark_char(c)]
    if not chars:
        return None

    all_x0 = [c["x0"] for c in chars]
    col_bounds = _cluster_x_positions(all_x0, gap_multiplier=2.5)

    if len(col_bounds) < 2:
        return None

    rows_by_y = _group_chars_into_rows(chars)
    result: List[List[str]] = []
    for y_mid, row_chars in rows_by_y:
        row = _assign_chars_to_columns(row_chars, col_bounds)
        result.append(row)

    return result if len(result) >= 2 else None


def detect_columns_by_word_anchors(page_plum) -> Optional[List[List[str]]]:
    """
    Word 锚点列检测。

    用 extract_words() 定位表头中每个 word 的 x 位置作为列左边界,
    然后用 char 级别归箱提取数据。

    相比 char 级聚类, word 级间隙更明显, 能处理窄间距多列布局
    (如兴业银行: 交易金额/账户余额/交易地点 间距仅 8-9pt)。
    """
    try:
        # 先用更紧的 x_tolerance 提取 (区分 2-3pt 列间距)
        # 再用默认值提取, 取 words 更多的结果 (= 更精细的列分割)
        best_words = None
        for x_tol in (2, 3):
            w = page_plum.extract_words(
                keep_blank_chars=True, x_tolerance=x_tol
            )
            if w and (best_words is None or len(w) > len(best_words)):
                best_words = w
        words = best_words
    except Exception:
        return None
    if not words or len(words) < 5:
        return None

    # ── 按 y 分组 words 为行 ──
    ROW_TOL = 5
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    word_rows: List[Tuple[float, List[Dict]]] = []
    cur_y = sorted_words[0]["top"]
    cur_row = [sorted_words[0]]
    for w in sorted_words[1:]:
        if abs(w["top"] - cur_y) <= ROW_TOL:
            cur_row.append(w)
        else:
            y_mid = sum(ww["top"] for ww in cur_row) / len(cur_row)
            word_rows.append((y_mid, sorted(cur_row, key=lambda x: x["x0"])))
            cur_row = [w]
            cur_y = w["top"]
    if cur_row:
        y_mid = sum(ww["top"] for ww in cur_row) / len(cur_row)
        word_rows.append((y_mid, sorted(cur_row, key=lambda x: x["x0"])))

    if len(word_rows) < 2:
        return None

    # ── 找表头行: 前 5 行中 word 最多且都像表头的行 ──
    header_row_idx = -1
    for i, (y_mid, rw) in enumerate(word_rows[:5]):
        texts = [w["text"].strip() for w in rw if w["text"].strip()]
        if len(texts) < 3:
            continue
        header_count = sum(1 for t in texts if _is_header_cell(t))
        if header_count / len(texts) >= 0.5:
            header_row_idx = i
            break

    if header_row_idx == -1:
        return None

    header_words = word_rows[header_row_idx][1]
    if len(header_words) < 3:
        return None

    # ── 从表头 word 位置构建列边界 ──
    # 每个 word 的 x0, x1 为边界，_assign_chars_to_columns 的 split line 会自动落在 gap 之间
    col_bounds: List[Tuple[float, float]] = []
    for i, w in enumerate(header_words):
        x_start = w["x0"]
        x_end = w.get("x1", w["x0"] + 10)
        col_bounds.append((x_start, x_end))

    if len(col_bounds) < 3:
        return None

    # ── 用 char 级别归箱提取数据 ──
    chars = page_plum.chars
    if not chars:
        return None
    chars = [c for c in chars if not is_watermark_char(c)]
    if not chars:
        return None

    char_rows = _group_chars_into_rows(chars)

    # 从表头行的 y 位置开始提取
    header_y = word_rows[header_row_idx][0]
    result: List[List[str]] = []
    for y_mid, row_chars in char_rows:
        if y_mid < header_y - 3:
            continue
        row = _assign_chars_to_columns(row_chars, col_bounds)
        result.append(row)

    if len(result) < 2:
        return None

    logger.info(
        f"[v2] word-anchors: {len(result)-1} data rows, "
        f"{len(col_bounds)} cols from {len(header_words)} header words"
    )
    return result


def detect_columns_by_data_voting(
    page_plum,
) -> Optional[List[List[str]]]:
    """数据行驱动的列边界检测。

    用数据行（含日期/金额的行）的 word 间隙位置投票, 确定列边界。
    比 header-anchors 更鲁棒: 不依赖表头行, 能处理双语混排表头。
    """
    try:
        words = page_plum.extract_words(
            keep_blank_chars=True, x_tolerance=2
        )
    except Exception:
        return None
    if not words or len(words) < 10:
        return None

    # ── 按 y 分组为行 ──
    ROW_TOL = 5
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    word_rows: List[Tuple[float, List[Dict]]] = []
    cur_y = sorted_words[0]["top"]
    cur_row = [sorted_words[0]]
    for w in sorted_words[1:]:
        if abs(w["top"] - cur_y) <= ROW_TOL:
            cur_row.append(w)
        else:
            y_mid = sum(ww["top"] for ww in cur_row) / len(cur_row)
            word_rows.append(
                (y_mid, sorted(cur_row, key=lambda x: x["x0"]))
            )
            cur_row = [w]
            cur_y = w["top"]
    if cur_row:
        y_mid = sum(ww["top"] for ww in cur_row) / len(cur_row)
        word_rows.append(
            (y_mid, sorted(cur_row, key=lambda x: x["x0"]))
        )

    if len(word_rows) < 5:
        return None

    # ── 筛选数据行: 含日期或金额的行 ──
    data_rows: List[Tuple[float, List[Dict]]] = []
    for y_mid, rw in word_rows:
        texts = " ".join(w["text"] for w in rw)
        if _RE_IS_DATE.search(texts):
            data_rows.append((y_mid, rw))
        elif any(
            _RE_IS_AMOUNT.match(
                w["text"].strip().replace(",", "").replace("¥", "")
            )
            for w in rw
            if w["text"].strip()
        ):
            data_rows.append((y_mid, rw))

    if len(data_rows) < 3:
        return None

    # ── 收集 gap 中点位置 (3pt 分辨率) ──
    gap_votes: Dict[int, int] = defaultdict(int)
    page_w = page_plum.width or 600
    for _, rw in data_rows[:30]:
        for i in range(len(rw) - 1):
            gap_left = rw[i]["x1"]
            gap_right = rw[i + 1]["x0"]
            if gap_right - gap_left < 3:
                continue  # 太窄, 不是列间隙
            gap_mid = (gap_left + gap_right) / 2
            bucket = round(gap_mid / 3) * 3
            gap_votes[bucket] += 1

    if not gap_votes:
        return None

    # ── 投票: gap 出现在 ≥40% 数据行中 → 列边界 ──
    n_voters = min(len(data_rows), 30)
    threshold = max(3, int(n_voters * 0.4))
    voted_gaps = sorted(
        x for x, count in gap_votes.items() if count >= threshold
    )

    if len(voted_gaps) < 2:
        return None

    # ── 合并相邻 gap (< 8pt → 同一边界) ──
    merged_gaps: List[float] = [voted_gaps[0]]
    for g in voted_gaps[1:]:
        if g - merged_gaps[-1] < 8:
            merged_gaps[-1] = (merged_gaps[-1] + g) / 2
        else:
            merged_gaps.append(g)

    # ── 从 gap 中点构建列边界 ──
    col_bounds: List[Tuple[float, float]] = []
    col_bounds.append((0, merged_gaps[0]))
    for i in range(len(merged_gaps) - 1):
        col_bounds.append((merged_gaps[i], merged_gaps[i + 1]))
    col_bounds.append((merged_gaps[-1], page_w))

    if len(col_bounds) < 3:
        return None

    # ── 用 char 级别归箱提取全部行 ──
    chars = page_plum.chars
    if not chars:
        return None
    chars = [c for c in chars if not is_watermark_char(c)]
    if not chars:
        return None

    char_rows = _group_chars_into_rows(chars)
    result: List[List[str]] = []
    for _, row_chars in char_rows:
        row = _assign_chars_to_columns(row_chars, col_bounds)
        result.append(row)

    if len(result) < 3:
        return None

    logger.info(
        f"[v2] data-voting: {len(result)} rows, "
        f"{len(col_bounds)} cols from "
        f"{len(data_rows)} data rows, "
        f"{len(merged_gaps)} voted gaps"
    )
    return result

