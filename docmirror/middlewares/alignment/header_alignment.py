"""
表头-数据对齐验证 (Header-Data Alignment Verification)
======================================================

从 ``column_mapper.py`` 提取: 基于内容类型推断的列对齐校验和修正。

核心功能:
  - ``infer_column_type()``: 对数据列做类型分布采样 (date/amount/seq/text)
  - ``verify_header_data_alignment()``: 检测并修正表头与数据的系统性偏移
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 日期正则 (宽松，覆盖 YYYYMMDD / YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD 等)
_RE_COL_DATE = re.compile(
    r'^\d{8}(\s*\d{1,2}:\d{2}(:\d{2})?)?$|'
    r'^\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?'
    r'(\s*\d{1,2}:\d{2}(:\d{2})?)?$|'
    r'^\d{2}[-/]\d{2}[-/]\d{4}$'
)
# 金额正则 (含逗号分隔)
_RE_COL_AMOUNT = re.compile(r'^[+-]?\d[\d,]*\.\d{1,4}$')
# 序号正则 (纯整数, 1~8 位)
_RE_COL_SEQ = re.compile(r'^\d{1,8}$')


def infer_column_type(
    data_rows: List[List[str]], col_idx: int, sample_size: int = 30,
) -> Dict[str, float]:
    """推断单列的数据类型分布。

    Returns:
        {"date": 0.9, "amount": 0.05, "seq": 0.0, "text": 0.05}
    """
    counts = {"date": 0, "amount": 0, "seq": 0, "text": 0}
    total = 0
    for row in data_rows[:sample_size]:
        if col_idx >= len(row):
            continue
        val = (row[col_idx] or "").strip()
        if not val:
            continue
        total += 1
        clean = val.replace(",", "").replace("，", "").replace("¥", "")
        if _RE_COL_DATE.match(val):
            counts["date"] += 1
        elif _RE_COL_AMOUNT.match(clean):
            counts["amount"] += 1
        elif _RE_COL_SEQ.match(val):
            counts["seq"] += 1
        else:
            counts["text"] += 1

    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def verify_header_data_alignment(
    headers: List[str],
    data_rows: List[List[str]],
    header_type_expectations: Dict[str, str],
    mutation_recorder=None,
    middleware_name: str = "ColumnMapper",
) -> List[str]:
    """验证表头与数据列是否对齐，检测并修正系统性偏移。

    Args:
        headers: 原始表头列表。
        data_rows: 数据行。
        header_type_expectations: 表头名 → 期望类型 ("date"/"amount"/"seq")。
        mutation_recorder: 可选的 EnhancedResult 用于记录 mutation。
        middleware_name: mutation 记录的中间件名。

    Returns:
        修正后的表头列表 (或原样返回)。
    """
    n_cols = len(headers)
    if len(data_rows) < 5 or n_cols < 3:
        return headers

    # ── Step 1: 收集锚点列 ──
    anchors: List[Dict] = []
    for i, h in enumerate(headers):
        h_clean = h.strip()
        if not h_clean:
            continue
        expected = header_type_expectations.get(h_clean)
        if not expected:
            for kw, typ in header_type_expectations.items():
                if len(kw) >= 2 and kw in h_clean:
                    expected = typ
                    break
        if expected:
            anchors.append({
                "header_idx": i,
                "expected_type": expected,
                "header_name": h_clean,
            })

    if len(anchors) < 2:
        return headers

    # ── Step 2: 对每个锚点列，检查是否对齐 ──
    offsets: List[Dict] = []
    for anchor in anchors:
        hi = anchor["header_idx"]
        et = anchor["expected_type"]
        current_profile = infer_column_type(data_rows, hi)
        current_match = current_profile.get(et, 0.0)

        if current_match >= 0.5:
            offsets.append({"anchor": anchor, "offset": 0, "confidence": current_match})
            continue

        best_offset = 0
        best_match = current_match
        for delta in [1, -1, 2, -2]:
            check_idx = hi + delta
            if 0 <= check_idx < n_cols:
                profile = infer_column_type(data_rows, check_idx)
                match = profile.get(et, 0.0)
                if match > best_match:
                    best_match = match
                    best_offset = delta

        if best_match >= 0.5 and best_offset != 0:
            offsets.append({"anchor": anchor, "offset": best_offset, "confidence": best_match})
        else:
            offsets.append({"anchor": anchor, "offset": 0, "confidence": current_match})

    # ── Step 3: 检测系统性偏移 ──
    non_zero = [o for o in offsets if o["offset"] != 0]
    if len(non_zero) < 2:
        return headers

    offset_counts = Counter(o["offset"] for o in non_zero)
    dominant_offset, dominant_count = offset_counts.most_common(1)[0]
    if dominant_count < 2:
        return headers

    zero_count = sum(1 for o in offsets if o["offset"] == 0)
    if zero_count > dominant_count:
        return headers

    # ── Step 4: 修正表头 ──
    logger.info(
        f"[{middleware_name}] alignment fix: detected systematic offset={dominant_offset}, "
        f"anchors={[(o['anchor']['header_name'], o['offset']) for o in offsets]}"
    )

    new_headers = list(headers)
    if dominant_offset > 0:
        for _ in range(abs(dominant_offset)):
            first_shift_idx = min(
                o["anchor"]["header_idx"] for o in non_zero
                if o["offset"] == dominant_offset
            )
            new_headers.insert(first_shift_idx, "")
        new_headers = new_headers[:n_cols]
    elif dominant_offset < 0:
        for _ in range(abs(dominant_offset)):
            first_shift_idx = min(
                o["anchor"]["header_idx"] for o in non_zero
                if o["offset"] == dominant_offset
            )
            remove_idx = None
            for ri in range(first_shift_idx, -1, -1):
                if not new_headers[ri].strip():
                    remove_idx = ri
                    break
            if remove_idx is not None:
                new_headers.pop(remove_idx)
                new_headers.append("")

    if mutation_recorder:
        mutation_recorder.record_mutation(
            middleware_name=middleware_name,
            target_block_id="document",
            field_changed="header_alignment",
            old_value=str(headers[:5]),
            new_value=str(new_headers[:5]),
            confidence=0.85,
            reason=f"systematic offset={dominant_offset}, {dominant_count} anchors confirmed",
        )

    logger.info(f"[{middleware_name}] alignment fix applied: {headers[:6]} → {new_headers[:6]}")
    return new_headers
