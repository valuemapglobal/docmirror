"""
借贷分列检测 (Amount Split Detection)
=======================================

从 ``column_mapper.py`` 提取: 检测表格中收入/支出分列模式。

支持三种模式:
  1. 显式分列: 表头含收入/支出关键字
  2. 隐式分列: 金额列旁有空表头列 (如浦发银行 '发生额'+空列)
  3. 粘连分列: 粘连列名中嵌入借贷关键字

F-6 增强:
  - 数据行验证: 检查借贷列是否同时有值 (>30% 冲突 → 非分列)
  - 借贷标志列检测: 有「借/贷」标志列时不做 amount split
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# F-6: 数据验证 — 金额正则
_RE_HAS_NUMBER = re.compile(r"\d")


def _validate_split_by_data(
    data_rows: List[List[str]],
    inc_idx: Optional[int],
    exp_idx: Optional[int],
    max_sample: int = 20,
) -> bool:
    """F-6: 采样数据行验证分列是否合理。

    如果 >30% 的数据行在 income 和 expense 列同时有值，
    说明这不是真正的借贷分列 (而是两个独立的金额列)。
    """
    if inc_idx is None or exp_idx is None or not data_rows:
        return True  # 无法验证，默认信任表头

    sample = data_rows[:max_sample]
    both_count = 0
    valid_count = 0

    for row in sample:
        if inc_idx >= len(row) or exp_idx >= len(row):
            continue
        inc_val = (row[inc_idx] or "").strip()
        exp_val = (row[exp_idx] or "").strip()
        inc_has = bool(inc_val and _RE_HAS_NUMBER.search(inc_val))
        exp_has = bool(exp_val and _RE_HAS_NUMBER.search(exp_val))

        if inc_has or exp_has:
            valid_count += 1
            if inc_has and exp_has:
                both_count += 1

    if valid_count < 3:
        return True  # 数据太少，默认信任

    conflict_ratio = both_count / valid_count
    if conflict_ratio > 0.3:
        logger.info(
            f"[AmountSplit] F-6: split rejected by data validation "
            f"(conflict={both_count}/{valid_count}={conflict_ratio:.0%})"
        )
        return False
    return True


def detect_split_amount(
    headers: List[str],
    mapping: Dict[str, Optional[str]],
    income_keywords: Set[str],
    expense_keywords: Set[str],
    amount_like_keywords: Set[str],
    data_rows: Optional[List[List[str]]] = None,
) -> Tuple[bool, Optional[int], Optional[int]]:
    """检测是否存在收入/支出分列。

    Args:
        headers: 原始表头列表。
        mapping: 列映射结果 {raw_header: standard_name or None}。
        income_keywords: 收入关键字集合。
        expense_keywords: 支出关键字集合。
        amount_like_keywords: 金额类关键字集合。
        data_rows: (F-6) 可选的数据行，用于验证分列是否合理。

    Returns:
        (has_split, income_idx, expense_idx)
    """
    # F-6: 检测借贷标志列 — 有标志列时不做 amount split
    _DEBIT_CREDIT_FLAGS = {"借贷标志", "借贷", "借/贷", "收支", "支/收",
                           "借贷状态", "收支标志", "DC标志"}
    header_set = {h.strip() for h in headers if h}
    if header_set & _DEBIT_CREDIT_FLAGS:
        logger.info("[AmountSplit] F-6: skipped — debit/credit flag column found")
        return False, None, None

    has_income = bool(header_set & income_keywords)
    has_expense = bool(header_set & expense_keywords)

    if has_income and has_expense:
        # 模式1: 显式分列
        inc_idx = exp_idx = None
        for i, h in enumerate(headers):
            h_clean = h.strip()
            if h_clean in income_keywords and inc_idx is None:
                inc_idx = i
            elif h_clean in expense_keywords and exp_idx is None:
                exp_idx = i
        # F-6: 数据验证
        if data_rows and not _validate_split_by_data(data_rows, inc_idx, exp_idx):
            return False, None, None
        return True, inc_idx, exp_idx

    # 模式2: 金额列 + 相邻空表头列 → 隐式借贷分列
    for i, h in enumerate(headers):
        h_clean = h.strip()
        if h_clean in amount_like_keywords and i + 1 < len(headers):
            next_h = headers[i + 1].strip()
            if not next_h:
                # F-6: 数据验证
                if data_rows and not _validate_split_by_data(data_rows, i + 1, i):
                    continue
                logger.info(
                    f"[AmountSplit] detected implicit split: "
                    f"'{h_clean}'(idx={i})=expense + empty(idx={i+1})=income"
                )
                return True, i + 1, i

    # 模式3: 粘连列名中嵌入借贷关键字
    merged_inc_idx = merged_exp_idx = None
    for i, h in enumerate(headers):
        h_clean = h.strip()
        if not h_clean:
            continue
        for kw in income_keywords:
            if kw in h_clean:
                if h_clean == kw:
                    merged_inc_idx = i
                elif merged_inc_idx is None:
                    merged_inc_idx = i
                break
        for kw in expense_keywords:
            if kw in h_clean:
                if h_clean == kw:
                    merged_exp_idx = i
                elif merged_exp_idx is None:
                    merged_exp_idx = i
                break

    if (merged_inc_idx is not None and merged_exp_idx is not None
            and merged_inc_idx != merged_exp_idx):
        # F-6: 数据验证
        if data_rows and not _validate_split_by_data(data_rows, merged_inc_idx, merged_exp_idx):
            return False, None, None
        logger.info(
            f"[AmountSplit] detected merged split: "
            f"income(idx={merged_inc_idx})='{headers[merged_inc_idx].strip()[:20]}' + "
            f"expense(idx={merged_exp_idx})='{headers[merged_exp_idx].strip()[:20]}'"
        )
        return True, merged_inc_idx, merged_exp_idx

    return False, None, None
