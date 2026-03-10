"""
数据校验中间件 (Validator)
===========================

2 维加权评分体系:
    1. 列一致性 (column_consistency): 每行列数是否与表头一致
    2. 日期格式覆盖率 (date_coverage): 日期列的有效格式占比

第一性原理优化:
    - 宽容策略: 加权总分 ≥ 阈值即通过，不要求所有维度满分
    - 纯检查，不修改数据 — 修复由 Repairer 负责
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseMiddleware
from ...models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)


# 日期正则
_RE_DATE = re.compile(
    r'^\d{8}\s*(\d{1,2}:\d{2}(:\d{2})?)?$|'
    r'^\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?'
    r'(\s*\d{1,2}:\d{2}(:\d{2})?)?$|'
    r'^\d{2}[-/]\d{2}[-/]\d{4}$'
)

# 检查维度权重 (运行时根据表格列类型动态调整)
DEFAULT_WEIGHTS = {
    "column_consistency": 0.50,
    "date_coverage": 0.50,
}

# 金额正则
_RE_AMOUNT = re.compile(r'^[+-]?\d[\d,]*\.?\d*$')


class Validator(BaseMiddleware):
    """
    数据校验中间件。

    将验证结果写入 ``EnhancedResult.enhanced_data["validation"]``。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, pass_threshold: float = 0.7):
        super().__init__(config)
        self.pass_threshold = pass_threshold

    def process(self, result: EnhancedResult) -> EnhancedResult:
        """执行 6 维数据校验。"""
        std_table = result.standardized_table
        if not std_table or len(std_table) < 2:
            # 如果没有标准化表格，尝试从 base_result 验证原始表格
            if result.base_result and result.base_result.table_blocks:
                main = max(
                    result.base_result.table_blocks,
                    key=lambda b: len(b.raw_content) if isinstance(b.raw_content, list) else 0
                )
                raw = main.raw_content
                if isinstance(raw, list) and len(raw) >= 2:
                    std_table = raw
                else:
                    result.enhanced_data["validation"] = {
                        "passed": False,
                        "total_score": 0.0,
                        "reason": "no_table_data",
                    }
                    return result
            else:
                result.enhanced_data["validation"] = {
                    "passed": False,
                    "total_score": 0.0,
                    "reason": "no_table_data",
                }
                return result

        headers = std_table[0]
        data_rows = std_table[1:]

        # ── 检查维度 ──
        details: Dict[str, float] = {}

        details["column_consistency"] = self._check_column_consistency(headers, data_rows)

        # 自适应权重: 有日期列 → date_coverage; 无日期列 → amount_validity 替代
        date_idx = self._find_column(headers, ["交易时间", "交易日期", "日期", "Date"])
        amount_idx = self._find_column(headers, ["交易金额", "金额", "Amount", "发生额"])

        if date_idx is not None:
            details["date_coverage"] = self._check_date_coverage(headers, data_rows)
            weights = {"column_consistency": 0.50, "date_coverage": 0.50}
        elif amount_idx is not None:
            details["amount_validity"] = self._check_amount_validity(data_rows, amount_idx)
            weights = {"column_consistency": 0.50, "amount_validity": 0.50}
        else:
            weights = {"column_consistency": 1.0}

        # ── 加权总分 ──
        total_score = sum(
            details.get(dim, 0.0) * weight
            for dim, weight in weights.items()
        )
        passed = total_score >= self.pass_threshold

        validation = {
            "passed": passed,
            "total_score": round(total_score, 4),
            "details": {k: round(v, 4) for k, v in details.items()},
            "threshold": self.pass_threshold,
            "row_count": len(data_rows),
        }

        result.enhanced_data["validation"] = validation

        # 记录 Mutation
        result.record_mutation(
            middleware_name=self.name,
            target_block_id="document",
            field_changed="validation",
            old_value=None,
            new_value=f"score={total_score:.3f} passed={passed}",
            confidence=1.0,
            reason=f"2-dim check: {details}",
        )

        logger.info(
            f"[Validator] score={total_score:.3f} | "
            f"passed={passed} | rows={len(data_rows)}"
        )

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # 检查维度
    # ═══════════════════════════════════════════════════════════════════════════

    def _check_column_consistency(
        self, headers: List[str], data_rows: List[List[str]]
    ) -> float:
        """每行列数是否与表头一致。"""
        if not data_rows:
            return 1.0
        expected = len(headers)
        consistent = sum(1 for row in data_rows if len(row) == expected)
        return consistent / len(data_rows)

    def _check_date_coverage(
        self, headers: List[str], data_rows: List[List[str]]
    ) -> float:
        """日期列的有效格式占比。"""
        date_idx = self._find_column(headers, ["交易时间", "交易日期", "日期", "Date"])
        if date_idx is None:
            return 0.5  # 没有日期列不扣太多分

        valid = 0
        total = 0
        for row in data_rows:
            if date_idx < len(row):
                val = row[date_idx].strip()
                total += 1
                if val and _RE_DATE.match(val):
                    valid += 1

        return valid / total if total > 0 else 0.5

    def _check_amount_validity(
        self, data_rows: List[List[str]], amount_idx: int,
    ) -> float:
        """金额列的有效数值占比 — 替代无日期列时的 date_coverage。"""
        valid = 0
        total = 0
        for row in data_rows:
            if amount_idx < len(row):
                val = row[amount_idx].strip().replace(",", "").replace("，", "").replace("¥", "")
                total += 1
                if val and _RE_AMOUNT.match(val):
                    valid += 1

        return valid / total if total > 0 else 0.5

    # ═══════════════════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _find_column(headers: List[str], keywords: List[str]) -> Optional[int]:
        """找到第一个匹配关键字的列索引。"""
        for i, h in enumerate(headers):
            h_clean = h.strip()
            if not h_clean:
                continue
            for kw in keywords:
                if kw in h_clean or h_clean in kw:
                    return i
        return None
