"""
词汇表与行分类器 (Vocabulary & Row Classifiers)
=================================================

从 layout_analysis.py 拆分的词汇表系统和行判定函数。
包含 VOCAB_BY_CATEGORY、表头评分、行分类（header/data/junk）等。
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, List, Optional

from .text_utils import _is_cjk_char

logger = logging.getLogger(__name__)

_RE_IS_DATE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{8}")
_RE_IS_AMOUNT = re.compile(r"^[-+]?\d[\d,]*\.?\d*$")
_RE_VALID_DATE = re.compile(r'(?:19|20)\d{2}[-/.]?\d{1,2}[-/.]?\d{1,2}')


# ══════════════════════════════════════════════════════════════════════════════
# 分类词汇表 — 按文档类型隔离, 防止跨场景误匹配
# ══════════════════════════════════════════════════════════════════════════════

VOCAB_BY_CATEGORY: Dict[str, frozenset] = {
    # Bank Statement (银行对账单 / 流水单)
    "BANK_STATEMENT": frozenset({
        # 时间类
        "日期", "交易日期", "记账日期", "入账日期", "交易时间", "交易日",
        # 金额类
        "金额", "交易金额", "发生额", "收入金额", "支出金额", "本次余额",
        # 余额类
        "余额", "账户余额", "结存", "可用余额",
        # 摘要/附言
        "摘要", "摘要内容", "交易摘要", "用途", "附言", "备注",
        "交易类型", "交易类别", "业务摘要",
        # 对方信息
        "对方户名", "对方账号", "对手账号", "对手户名", "对方信息",
        "对方账户", "对方名称", "对方行名",
        # 序号/凭证
        "序号", "账户序号", "账号序号", "凭证类型", "凭证号码", "流水号",
        # 币种/钞汇
        "币种", "钞汇", "现/转", "现转", "钞汇标识", "现转标志",
        # 借贷方向
        "借贷", "借贷状态", "支/收", "收支", "借贷标志",
        # 渠道/机构
        "交易渠道", "交易机构", "交易方式", "交易地点",
        # 标识/标志
        "被冲账标识", "冲账标识",
        # 流水号 / 备注
        "交易流水号", "交易备注",
    }),
    # 预留: 增值税发票
    # "VAT_INVOICE": frozenset({
    #     "品名", "规格型号", "单位", "数量", "单价", "金额", "税率", "税额",
    #     "价税合计", "购买方", "销售方", "发票代码", "发票号码",
    # }),
    # 预留: 其他类型文书
    # "GENERAL": frozenset({...}),
}

# 向后兼容 — 所有 category 词汇的并集 (现有调用方无需改动)
KNOWN_HEADER_WORDS: frozenset = frozenset().union(*VOCAB_BY_CATEGORY.values())


# ── CJK 归一化: NFKC + 繁→简 补丁 ──
_TRAD_TO_SIMP = str.maketrans({
    "\u6236": "\u6237",  # 戶 → 户
    "\u2ea0": "\u6c11",  # ⺠ → 民
    "\u8cf3": "\u8d26",  # 賳 → 账 (traditional 賳)
    "\u865f": "\u53f7",  # 號 → 号
    "\u984d": "\u989d",  # 額 → 额
    "\u6642": "\u65f6",  # 時 → 时
    "\u6a5f": "\u673a",  # 機 → 机
    "\u69cb": "\u6784",  # 構 → 构
})

def _normalize_for_vocab(s: str) -> str:
    """对文本做 NFKC + 繁简补丁归一化, 用于词表匹配。"""
    return unicodedata.normalize("NFKC", s).translate(_TRAD_TO_SIMP)


def _score_header_by_vocabulary(
    row: List[str],
    categories: Optional[List[str]] = None,
) -> int:
    """计算一行中匹配已知表头词的单元格数量 (NFKC + 繁简归一化)。

    Args:
        row: 表格行单元格列表
        categories: 用于匹配的文档类 category 列表 (e.g. ["BANK_STATEMENT"]);
                    为 None 时使用全量 KNOWN_HEADER_WORDS (向后兼容)。
    """
    vocab = (
        frozenset().union(*(VOCAB_BY_CATEGORY.get(c, frozenset()) for c in categories))
        if categories else KNOWN_HEADER_WORDS
    )
    score = 0
    for cell in row:
        text = _normalize_for_vocab((cell or "")).strip()
        if text in vocab:
            score += 1
    return score


# 币种前缀金额: "RMB 631.20", "USD 100.00" 等
_RE_CURRENCY_AMOUNT = re.compile(r'^[A-Z]{2,4}\s+[\d,.]+$')
# 裸币种代码: "RMB", "USD", "CNY" 等
_RE_BARE_CURRENCY = re.compile(r'^[A-Z]{2,4}$')


# 纯时间格式: "08:54:19"
_RE_IS_TIME_ONLY = re.compile(r'^\d{2}:\d{2}:\d{2}$')
# 长纯数字串 (账号/流水号等, ≥10 位)
_RE_LONG_DIGIT = re.compile(r'^\d{10,}$')


def _is_header_cell(cell: str) -> bool:
    """单元格是否像表头 (短文本, 非日期, 非金额, 非时间, 非币种, 非长数字串)。"""
    cell = cell.strip()
    if not cell:
        return False
    if _RE_IS_DATE.search(cell):
        return False
    if _RE_IS_AMOUNT.match(cell.replace(",", "").replace("¥", "")):
        return False
    # 时间格式: "08:54:19" 是数据，不是表头
    if _RE_IS_TIME_ONLY.match(cell):
        return False
    # 长纯数字串 (账号/流水号 ≥10 位)
    if _RE_LONG_DIGIT.match(cell):
        return False
    # 币种前缀金额: "RMB 631.20"
    if _RE_CURRENCY_AMOUNT.match(cell):
        return False
    # 裸币种代码: "RMB", "USD"
    if _RE_BARE_CURRENCY.match(cell):
        return False
    # 单字符 (如 "N", "Y") 信息量不足, 不应视为表头
    if len(cell) <= 1:
        return False
    if len(cell) > 30:
        return False
    return True


def _is_header_row(row: List[str]) -> bool:
    """一行是否是表头行 — 双门控算法。

    Gate 1 (决定性否决): 任意 cell 含日期/金额/时间/长数字串 → 是数据行, 直接 False。
      - 这四类信号在真实表头中不会出现, precision ≈ 100%。

    Gate 2 (决定性确认): vocab_score ≥ 3 → 命中已知列名词汇, 直接 True。

    Fallback (结构启发): ≥60% 非空 cell 通过 _is_header_cell → True。
    """
    non_empty = [c.strip() for c in row if (c or "").strip()]
    if len(non_empty) < 2:
        return False

    # ── Gate 1: 强数据信号 → 决定性否决 ──
    for cell in non_empty:
        clean = cell.replace(",", "").replace("¥", "")
        if _RE_IS_DATE.search(cell):          # 日期
            return False
        if _RE_IS_AMOUNT.match(clean):        # 金额 (含小数点数字)
            return False
        if _RE_IS_TIME_ONLY.match(cell):      # 时间 HH:MM:SS
            return False
        if _RE_LONG_DIGIT.match(cell):        # 长数字串 (账号/流水号)
            return False

    # ── Gate 2: vocab 命中 ≥3 → 决定性确认 ──
    if _score_header_by_vocabulary(non_empty) >= 3:
        return True

    # ── Fallback: 结构启发 ──
    header_count = sum(1 for c in non_empty if _is_header_cell(c))
    return header_count / len(non_empty) >= 0.6


def _is_junk_row(row: List[str]) -> bool:
    """是否是页眉/页脚/合计等无效行 (上下文感知)。

    策略:
      1. 强信号 (合计/总计) 无论填充率都视为 junk
      2. 高填充率行 (>=60% 非空且 >3 个非空 cell) 更可能是数据行,
         不因备注列含页码/打印时间等而误杀
      3. 低填充率行用宽松正则匹配
    """
    non_empty = [str(c).strip() for c in row if (c or "").strip()]
    fill_ratio = len(non_empty) / max(len(row), 1)
    row_text = " ".join(str(c) for c in row if c)

    # 强信号: 无论填充率都视为 junk
    if re.match(r"^\s*(合计|总计|累计|汇总)[：:]?", row_text):
        return True

    # 高填充率 -> 更可能是数据行, 不误杀
    if fill_ratio >= 0.6 and len(non_empty) > 3:
        return False

    if re.search(
        r"生成时间|前页|本页合计|合计金额|共\d+页|第\d+页|以下空白|打印时间|page\s*\d+"
        r"|最终解释权|客服电话|免责声明|本文件.*所有|统一客服"
        r"|金额单位[：:元]?|打印操作员|打印日期",
        row_text, re.IGNORECASE
    ):
        return True
    return False


def _is_data_row(row: List[str]) -> bool:
    """行中至少有一个日期或一个金额 → 是数据行。"""
    for cell in row:
        cell = (cell or "").strip()
        if _RE_IS_DATE.search(cell):
            return True
        clean = cell.replace(",", "").replace("¥", "").replace(" ", "")
        if clean and _RE_IS_AMOUNT.match(clean):
            return True
    return False


# ── 管道/画线字符集 (mainframe ASCII-art table) ──
# ── 管道/画线字符集 (mainframe ASCII-art table) ──
PIPE_CHARS = frozenset('|│┃┆┇┊┋')
HLINE_CHARS = frozenset('─━┄┅┈┉—–-')
_ALL_BORDER_CHARS = PIPE_CHARS | HLINE_CHARS | frozenset('+┌┐└┘├┤┬┴┼')

_ALL_BORDER_CHARS = PIPE_CHARS | HLINE_CHARS | frozenset('+┌┐└┘├┤┬┴┼')
