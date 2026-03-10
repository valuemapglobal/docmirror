"""
文本工具函数 (Text Utilities)
===============================

从 layout_analysis.py 拆分的通用文本处理函数。
包含 CJK 工具、normalize_text、normalize_table、headers_match、parse_amount 等。
"""

from __future__ import annotations

import re
import unicodedata
from typing import List


# ═══════════════════════════════════════════════════════════════════════════════
# CJK 统一工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _is_cjk_char(ch: str) -> bool:
    """判断单个字符是否为 CJK 统一表意文字。"""
    if not ch:
        return False
    cp = ord(ch[0])
    return (0x4E00 <= cp <= 0x9FFF       # CJK 统一汉字
            or 0x3400 <= cp <= 0x4DBF    # CJK 扩展 A
            or 0xF900 <= cp <= 0xFAFF)   # CJK 兼容汉字


def _smart_join(left: str, right: str) -> str:
    """CJK-aware 拼接: 中文之间无空格, 其他加空格。"""
    if not left or not right:
        return left + right
    if _is_cjk_char(left[-1]) or _is_cjk_char(right[0]):
        return left + right
    return left + " " + right


# ═══════════════════════════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════════════════════════

_RE_DATE_COMPACT = re.compile(r"(\d{8})")
_RE_DATE_HYPHEN = re.compile(r"\d{4}-\d{2}-\d{2}")
_RE_TIME = re.compile(r"\d{2}:\d{2}:\d{2}")
_RE_ONLY_CJK = re.compile(r"[^\u4e00-\u9fff]")


def normalize_text(text: str) -> str:
    """Unicode NFKC + 换行/空格清理。"""
    if not text:
        return text
    
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")

    def is_cjk_char(ch):
        return _is_cjk_char(ch)

    # 处理换行符：中文之间去除换行不加空格，中英/英数字之间加空格
    # 使用 while 循环处理所有换行符
    while "\n" in text:
        idx = text.find("\n")
        left_cjk = idx > 0 and _is_cjk_char(text[idx-1])
        right_cjk = idx < len(text) - 1 and _is_cjk_char(text[idx+1])

        if left_cjk and right_cjk: # Both sides are CJK, remove newline
            text = text[:idx] + text[idx+1:]
        else: # Otherwise, replace with space
            text = text[:idx] + " " + text[idx+1:]

    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_table(table: List[List[str]]) -> List[List[str]]:
    """对整个二维表格执行 normalize_text。"""
    return [
        [normalize_text(cell) if cell else "" for cell in row]
        for row in table
    ]


def headers_match(base: List[str], candidate: List[str], threshold: float = 0.6) -> bool:
    """检查两行表头的匹配程度 (>threshold)。"""
    if not base or not candidate:
        return False
    # 内联 NFKC 归一化 (避免循环依赖 vocabulary.py)
    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKC", s).strip()
    base_set = {_norm(str(h)) for h in base if str(h).strip()}
    cand_set = {_norm(str(h)) for h in candidate if str(h).strip()}
    if not base_set:
        return False
    overlap = len(base_set & cand_set)
    return overlap / len(base_set) >= threshold


def parse_amount(s: str) -> str:
    """金额字符串标准化。

    支持:
      - 千分位逗号: 1,234.56
      - 货币符号: ¥ $ ￥
      - 括号表示负数: (1,234.56) -> -1234.56
      - CR/DR 后缀 (银行借贷): 1234.56CR -> -1234.56
    """
    s = s.strip()
    if not s:
        return s

    # 括号表示负数: "(1,234.56)" -> "-1234.56"
    neg = False
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
        neg = True

    # CR/DR 后缀 (银行: CR=贷方/收入, DR=借方/支出)
    s_upper = s.upper().strip()
    if s_upper.endswith("CR"):
        s = s_upper[:-2]
        neg = True
    elif s_upper.endswith("DR"):
        s = s_upper[:-2]

    s = s.replace(",", "").replace("\u00a5", "").replace("\uffe5", "").replace("$", "").replace(" ", "")
    try:
        val = float(s)
        if neg:
            val = -abs(val)
        return f"{val:.2f}"
    except (ValueError, TypeError):
        return s
