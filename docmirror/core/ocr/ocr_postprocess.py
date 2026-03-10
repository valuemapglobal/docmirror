"""
OCR 后处理纠正引擎 (OCR Post-Processing Correction Engine)
==========================================================

通用、泛化的 OCR 文本纠正模块。
- 金额格式修复 (标点混淆: 冒号/分号/空格 → 小数点)
- 日期格式修复 (波浪号/空格 → 连字符)
- 数字清洗 (常见 OCR 误识别修正)
- 领域词典纠正 (字形混淆修正, 可扩展)

设计原则:
    1. 纯函数, 无状态, 无副作用
    2. 通用泛化 — 不绑定特定银行/行业
    3. 分层修正 — 先修格式, 再修内容
    4. 安全优先 — 只修高置信度的错误
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: 字符级清洗 (最安全, 无歧义)
# ═══════════════════════════════════════════════════════════════════════════════

# 全角 → 半角映射 (OCR 常见)
_FULLWIDTH_MAP = str.maketrans(
    "０１２３４５６７８９"
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
    "，。：；（）【】",
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    ",.：;()[]",
)


def normalize_chars(text: str) -> str:
    """字符级标准化: 全角→半角, NFKC, 控制字符清理。"""
    # Unicode NFKC 标准化
    text = unicodedata.normalize("NFKC", text)
    # 全角数字/字母 → 半角
    text = text.translate(_FULLWIDTH_MAP)
    # 零宽字符/控制字符
    text = re.sub(r"[\u200b-\u200f\u2028-\u202f\ufeff]", "", text)
    # 多余空白合并
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: 金额格式修复
# ═══════════════════════════════════════════════════════════════════════════════

# 编译正则 (一次性)
_AMOUNT_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # "-230: 43" → "-230.43" (冒号+空格 → 小数点)
    (re.compile(r"([+-]?\d[\d,]*): (\d{2})\b"), r"\1.\2", "colon_space→dot"),
    # "132.995:40" → "132,995.40" (冒号 → 小数点, 修复千分位)
    (re.compile(r"(\d{1,3})\.(\d{3}):(\d{2})\b"), r"\1,\2.\3", "dot_colon→comma_dot"),
    # "-15;324.55" → "-15,324.55" (分号 → 逗号千分位)
    (re.compile(r"([+-]?\d{1,3});(\d{3}[.\d]*)"), r"\1,\2", "semicolon→comma"),
    # "15,458-75" → "15,458.75" (减号 in decimal → 小数点)
    (re.compile(r"(\d{3})-(\d{2})\b"), r"\1.\2", "hyphen→decimal"),
    # "-3. 290. 46" → "-3,290.46" (点+空格 → 千分位)
    (re.compile(r"(\d)\. (\d{3})\. (\d{2})\b"), r"\1,\2.\3", "spaced_dots→amount"),
    # "-3. 290. 46" 的变体: "4. 088. 31"
    (re.compile(r"(\d)\. (\d{3})\. (\d{2})"), r"\1,\2.\3", "spaced_dots_v2"),
    # ".4,088.31" → "4,088.31" (前置点号)
    (re.compile(r"^\.(\d{1,3},\d{3}\.\d{2})"), r"\1", "leading_dot"),
    # "+4;400.00" → "+4,400.00"
    (re.compile(r"([+-]?\d{1,3});(\d{3}\.\d{2})"), r"\1,\2", "semicolon_amount"),
    # 金额中多余空格: "1, 234. 56" → "1,234.56"
    (re.compile(r"(\d), (\d{3})"), r"\1,\2", "comma_space"),
    (re.compile(r"(\d)\. (\d{2})\b"), r"\1.\2", "dot_space_decimal"),
]


def fix_amount_format(text: str) -> str:
    """修复 OCR 金额中的标点混淆。"""
    for pattern, replacement, _name in _AMOUNT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3: 日期格式修复
# ═══════════════════════════════════════════════════════════════════════════════

_DATE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "2024~08-05" → "2024-08-05" (波浪号→连字符)
    (re.compile(r"(\d{4})[~～](\d{2})[-~～]?(\d{2})"), r"\1-\2-\3"),
    # "2024 -08-05" → "2024-08-05" (空格+连字符)
    (re.compile(r"(\d{4})\s*[-–—]\s*(\d{2})\s*[-–—]\s*(\d{2})"), r"\1-\2-\3"),
    # "2024.08.05" → "2024-08-05" (点号日期)
    (re.compile(r"(\d{4})\.(\d{2})\.(\d{2})"), r"\1-\2-\3"),
    # "2024/08/05" 保持不变 (合法格式)
]


def fix_date_format(text: str) -> str:
    """修复 OCR 日期中的格式错误。"""
    for pattern, replacement in _DATE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4: 领域词典纠正 (泛化版)
# ═══════════════════════════════════════════════════════════════════════════════

# 通用高频 OCR 字形混淆词典
# key: 错误形式, value: 正确形式
# 设计原则: 只收录 OCR 中高频出现且无歧义的纠正
_GENERIC_CORRECTIONS: Dict[str, str] = {
    # ── 账户类型 (银行通用) ──
    "活川": "活期", "活圳": "活期", "活助": "活期", "活斯": "活期",
    "活州": "活期", "活朋": "活期",
    "定册": "定期", "定朋": "定期",

    # ── 支付渠道 (通用) ──
    "快提支付": "快捷支付", "块捷支付": "快捷支付",
    "快措支付": "快捷支付", "快据支付": "快捷支付",

    # ── 交易类型 (通用) ──
    "转帐": "转账", "转帖": "转账",
    "汇入汇": "汇入", "他行汇人": "他行汇入",
    "跨行转人": "跨行转入", "跨行转人账": "跨行转入",
    "网上银行": "网上银行",  # 保留 (已正确)

    # ── 货币/通用 ──
    "人民帀": "人民币", "人民巾": "人民币",
    "借记卞": "借记卡", "借记下": "借记卡",

    # ── 常见动词混淆 ──
    "消赀": "消费", "消贵": "消费",
    "还歉": "还款",
}

# 支付公司名称纠正 (通用模式: X友 → X友)
_COMPANY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "富发支付" → "富友支付" (发↔友 混淆)
    (re.compile(r"富发支付"), "富友支付"),
    # "高友支付" → "富友支付" (富↔高 混淆)
    (re.compile(r"高友支付"), "富友支付"),
    # "通联支忖" → "通联支付"
    (re.compile(r"支忖"), "支付"),
    (re.compile(r"支村"), "支付"),
]


def fix_domain_terms(text: str) -> str:
    """修复 OCR 领域术语的字形混淆。"""
    # 词典替换
    for wrong, correct in _GENERIC_CORRECTIONS.items():
        if wrong in text:
            text = text.replace(wrong, correct)

    # 正则模式替换
    for pattern, replacement in _COMPANY_PATTERNS:
        text = pattern.sub(replacement, text)

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 5: 数字清洗 (修复纯数字串中的 OCR 错误)
# ═══════════════════════════════════════════════════════════════════════════════

_DIGIT_CLEANUP: List[Tuple[re.Pattern, str]] = [
    # "00000," → "00000" (尾部逗号)
    (re.compile(r"(\d{5}),\s*$"), r"\1"),
    # "00:00002+" → 无法修复, 标记为低置信度 (不做替换)
]


def fix_digit_noise(text: str) -> str:
    """清理数字串中的 OCR 噪声。"""
    for pattern, replacement in _DIGIT_CLEANUP:
        text = pattern.sub(replacement, text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# 统一入口: 全流程后处理
# ═══════════════════════════════════════════════════════════════════════════════

def postprocess_ocr_text(text: str) -> str:
    """OCR 文本全流程后处理。

    分层执行:
        L1: 字符级清洗 (全角→半角, NFKC)
        L2: 金额格式修复
        L3: 日期格式修复
        L4: 领域词典纠正
        L5: 数字噪声清理

    Args:
        text: OCR 原始文本。

    Returns:
        纠正后的文本。
    """
    if not text or not text.strip():
        return text

    text = normalize_chars(text)      # L1
    text = fix_amount_format(text)    # L2
    text = fix_date_format(text)      # L3
    text = fix_domain_terms(text)     # L4
    text = fix_digit_noise(text)      # L5

    return text


def postprocess_table(
    table: List[List[str]],
) -> List[List[str]]:
    """对表格中每个单元格应用 OCR 后处理。

    Args:
        table: 表格数据 (二维字符串列表)。

    Returns:
        纠正后的表格。
    """
    return [
        [postprocess_ocr_text(cell) if isinstance(cell, str) else cell for cell in row]
        for row in table
    ]


def postprocess_ocr_result(
    result: Optional[dict],
) -> Optional[dict]:
    """对 analyze_scanned_page() 的完整返回结果应用后处理。

    Args:
        result: {'table': [[...]], 'header_text': str, 'footer_text': str}

    Returns:
        纠正后的结果 (原地修改)。
    """
    if not result:
        return result

    # 纠正表格
    if "table" in result and result["table"]:
        result["table"] = postprocess_table(result["table"])

    # 纠正多表格
    if "tables" in result and result["tables"]:
        result["tables"] = [postprocess_table(t) for t in result["tables"]]

    # 纠正 header/footer
    if "header_text" in result:
        result["header_text"] = postprocess_ocr_text(result["header_text"])
    if "footer_text" in result:
        result["footer_text"] = postprocess_ocr_text(result["footer_text"])

    return result
