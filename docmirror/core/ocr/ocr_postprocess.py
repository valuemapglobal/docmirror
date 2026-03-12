"""
OCR Post-Processing Correction Engine
======================================

General-purpose, domain-agnostic OCR text correction module.

Correction layers:
    - Amount format fixing (punctuation confusion: colon / semicolon /
      space → decimal point).
    - Date format fixing (tilde / space → hyphen).
    - Digit clean-up (common OCR misrecognition corrections).
    - Domain dictionary correction (glyph confusion, extensible).

Design principles:
    1. Pure functions — no state, no side-effects.
    2. Domain-agnostic — not bound to any specific bank or industry.
    3. Layered correction — fix format first, then content.
    4. Safety first — only correct high-confidence errors.
"""
from __future__ import annotations


import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: Character-level normalisation (safest — no ambiguity)
# ═══════════════════════════════════════════════════════════════════════════════

# Full-width → half-width mapping (common OCR artefact)
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
    """Character-level normalisation: full-width → half-width, NFKC, control character clean-up."""
    # Unicode NFKC normalisation
    text = unicodedata.normalize("NFKC", text)
    # Full-width digits/letters → half-width
    text = text.translate(_FULLWIDTH_MAP)
    # Zero-width / control characters
    text = re.sub(r"[\u200b-\u200f\u2028-\u202f\ufeff]", "", text)
    # Collapse multiple whitespace characters
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: Amount format fixing
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-compiled regex patterns (one-time compilation)
_AMOUNT_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # "-230: 43" → "-230.43" (colon + space → decimal point)
    (re.compile(r"([+-]?\d[\d,]*): (\d{2})\b"), r"\1.\2", "colon_space→dot"),
    # "132.995:40" → "132,995.40" (colon → decimal, fix thousands separator)
    (re.compile(r"(\d{1,3})\.(\d{3}):(\d{2})\b"), r"\1,\2.\3", "dot_colon→comma_dot"),
    # "-15;324.55" → "-15,324.55" (semicolon → thousands comma)
    (re.compile(r"([+-]?\d{1,3});(\d{3}[.\d]*)"), r"\1,\2", "semicolon→comma"),
    # "15,458-75" → "15,458.75" (hyphen in decimal position → decimal point)
    (re.compile(r"(\d{3})-(\d{2})\b"), r"\1.\2", "hyphen→decimal"),
    # "-3. 290. 46" → "-3,290.46" (spaced dots → thousands separator)
    (re.compile(r"(\d)\. (\d{3})\. (\d{2})\b"), r"\1,\2.\3", "spaced_dots→amount"),
    # Variant of spaced dots: "4. 088. 31"
    (re.compile(r"(\d)\. (\d{3})\. (\d{2})"), r"\1,\2.\3", "spaced_dots_v2"),
    # ".4,088.31" → "4,088.31" (spurious leading dot)
    (re.compile(r"^\.(\d{1,3},\d{3}\.\d{2})"), r"\1", "leading_dot"),
    # "+4;400.00" → "+4,400.00"
    (re.compile(r"([+-]?\d{1,3});(\d{3}\.\d{2})"), r"\1,\2", "semicolon_amount"),
    # Spurious space in amount: "1, 234. 56" → "1,234.56"
    (re.compile(r"(\d), (\d{3})"), r"\1,\2", "comma_space"),
    (re.compile(r"(\d)\. (\d{2})\b"), r"\1.\2", "dot_space_decimal"),
]


def fix_amount_format(text: str) -> str:
    """Fix punctuation confusion in OCR-recognised amounts."""
    for pattern, replacement, _name in _AMOUNT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3: Date format fixing
# ═══════════════════════════════════════════════════════════════════════════════

_DATE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "2024~08-05" → "2024-08-05" (tilde → hyphen)
    (re.compile(r"(\d{4})[~～](\d{2})[-~～]?(\d{2})"), r"\1-\2-\3"),
    # "2024 -08-05" → "2024-08-05" (space + dash variants)
    (re.compile(r"(\d{4})\s*[-–—]\s*(\d{2})\s*[-–—]\s*(\d{2})"), r"\1-\2-\3"),
    # "2024.08.05" → "2024-08-05" (dot-separated date)
    (re.compile(r"(\d{4})\.(\d{2})\.(\d{2})"), r"\1-\2-\3"),
    # "2024/08/05" is kept as-is (valid format)
]


def fix_date_format(text: str) -> str:
    """Fix FormatError in OCR Date."""
    for pattern, replacement in _DATE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4: Domain dictionary correction (generic version)
# ═══════════════════════════════════════════════════════════════════════════════

# Generic high-frequency OCR glyph-confusion dictionary
# key: incorrect form, value: correct form
# Design: only includes high-frequency unambiguous corrections
_GENERIC_CORRECTIONS: Dict[str, str] = {
    # ── Account types (banking, generic) ──
    "活川": "活期", "活圳": "活期", "活助": "活期", "活斯": "活期",
    "活州": "活期", "活朋": "活期",
    "定册": "定期", "定朋": "定期",

    # ── Payment channels (generic) ──
    "快提支付": "快捷支付", "块捷支付": "快捷支付",
    "快措支付": "快捷支付", "快据支付": "快捷支付",

    # ── Transaction types (generic) ──
    "转帐": "转账", "转帖": "转账",
    "汇入汇": "汇入", "他行汇人": "他行汇入",
    "跨行转人": "跨行转入", "跨行转人账": "跨行转入",
    "网上银行": "网上银行",  # keep (already correct)

    # ── Currency / general ──
    "人民帀": "人民币", "人民巾": "人民币",
    "借记卞": "借记卡", "借记下": "借记卡",

    # ── Common verb confusions ──
    "消赀": "消费", "消贵": "消费",
    "还歉": "还款",
}

# Payment company name correction (Generic format)
_COMPANY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "富发支付" -> "富友支付" (character confusion)
    (re.compile(r"富发支付"), "富友支付"),
    # "高友支付" -> "富友支付" (character confusion)
    (re.compile(r"高友支付"), "富友支付"),
    # "通联支忖" → "通联支付"
    (re.compile(r"支忖"), "支付"),
    (re.compile(r"支村"), "支付"),
]


def fix_domain_terms(text: str) -> str:
    """Fix OCR glyph confusion in domain-specific terms."""
    # Dictionary replacement
    for wrong, correct in _GENERIC_CORRECTIONS.items():
        if wrong in text:
            text = text.replace(wrong, correct)

    # Regex pattern replacement
    for pattern, replacement in _COMPANY_PATTERNS:
        text = pattern.sub(replacement, text)

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 5: Digit noise clean-up (fix pure digit strings)
# ═══════════════════════════════════════════════════════════════════════════════

_DIGIT_CLEANUP: List[Tuple[re.Pattern, str]] = [
    # "00000," → "00000" (trailing comma)
    (re.compile(r"(\d{5}),\s*$"), r"\1"),
    # "00:00002+" → cannot be fixed, mark as low confidence (no replacement)
]


def fix_digit_noise(text: str) -> str:
    """Clean OCR noise in pure digit strings."""
    for pattern, replacement in _DIGIT_CLEANUP:
        text = pattern.sub(replacement, text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 6: Levenshtein dictionary correction for standard keys
# ═══════════════════════════════════════════════════════════════════════════════

# Standard, high-value document keys (Property, Business License, etc.)
_STANDARD_KEYS = [
    "不动产单元号", "权利类型", "权利性质", "用途", "面积", "使用期限",
    "权利其他状况", "附记", "坐落", "权利人", "共有情况", "法定代表人",
    "注册资本", "成立日期", "营业期限", "经营范围", "统一社会信用代码",
    "宗地面积", "房屋结构", "建筑面积"
]

def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def fix_domain_keys(text: str) -> str:
    """Fuzzy match and correct standard document keys using Levenshtein distance."""
    # We only correct if the text is short enough to realistically be a key (or key+value)
    # If the text is very long (like a full paragraph of business scope), skip it.
    if len(text) > 25:
        return text
    
    # Try to find if this string contains a misspelled standard key
    for standard_key in _STANDARD_KEYS:
        key_len = len(standard_key)
        # If the text is exactly the key length (or very close), just compare directly
        if abs(len(text) - key_len) <= 1:
            if _levenshtein_distance(text, standard_key) <= 1:
                return standard_key
        
        # If the text contains the key as a prefix (like "执利类型：xxx")
        # Check the first N characters
        if len(text) > key_len:
            prefix = text[:key_len]
            if _levenshtein_distance(prefix, standard_key) <= 1:
                return standard_key + text[key_len:]
                
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 7: Alphanumeric Substitution (Heuristic Context Constraints)
# ═══════════════════════════════════════════════════════════════════════════════

# Fix typical character/digit confusions based on surrounding context.
# E.g., '0' inside a word should be 'O', 'O' inside a number should be '0'.
_ALPHANUM_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # 0 vs O/o
    # 'O' or 'o' surrounded by digits becomes '0'
    (re.compile(r'(?<=\d)[Oo](?=\d)'), '0', 'O_in_digits'),
    # 'O' or 'o' after a currency symbol or trailing a number: "￥10O" -> "￥100", "50o" -> "500"
    (re.compile(r'(?<=\d)[Oo](?P<end>[.,\s]|$)'), r'0\g<end>', 'O_at_end_of_digits'),
    # '0' surrounded by letters becomes 'O'
    (re.compile(r'(?<=[a-zA-Z])0(?=[a-zA-Z])'), 'O', '0_in_letters'),
    
    # 1 vs I/l
    # 'l' or 'I' surrounded by digits becomes '1'
    (re.compile(r'(?<=\d)[Il](?=\d)'), '1', 'I_in_digits'),
    # '1' surrounded by letters becomes 'l'
    (re.compile(r'(?<=[a-z])1(?=[a-z])'), 'l', '1_in_letters'),
    (re.compile(r'(?<=[A-Z])1(?=[A-Z])'), 'I', '1_in_upper_letters'),
    
    # 5 vs S
    # 'S' or 's' surrounded by numbers
    (re.compile(r'(?<=\d)[Ss](?=\d)'), '5', 'S_in_digits'),
    # '5' surrounded by letters
    (re.compile(r'(?<=[a-zA-Z])5(?=[a-zA-Z])'), 'S', '5_in_letters'),
    
    # Currency bounds: 'S' right before digits (often from $ or 5)
    (re.compile(r'^S(?=\d{2,})'), '5', 'S_at_start_of_digits'),
]

def fix_alphanumeric_confusion(text: str) -> str:
    """Fix common OCR confusions (0/O, 1/l/I, 5/S) using surrounding context constraints."""
    for pattern, replacement, _ in _ALPHANUM_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Unified entry point: full-pipeline post-processing
# ═══════════════════════════════════════════════════════════════════════════════

def postprocess_ocr_text(text: str) -> str:
    """OCR text full pipeline Post-processing.

    Layered execution:
        L1: Character-level cleaning (Full-width to Half-width, NFKC)
        L2: Amount format fix
        L3: Date format fix
        L4: Domain dictionary correction
        L5: Digit noise clean
        L6: Fuzzy key correction
        L7: Alphanumeric heuristic constraint
    """
    if not text or not text.strip():
        return text

    text = normalize_chars(text)      # L1
    text = fix_amount_format(text)    # L2
    text = fix_date_format(text)      # L3
    text = fix_domain_terms(text)     # L4
    text = fix_alphanumeric_confusion(text) # L7
    text = fix_digit_noise(text)      # L5
    text = fix_domain_keys(text)      # L6

    return text


def postprocess_table(
    table: List[List[str]],
) -> List[List[str]]:
    """Apply OCR post-processing to every cell in a table.

    Args:
        table: Table data (2-D list of strings).

    Returns:
        Corrected table.
    """
    return [
        [postprocess_ocr_text(cell) if isinstance(cell, str) else cell for cell in row]
        for row in table
    ]


def postprocess_ocr_result(
    result: Optional[dict],
) -> Optional[dict]:
    """Apply post-processing to the full result from ``analyze_scanned_page()``.

    Args:
        result: ``{'table': [[...]], 'header_text': str, 'footer_text': str}``

    Returns:
        Corrected result (modified in-place).
    """
    if not result:
        return result

    # Correct table cells
    if "table" in result and result["table"]:
        result["table"] = postprocess_table(result["table"])

    # Correct multiple tables
    if "tables" in result and result["tables"]:
        result["tables"] = [postprocess_table(t) for t in result["tables"]]

    # Correct header / footer text
    if "header_text" in result:
        result["header_text"] = postprocess_ocr_text(result["header_text"])
    if "footer_text" in result:
        result["footer_text"] = postprocess_ocr_text(result["footer_text"])

    return result
