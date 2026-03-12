"""
Text Utilities
==============

General-purpose text processing helpers extracted from layout_analysis.py.

Contents:
    - CJK character detection and CJK-aware string joining.
    - ``normalize_text`` / ``normalize_table`` — Unicode NFKC normalisation,
      full-width → half-width, and newline/whitespace clean-up.
    - ``headers_match`` — fuzzy header-row comparison.
    - ``parse_amount`` — financial amount string normalisation
      (thousands separators, currency symbols, parenthesised negatives,
      CR/DR suffixes).
"""
from __future__ import annotations


import re
import unicodedata
from typing import List


# ═══════════════════════════════════════════════════════════════════════════════
# CJK helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _is_cjk_char(ch: str) -> bool:
    """Return ``True`` if *ch* is a CJK Unified Ideograph."""
    if not ch:
        return False
    cp = ord(ch[0])
    return (0x4E00 <= cp <= 0x9FFF       # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF    # CJK Extension A
            or 0xF900 <= cp <= 0xFAFF)   # CJK Compatibility Ideographs


def _smart_join(left: str, right: str) -> str:
    """CJK-aware concatenation: no space between CJK characters, otherwise
    insert a single space."""
    if not left or not right:
        return left + right
    if _is_cjk_char(left[-1]) or _is_cjk_char(right[0]):
        return left + right
    return left + " " + right


# ═══════════════════════════════════════════════════════════════════════════════
# General utilities
# ═══════════════════════════════════════════════════════════════════════════════

_RE_DATE_COMPACT = re.compile(r"(\d{8})")
_RE_DATE_HYPHEN = re.compile(r"\d{4}-\d{2}-\d{2}")
_RE_TIME = re.compile(r"\d{2}:\d{2}:\d{2}")
_RE_ONLY_CJK = re.compile(r"[^\u4e00-\u9fff]")


def normalize_text(text: str) -> str:
    """Apply Unicode NFKC normalisation followed by newline and whitespace
    clean-up.

    Newline handling is CJK-aware:
        * Between two CJK characters the newline is simply removed (no space).
        * Otherwise the newline is replaced with a single space.
    """
    if not text:
        return text
    
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")

    def is_cjk_char(ch):
        return _is_cjk_char(ch)

    # Process all newline characters with CJK-awareness
    while "\n" in text:
        idx = text.find("\n")
        left_cjk = idx > 0 and _is_cjk_char(text[idx-1])
        right_cjk = idx < len(text) - 1 and _is_cjk_char(text[idx+1])

        if left_cjk and right_cjk: # Both sides are CJK — remove the newline
            text = text[:idx] + text[idx+1:]
        else: # Otherwise replace with a space
            text = text[:idx] + " " + text[idx+1:]

    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_table(table: List[List[str]]) -> List[List[str]]:
    """Apply ``normalize_text`` to every cell in a 2-D table."""
    return [
        [normalize_text(cell) if cell else "" for cell in row]
        for row in table
    ]


def headers_match(base: List[str], candidate: List[str], threshold: float = 0.6) -> bool:
    """Check whether two header rows match by measuring the overlap ratio
    of their non-empty cells.  Returns ``True`` when the overlap is at
    least *threshold* (default 60 %).
    """
    if not base or not candidate:
        return False
    # Inline NFKC normalisation (avoids circular dependency on vocabulary.py)
    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKC", s).strip()
    base_set = {_norm(str(h)) for h in base if str(h).strip()}
    cand_set = {_norm(str(h)) for h in candidate if str(h).strip()}
    if not base_set:
        return False
    overlap = len(base_set & cand_set)
    return overlap / len(base_set) >= threshold


def parse_amount(s: str) -> str:
    """Normalise a financial amount string.

    Supported formats:
      - Thousands separator comma:  ``1,234.56``
      - Currency symbols:           ``¥``, ``$``, ``￥``
      - Parenthesised negative:     ``(1,234.56)`` → ``-1234.56``
      - CR/DR suffixes (banking):   ``1234.56CR`` → ``-1234.56``
        (CR = credit/income, DR = debit/expense)
    """
    s = s.strip()
    if not s:
        return s

    # Parenthesised negative: "(1,234.56)" → "-1234.56"
    neg = False
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
        neg = True

    # CR/DR suffix (banking convention: CR = credit, DR = debit)
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
