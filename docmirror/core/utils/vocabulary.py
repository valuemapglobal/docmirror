# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Vocabulary & Row Classifiers
=============================

Vocabulary-driven row classification system extracted from layout_analysis.py.

Contents:
    - ``VOCAB_BY_CATEGORY`` — per-document-type header keyword sets (currently
      only ``BANK_STATEMENT``; extensible to VAT invoices, contracts, etc.).
    - ``KNOWN_HEADER_WORDS`` — union of all category vocab sets
      (backward-compatible shortcut).
    - Row classification functions:
        * ``_score_header_by_vocabulary`` — count how many cells match
          known header words (NFKC + Traditional→Simplified normalised).
        * ``_is_header_cell`` — single-cell heuristic (short text, not a
          date / amount / time / long digit string / currency).
        * ``_is_header_row`` — dual-gate algorithm:
            Gate 1 (definitive reject):  any strong data signal → False.
            Gate 2 (definitive accept):  vocab score ≥ 3 → True.
            Fallback (structural):       ≥ 60 % header-like cells → True.
        * ``_is_junk_row`` — detect page headers, footers, totals.
        * ``_is_data_row`` — detect rows containing a date or an amount.
    - ``PIPE_CHARS`` / ``HLINE_CHARS`` / ``_ALL_BORDER_CHARS`` — character
      sets for identifying mainframe ASCII-art table borders.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import deque
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_RE_IS_DATE = re.compile(r"\d{4}[-/.]?\d{1,2}[-/.]?\d{1,2}|\d{8}")
_RE_IS_AMOUNT = re.compile(r"^[-+]?\d[\d,]*\.?\d*$")
_RE_VALID_DATE = re.compile(r"(?:19|20)\d{2}[-/.]?\d{1,2}[-/.]?\d{1,2}")

# Pre-compiled junk row patterns (C3: avoid per-call re.compile overhead)
_RE_JUNK_STRONG = re.compile(r"^\s*(合计|总计|累计|汇总)[：:]?")
_RE_JUNK_PATTERNS = re.compile(
    r"生成时间|前页|本页合计|合计金额|共\d+页|第\d+页|以下空白|打印时间|page\s*\d+"
    r"|最终解释权|客服电话|免责声明|本文件.*所有|统一客服"
    r"|金额单位[：:元]?|打印操作员|打印日期",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
# Category-specific vocabulary — isolated by document type to prevent
# cross-domain false positives
# ══════════════════════════════════════════════════════════════════════════════

VOCAB_BY_CATEGORY: dict[str, frozenset] = {
    # Bank Statement
    "BANK_STATEMENT": frozenset(
        {
            # Date / time
            "日期",
            "交易日期",
            "记账日期",
            "入账日期",
            "交易时间",
            "交易日",
            # Amount
            "金额",
            "交易金额",
            "发生额",
            "收入金额",
            "支出金额",
            "本次余额",
            # Balance
            "余额",
            "账户余额",
            "结存",
            "可用余额",
            # Description / remarks
            "摘要",
            "摘要内容",
            "交易摘要",
            "用途",
            "附言",
            "备注",
            "交易类型",
            "交易类别",
            "业务摘要",
            # Counterparty information
            "对方户名",
            "对方账号",
            "对手账号",
            "对手户名",
            "对方信息",
            "对方账户",
            "对方名称",
            "对方行名",
            # Sequence / voucher
            "序号",
            "账户序号",
            "账号序号",
            "凭证类型",
            "凭证号码",
            "流水号",
            # Currency / cash–transfer flag
            "币种",
            "钞汇",
            "现/转",
            "现转",
            "钞汇标识",
            "现转标志",
            # Debit / credit direction
            "借贷",
            "借贷状态",
            "支/收",
            "收支",
            "借贷标志",
            # Channel / institution
            "交易渠道",
            "交易机构",
            "交易方式",
            "交易地点",
            # Reversal flags
            "被冲账标识",
            "冲账标识",
            # Transaction reference / remarks
            "交易流水号",
            "交易备注",
        }
    ),
    # Reserved: VAT Invoice (commented out — enable when VAT extraction is implemented)
    # "VAT_INVOICE": frozenset({
    #     "品名", "规格型号", "单位", "数量", "单价", "金额", "税率", "税额",
    #     "价税合计", "购买方", "销售方", "发票代码", "发票号码",
    # }),
    # Reserved: General document
    # "GENERAL": frozenset({...}),
}

# Backward-compatible union of all category vocabularies
KNOWN_HEADER_WORDS: frozenset = frozenset().union(*VOCAB_BY_CATEGORY.values())


# ══════════════════════════════════════════════════════════════════════════════
# Aho-Corasick Multi-Pattern Matcher
# ══════════════════════════════════════════════════════════════════════════════


class AhoCorasick:
    """Pure-Python Aho-Corasick automaton for O(L) multi-pattern matching.

    Build once from a set of patterns, then call ``search()`` or
    ``search_longest_non_overlapping()`` on any text string.
    """

    __slots__ = ("_goto", "_fail", "_output", "_patterns")

    def __init__(self, patterns: Iterable[str]):
        self._patterns = list(patterns)
        self._goto: list[dict[str, int]] = [{}]  # state 0 = root
        self._output: list[list[int]] = [[]]  # output[state] = [pattern_idx]
        self._fail: list[int] = [0]

        # ── Phase 1: Build Trie (goto function) ──
        for pi, pat in enumerate(self._patterns):
            state = 0
            for ch in pat:
                if ch not in self._goto[state]:
                    new_state = len(self._goto)
                    self._goto.append({})
                    self._output.append([])
                    self._fail.append(0)
                    self._goto[state][ch] = new_state
                state = self._goto[state][ch]
            self._output[state].append(pi)

        # ── Phase 2: Build failure links via BFS ──
        q = deque()
        for ch, s in self._goto[0].items():
            self._fail[s] = 0
            q.append(s)

        while q:
            r = q.popleft()
            for ch, s in self._goto[r].items():
                q.append(s)
                state = self._fail[r]
                while state != 0 and ch not in self._goto[state]:
                    state = self._fail[state]
                self._fail[s] = self._goto[state].get(ch, 0)
                if self._fail[s] == s:
                    self._fail[s] = 0
                self._output[s] = self._output[s] + self._output[self._fail[s]]

    def search(self, text: str) -> list[tuple[str, int, int]]:
        """Find all pattern occurrences in *text*.

        Returns:
            List of ``(pattern, start, end)`` tuples, unsorted.
        """
        results = []
        state = 0
        for i, ch in enumerate(text):
            while state != 0 and ch not in self._goto[state]:
                state = self._fail[state]
            state = self._goto[state].get(ch, 0)
            for pi in self._output[state]:
                pat = self._patterns[pi]
                start = i - len(pat) + 1
                results.append((pat, start, i + 1))
        return results

    def search_longest_non_overlapping(
        self,
        text: str,
    ) -> list[tuple[str, int, int]]:
        """Find non-overlapping matches, preferring longest match.

        Greedy left-to-right: among overlapping matches at the same
        start position, keep the longest.  Discard matches whose
        character positions are already claimed.
        """
        all_matches = self.search(text)
        if not all_matches:
            return []

        # Sort by start position, then by length descending (longest first)
        all_matches.sort(key=lambda m: (m[1], -(m[2] - m[1])))

        used = set()
        result = []
        for pat, start, end in all_matches:
            if any(p in used for p in range(start, end)):
                continue
            result.append((pat, start, end))
            used.update(range(start, end))

        return sorted(result, key=lambda m: m[1])


# ── Module-level AC singletons (built once at import time) ──
_AC_ALL = AhoCorasick(KNOWN_HEADER_WORDS)
_AC_BY_CATEGORY: dict[str, AhoCorasick] = {cat: AhoCorasick(words) for cat, words in VOCAB_BY_CATEGORY.items()}


# ── CJK normalisation: NFKC + Traditional → Simplified patches ──
_TRAD_TO_SIMP = str.maketrans(
    {
        "\u6236": "\u6237",  # Traditional 戶 → Simplified 户
        "\u2ea0": "\u6c11",  # CJK radical ⺠ → 民
        "\u8cf3": "\u8d26",
        "\u865f": "\u53f7",
        "\u984d": "\u989d",
        "\u6642": "\u65f6",
        "\u6a5f": "\u673a",
        "\u69cb": "\u6784",
    }
)


def _normalize_for_vocab(s: str) -> str:
    """Apply NFKC + Traditional-to-Simplified patches for vocabulary matching."""
    return unicodedata.normalize("NFKC", s).translate(_TRAD_TO_SIMP)


def _score_header_by_vocabulary(
    row: list[str],
    categories: list[str] | None = None,
) -> int:
    """Count how many cells in *row* match known header keywords.

    Matching is NFKC- and Traditional/Simplified-normalised.

    Args:
        row: List of cell strings (one table row).
        categories: Restrict matching to these category keys
            (e.g. ``["BANK_STATEMENT"]``).  ``None`` uses the full
            ``KNOWN_HEADER_WORDS`` union (backward-compatible).
    """
    vocab = (
        frozenset().union(*(VOCAB_BY_CATEGORY.get(c, frozenset()) for c in categories))
        if categories
        else KNOWN_HEADER_WORDS
    )
    score = 0
    for cell in row:
        text = _normalize_for_vocab(cell or "").strip()
        if text in vocab:
            score += 1
    return score


# Currency-prefixed amount: "RMB 631.20", "USD 100.00"
_RE_CURRENCY_AMOUNT = re.compile(r"^[A-Z]{2,4}\s+[\d,.]+$")
# Bare currency code: "RMB", "USD", "CNY"
_RE_BARE_CURRENCY = re.compile(r"^[A-Z]{2,4}$")

# Pure time format: "08:54:19"
_RE_IS_TIME_ONLY = re.compile(r"^\d{2}:\d{2}:\d{2}$")
# Long digit string (account / transaction numbers, ≥ 10 digits)
_RE_LONG_DIGIT = re.compile(r"^\d{10,}$")


def _is_header_cell(cell: str) -> bool:
    """Heuristic: does a single cell look like a header label?

    Returns ``False`` for cells that contain dates, amounts, times,
    currency codes, or long digit strings.  Also rejects single-char
    cells (too little information) and cells longer than 30 chars
    (likely data content).
    """
    cell = cell.strip()
    if not cell:
        return False
    if _RE_IS_DATE.search(cell):
        return False
    if _RE_IS_AMOUNT.match(cell.replace(",", "").replace("¥", "")):
        return False
    # Time "08:54:19" is data, not a header
    if _RE_IS_TIME_ONLY.match(cell):
        return False
    # Long pure-digit string (account / transaction numbers)
    if _RE_LONG_DIGIT.match(cell):
        return False
    # Currency-prefixed amount: "RMB 631.20"
    if _RE_CURRENCY_AMOUNT.match(cell):
        return False
    # Bare currency code: "RMB", "USD"
    if _RE_BARE_CURRENCY.match(cell):
        return False
    # Single character (e.g. "N", "Y") — insufficient information
    if len(cell) <= 1:
        return False
    if len(cell) > 30:
        return False
    return True


def _is_header_row(row: list[str]) -> bool:
    """Determine whether a row is a header row using a dual-gate algorithm.

    Gate 1 (definitive reject):
        If *any* cell contains a date, amount, time, or long digit string
        the row is certainly data.  These signals have near-100 % precision
        for rejecting header rows.

    Gate 2 (definitive accept):
        ``vocab_score >= 3`` — enough cells match known column-name
        vocabulary to be certain.

    Fallback (structural heuristic):
        ≥ 60 % of non-empty cells pass ``_is_header_cell`` → ``True``.
    """
    non_empty = [c.strip() for c in row if (c or "").strip()]
    if len(non_empty) < 2:
        return False

    # ── Gate 1: strong data signals → definitive reject ──
    for cell in non_empty:
        clean = cell.replace(",", "").replace("¥", "")
        if _RE_IS_DATE.search(cell):  # Date
            return False
        if _RE_IS_AMOUNT.match(clean):  # Amount (decimal number)
            return False
        if _RE_IS_TIME_ONLY.match(cell):  # Time HH:MM:SS
            return False
        if _RE_LONG_DIGIT.match(cell):  # Long digit string
            return False

    # ── Gate 2: vocab hit count ≥ 3 → definitive accept ──
    if _score_header_by_vocabulary(non_empty) >= 3:
        return True

    # ── Fallback: structural heuristic ──
    header_count = sum(1 for c in non_empty if _is_header_cell(c))
    return header_count / len(non_empty) >= 0.6


def _is_junk_row(row: list[str]) -> bool:
    """Detect page headers, footers, totals, and other non-data rows.

    Strategy:
      1. Strong signals (totals like 合计 / 总计) are always junk
         regardless of fill ratio.
      2. High-fill rows (≥ 60 % non-empty and > 3 non-empty cells) are
         more likely data rows — don't discard them just because a remarks
         column contains page numbers or print timestamps.
      3. Low-fill rows are tested against a broad junk-pattern regex.
    """
    non_empty = [str(c).strip() for c in row if (c or "").strip()]
    fill_ratio = len(non_empty) / max(len(row), 1)
    row_text = " ".join(str(c) for c in row if c)

    # Strong signal: always junk regardless of fill ratio
    if _RE_JUNK_STRONG.match(row_text):
        return True

    # High fill ratio → likely a data row — don't discard
    if fill_ratio >= 0.6 and len(non_empty) > 3:
        return False

    if _RE_JUNK_PATTERNS.search(row_text):
        return True
    return False


def _is_data_row(row: list[str]) -> bool:
    """Return ``True`` if the row contains at least one date or one amount."""
    for cell in row:
        cell = (cell or "").strip()
        if _RE_IS_DATE.search(cell):
            return True
        clean = cell.replace(",", "").replace("¥", "").replace(" ", "")
        if clean and _RE_IS_AMOUNT.match(clean):
            return True
    return False


# ── Pipe / line-drawing character sets (mainframe ASCII-art tables) ──
PIPE_CHARS = frozenset("|│┃┆┇┊┋")
HLINE_CHARS = frozenset("─━┄┅┈┉—–-")
_ALL_BORDER_CHARS = PIPE_CHARS | HLINE_CHARS | frozenset("+┌┐└┘├┤┬┴┼")
