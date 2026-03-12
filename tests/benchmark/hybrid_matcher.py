"""
Hybrid Matcher — OmniDocBench-Style Evaluation
================================================

Implements the hybrid matching algorithm used by OmniDocBench V1.5
for fair evaluation of document parsing results.

Traditional evaluation fails on:
    - Unicode variants (α vs a, ∫ vs \\int)
    - LaTeX semantic equivalences
    - Minor paragraph segmentation differences

This module provides a matching pipeline that handles these cases.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Unicode Normalization
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping of common Unicode→ASCII equivalences for mathematical symbols
_UNICODE_MAP = {
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e",
    "ζ": "z", "η": "h", "θ": "th", "ι": "i", "κ": "k",
    "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "π": "pi",
    "ρ": "r", "σ": "s", "τ": "t", "υ": "u", "φ": "ph",
    "χ": "ch", "ψ": "ps", "ω": "w",
    "∫": "\\int", "∑": "\\sum", "∏": "\\prod",
    "∞": "\\infty", "√": "\\sqrt", "≠": "\\neq",
    "≤": "\\leq", "≥": "\\geq", "×": "\\times",
    "÷": "\\div", "±": "\\pm", "∓": "\\mp",
    "→": "\\to", "←": "\\leftarrow",
    "⊂": "\\subset", "⊃": "\\supset",
    "∈": "\\in", "∉": "\\notin",
    "∅": "\\emptyset",
    "\u2013": "-", "\u2014": "-",  # en-dash, em-dash
    "\u2018": "'", "\u2019": "'",  # smart quotes
    "\u201c": '"', "\u201d": '"',
}


def normalize_unicode(text: str) -> str:
    """Apply NFKC normalization + custom mathematical symbol mapping.

    Args:
        text: Input string.

    Returns:
        Normalized string where Unicode math symbols are replaced
        with their LaTeX or ASCII equivalents.
    """
    text = unicodedata.normalize("NFKC", text)

    # Apply custom mappings
    for src, dst in _UNICODE_MAP.items():
        text = text.replace(src, dst)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX Semantic Equivalence
# ═══════════════════════════════════════════════════════════════════════════════

# Pairs of LaTeX commands that are semantically equivalent
_LATEX_EQUIVALENCES = [
    (r"\frac", r"\dfrac"),
    (r"\tfrac", r"\frac"),
    (r"\bm", r"\mathbf"),
    (r"\boldsymbol", r"\mathbf"),
    (r"\lvert", r"|"),
    (r"\rvert", r"|"),
    (r"\left(", "("),
    (r"\right)", ")"),
    (r"\left[", "["),
    (r"\right]", "]"),
    (r"\left\{", r"\{"),
    (r"\right\}", r"\}"),
]


def _normalize_latex(text: str) -> str:
    """Normalize LaTeX expressions to a canonical form.

    Applies equivalence substitutions and strips purely stylistic commands.
    """
    for src, dst in _LATEX_EQUIVALENCES:
        text = text.replace(src, dst)

    # Remove \displaystyle and \textstyle (purely visual)
    text = re.sub(r"\\(displaystyle|textstyle)\s*", "", text)

    # Collapse spaces within math expressions
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_latex_equivalent(pred: str, gt: str) -> bool:
    """Check if two LaTeX strings are semantically equivalent.

    Args:
        pred: Predicted LaTeX string.
        gt: Ground truth LaTeX string.

    Returns:
        True if the strings are equivalent after normalization.
    """
    return _normalize_latex(pred) == _normalize_latex(gt)


# ═══════════════════════════════════════════════════════════════════════════════
# Fuzzy Segment Matching
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_char_overlap(s1: str, s2: str) -> float:
    """Compute character overlap ratio between two strings.

    Uses the Sørensen–Dice coefficient on character bigrams for
    position-independent similarity.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    def _bigrams(s: str):
        return set(s[i:i+2] for i in range(len(s) - 1)) if len(s) > 1 else {s}

    bg1 = _bigrams(s1)
    bg2 = _bigrams(s2)

    if not bg1 and not bg2:
        return 1.0

    overlap = len(bg1 & bg2)
    return 2 * overlap / (len(bg1) + len(bg2))


def fuzzy_segment_match(
    pred: str, gt: str, threshold: float = 0.85
) -> bool:
    """Check if two text segments match with allowance for segmentation differences.

    Handles cases where the model splits or merges paragraphs differently
    from the ground truth.

    Args:
        pred: Predicted text.
        gt: Ground truth text.
        threshold: Minimum Dice coefficient for a match.

    Returns:
        True if the segments are sufficiently similar.
    """
    pred_norm = normalize_unicode(pred).lower()
    gt_norm = normalize_unicode(gt).lower()

    # Exact match after normalization
    if pred_norm == gt_norm:
        return True

    # Fuzzy match
    return _compute_char_overlap(pred_norm, gt_norm) >= threshold


# ═══════════════════════════════════════════════════════════════════════════════
# Full Hybrid Match
# ═══════════════════════════════════════════════════════════════════════════════

def hybrid_match(
    pred: str,
    gt: str,
    fuzzy_threshold: float = 0.85,
) -> bool:
    """OmniDocBench-style hybrid matching.

    Tries matching in order of strictness:
        1. Unicode normalization + exact match
        2. LaTeX semantic equivalence
        3. Fuzzy segment match (Dice coefficient)

    Args:
        pred: Predicted output string.
        gt: Ground truth string.
        fuzzy_threshold: Threshold for fuzzy matching (default 0.85).

    Returns:
        True if the prediction matches the ground truth
        by any of the matching criteria.
    """
    # Step 1: Unicode normalization
    pred_n = normalize_unicode(pred)
    gt_n = normalize_unicode(gt)

    if pred_n == gt_n:
        return True

    # Step 2: LaTeX equivalence
    if is_latex_equivalent(pred_n, gt_n):
        return True

    # Step 3: Fuzzy segment match
    if fuzzy_segment_match(pred_n, gt_n, threshold=fuzzy_threshold):
        return True

    return False
