"""
Benchmark Metrics
==================

Pure-Python implementations of document parsing evaluation metrics.
No GPU or external dependencies required.

Metrics:
    - CER (Character Error Rate): Levenshtein-distance-based.
    - TEDS (Tree Edit Distance Similarity): for table structure evaluation.
    - Reading Order Accuracy: Kendall tau-based correlation.
"""

from __future__ import annotations

import unicodedata
from typing import List, Optional, Sequence


# ═══════════════════════════════════════════════════════════════════════════════
# CER — Character Error Rate
# ═══════════════════════════════════════════════════════════════════════════════

def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings.

    Uses the standard dynamic-programming approach with O(min(m, n))
    space via a rolling single-row buffer.
    """
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)

    if not s2:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev[j + 1] + 1
            delete = curr[j] + 1
            replace = prev[j] + (0 if c1 == c2 else 1)
            curr.append(min(insert, delete, replace))
        prev = curr

    return prev[-1]


def compute_cer(pred: str, gt: str, normalize: bool = True) -> float:
    """Compute Character Error Rate.

    Args:
        pred: Predicted text string.
        gt: Ground-truth text string.
        normalize: If True, apply NFKC normalization before comparison.

    Returns:
        CER value in [0.0, 1.0].  0.0 = perfect match.
        Returns 0.0 when both strings are empty.
    """
    if normalize:
        pred = unicodedata.normalize("NFKC", pred)
        gt = unicodedata.normalize("NFKC", gt)

    if not gt:
        return 0.0 if not pred else 1.0

    distance = _levenshtein(pred, gt)
    return min(1.0, distance / len(gt))


# ═══════════════════════════════════════════════════════════════════════════════
# TEDS — Tree Edit Distance Similarity
# ═══════════════════════════════════════════════════════════════════════════════

def _table_to_tree_str(table: List[List[str]]) -> str:
    """Convert a 2D table to a canonical tree-like string for comparison.

    The tree structure is:
        TABLE( ROW( CELL(text) CELL(text) ) ROW( ... ) )

    This allows tree edit distance to capture both structural and
    content differences.
    """
    if not table:
        return "TABLE()"

    parts = []
    for row in table:
        cells = " ".join(
            f"CELL({str(c).strip()})" for c in (row if row else [""])
        )
        parts.append(f"ROW({cells})")

    return f"TABLE({' '.join(parts)})"


def compute_teds(
    pred_table: List[List[str]],
    gt_table: List[List[str]],
) -> float:
    """Compute Tree Edit Distance Similarity for tables.

    Converts tables to canonical tree strings and uses Levenshtein
    distance as a proxy for tree edit distance.

    Args:
        pred_table: Predicted table (list of row lists).
        gt_table: Ground truth table (list of row lists).

    Returns:
        TEDS score in [0.0, 1.0].  1.0 = perfect structural match.
    """
    pred_str = _table_to_tree_str(pred_table)
    gt_str = _table_to_tree_str(gt_table)

    if not gt_str and not pred_str:
        return 1.0
    if not gt_str or not pred_str:
        return 0.0

    distance = _levenshtein(pred_str, gt_str)
    max_len = max(len(pred_str), len(gt_str))
    return max(0.0, 1.0 - distance / max_len)


# ═══════════════════════════════════════════════════════════════════════════════
# Reading Order Accuracy (Kendall tau)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_reading_order_accuracy(
    pred_order: Sequence[int],
    gt_order: Sequence[int],
) -> float:
    """Compute reading order accuracy using Kendall tau correlation.

    Both sequences must contain the same set of zone indices.

    Args:
        pred_order: Predicted reading order (list of zone indices).
        gt_order: Ground truth reading order (list of zone indices).

    Returns:
        Accuracy in [0.0, 1.0].  1.0 = identical ordering.
    """
    if len(pred_order) != len(gt_order):
        return 0.0
    if len(pred_order) <= 1:
        return 1.0

    n = len(pred_order)

    # Build rank map from gt
    rank_map = {v: i for i, v in enumerate(gt_order)}
    try:
        pred_ranks = [rank_map[v] for v in pred_order]
    except KeyError:
        return 0.0  # different element sets

    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pred_ranks[i] < pred_ranks[j]:
                concordant += 1
            else:
                discordant += 1

    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 1.0

    # Kendall tau: (concordant - discordant) / total_pairs ∈ [-1, 1]
    # Normalise to [0, 1]
    tau = (concordant - discordant) / total_pairs
    return (tau + 1.0) / 2.0
