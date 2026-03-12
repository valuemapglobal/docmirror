"""
Markdown Exporter (OmniDocBench Adapt)
======================================

Convert the BaseResult produced by CoreExtractor into the per-page Markdown File required for OmniDocBench evaluation.

OmniDocBench evaluation process::

    model Parse PDF -> per-page .md -> EvalScript compares GT -> score

Core Map:
    - title  → # / ## / ### (by heading_level)
    - text   → Paragraph (Double newlines separated)
    - table  → Markdown table (header + |---| + rows)
    - formula → $$LaTeX$$
    - key_value / footer / image → Skip (benchmark not evaluated)
"""
from __future__ import annotations


import logging
import re
import unicodedata
from typing import List, Optional

from docmirror.models.domain import BaseResult, Block, PageLayout

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def export_document(result: BaseResult) -> List[str]:
    """Convert the entire BaseResult into a Markdown List split by page.

    Args:
        result: ImmutableExtractResult produced by CoreExtractor.

    Returns:
        List[str]: Each element is a page of Markdown text.
        Index 0 corresponds to the first page.
    """
    return [export_page(page) for page in result.pages]


def export_page(page: PageLayout) -> str:
    """Convert a single page PageLayout into a Markdown string.

    Blocks are rendered sequentially after sorting by reading_order.
    Adjacent blocks are separated by double newlines (Markdown ParagraphSeparator).

    Args:
        page: Single page layout structure.

    Returns:
        Complete Markdown string.
    """
    if not page.blocks:
        return ""

    sorted_blocks = sorted(page.blocks, key=lambda b: b.reading_order)
    parts: List[str] = []

    for block in sorted_blocks:
        rendered = _render_block(block)
        if rendered is not None:
            parts.append(rendered)

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Render by Type
# ═══════════════════════════════════════════════════════════════════════════════


def _render_block(block: Block) -> Optional[str]:
    """Dispatch render based on block_type.

    Returns:
        Rendered Markdown snippet, or None indicates Skip.
    """
    renderer = _RENDERERS.get(block.block_type)
    if renderer is None:
        return None
    return renderer(block)


def _render_title(block: Block) -> Optional[str]:
    """Title -> # level."""
    text = _get_text(block)
    if not text:
        return None

    level = block.heading_level or 1
    prefix = "#" * min(level, 6)
    return f"{prefix} {text}"


def _render_text(block: Block) -> Optional[str]:
    """Body text Paragraph -> plain text."""
    text = _get_text(block)
    return text if text else None


def _render_table(block: Block) -> Optional[str]:
    """Table → Markdown table。

    raw_content Format: List[List[str]]
    The first row is treated as header, subsequent rows are data.
    If there is only one row, output as a header-only table.
    """
    rows = block.raw_content
    if not rows or not isinstance(rows, list):
        return None

    # Clean: ensure each cell is a string
    clean_rows: List[List[str]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        clean_rows.append([_clean_cell(c) for c in row])

    if not clean_rows:
        return None

    # Unify column count (take max col count)
    max_cols = max(len(r) for r in clean_rows)
    for row in clean_rows:
        while len(row) < max_cols:
            row.append("")

    # Render
    header = clean_rows[0]
    lines: List[str] = []

    # Header row
    lines.append("| " + " | ".join(header) + " |")
    # Separator row
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    # Data rows
    for row in clean_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _render_formula(block: Block) -> Optional[str]:
    """Display formula → $$LaTeX$$。"""
    latex = _get_text(block)
    if not latex:
        return None

    # Remove potentially existing $ delimiters
    latex = latex.strip()
    if latex.startswith("$$") and latex.endswith("$$"):
        return latex
    if latex.startswith("$") and latex.endswith("$") and not latex.startswith("$$"):
        latex = latex[1:-1]

    return f"$$\n{latex}\n$$"


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════


def _get_text(block: Block) -> str:
    """Extract text from Block.

    Prefer to extract from raw_content (if str),
    Otherwise concatenate from spans.
    """
    if isinstance(block.raw_content, str):
        return _normalize_text(block.raw_content)

    # Concatenate from spans
    if block.spans:
        return _normalize_text(" ".join(s.text for s in block.spans))

    return ""


def _normalize_text(text: str) -> str:
    """Text normalization: NFC + remove redundant whitespace."""
    text = unicodedata.normalize("NFC", text)
    # Merge multiple spaces/tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Remove leading and trailing whitespace
    text = text.strip()
    return text


def _clean_cell(value) -> str:
    """Clean Table cell value."""
    if value is None:
        return ""
    s = str(value).strip()
    # Pipe char will break Markdown table syntax
    s = s.replace("|", "\\|")
    # Newline merging
    s = s.replace("\n", " ")
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Renderer Registry
# ═══════════════════════════════════════════════════════════════════════════════

_RENDERERS = {
    "title": _render_title,
    "text": _render_text,
    "table": _render_table,
    "formula": _render_formula,
    # Skip types below
    "key_value": None,
    "footer": None,
    "image": None,
}
