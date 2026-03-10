"""
Markdown Exporter (OmniDocBench 适配)
======================================

将 CoreExtractor 产出的 BaseResult 转换为 OmniDocBench 评测所需的
per-page Markdown 文件。

OmniDocBench 评估流程::

    model 解析 PDF → 每页 .md → 评测脚本对比 GT → 分数

核心映射:
    - title  → # / ## / ### (按 heading_level)
    - text   → 段落 (双换行分隔)
    - table  → Markdown table (header + |---| + rows)
    - formula → $$LaTeX$$
    - key_value / footer / image → 跳过 (benchmark 不评测)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Optional

from docmirror.models.domain import BaseResult, Block, PageLayout

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 公共 API
# ═══════════════════════════════════════════════════════════════════════════════


def export_document(result: BaseResult) -> List[str]:
    """将整个 BaseResult 转换为按页分割的 Markdown 列表。

    Args:
        result: CoreExtractor 产出的不可变提取结果。

    Returns:
        List[str]: 每个元素是一页的 Markdown 文本。
        索引 0 对应第一页。
    """
    return [export_page(page) for page in result.pages]


def export_page(page: PageLayout) -> str:
    """将单页 PageLayout 转换为 Markdown 字符串。

    Blocks 按 reading_order 排序后依次渲染。
    相邻块之间用双换行分隔 (Markdown 段落分隔符)。

    Args:
        page: 单页版面结构。

    Returns:
        完整的 Markdown 字符串。
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
# 逐类型渲染
# ═══════════════════════════════════════════════════════════════════════════════


def _render_block(block: Block) -> Optional[str]:
    """根据 block_type 分派渲染。

    Returns:
        渲染后的 Markdown 片段, 或 None 表示跳过。
    """
    renderer = _RENDERERS.get(block.block_type)
    if renderer is None:
        return None
    return renderer(block)


def _render_title(block: Block) -> Optional[str]:
    """标题 → # 层级。"""
    text = _get_text(block)
    if not text:
        return None

    level = block.heading_level or 1
    prefix = "#" * min(level, 6)
    return f"{prefix} {text}"


def _render_text(block: Block) -> Optional[str]:
    """正文段落 → 纯文本。"""
    text = _get_text(block)
    return text if text else None


def _render_table(block: Block) -> Optional[str]:
    """表格 → Markdown table。

    raw_content 格式: List[List[str]]
    第一行视为 header，后续行为 data。
    如果只有一行，也输出为 header-only table。
    """
    rows = block.raw_content
    if not rows or not isinstance(rows, list):
        return None

    # 清洗: 确保每个 cell 都是字符串
    clean_rows: List[List[str]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        clean_rows.append([_clean_cell(c) for c in row])

    if not clean_rows:
        return None

    # 统一列数 (取最大列数)
    max_cols = max(len(r) for r in clean_rows)
    for row in clean_rows:
        while len(row) < max_cols:
            row.append("")

    # 渲染
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

    # 去掉可能已存在的 $ 定界符
    latex = latex.strip()
    if latex.startswith("$$") and latex.endswith("$$"):
        return latex
    if latex.startswith("$") and latex.endswith("$") and not latex.startswith("$$"):
        latex = latex[1:-1]

    return f"$$\n{latex}\n$$"


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


def _get_text(block: Block) -> str:
    """从 Block 中提取文本。

    优先从 raw_content 提取 (如果是 str)，
    否则从 spans 拼接。
    """
    if isinstance(block.raw_content, str):
        return _normalize_text(block.raw_content)

    # 从 spans 拼接
    if block.spans:
        return _normalize_text(" ".join(s.text for s in block.spans))

    return ""


def _normalize_text(text: str) -> str:
    """文本规范化: NFC + 去除多余空白。"""
    text = unicodedata.normalize("NFC", text)
    # 多个空格/制表符合并为单个空格
    text = re.sub(r"[ \t]+", " ", text)
    # 去除首尾空白
    text = text.strip()
    return text


def _clean_cell(value) -> str:
    """清洗表格 cell 值。"""
    if value is None:
        return ""
    s = str(value).strip()
    # 管道符会破坏 Markdown table 语法
    s = s.replace("|", "\\|")
    # 换行合并
    s = s.replace("\n", " ")
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# 渲染器注册表
# ═══════════════════════════════════════════════════════════════════════════════

_RENDERERS = {
    "title": _render_title,
    "text": _render_text,
    "table": _render_table,
    "formula": _render_formula,
    # 以下类型跳过
    "key_value": None,
    "footer": None,
    "image": None,
}
