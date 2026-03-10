"""
核心不可变数据模型 (Frozen Domain Models)
==========================================

本模块定义了 MultiModal 的"数据地基"。所有模型使用 ``frozen=True``，
一旦由 ``CoreExtractor`` 生成便不可修改 — 这是整个系统"可追溯"的基石。

设计决策:
    - frozen dataclass 而非 Pydantic: 提取层追求极致性能，避免验证开销。
    - str block_id: UUID 字符串，保证跨页合并后仍全局唯一。
    - reading_order: 显式整数，由 CoreExtractor 在全局分析阶段赋值。
    - raw_content: Union 类型，按 block_type 存储原始内容:
        - "text"/"title": str
        - "table":        List[List[str]] (二维数组)
        - "image":        bytes
        - "formula":      str (LaTeX)
    - heading_level: 标题层级 (1=h1, 2=h2, 3=h3)，仅 title Block 有值。
    - caption: 关联的图注文字，仅 image Block 有值。
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


@dataclasses.dataclass(frozen=True)
class Style:
    """文本视觉样式 — 由 PyMuPDF 的 span 属性提取。"""
    font_name: str = ""
    font_size: float = 0.0
    color: str = "#000000"
    is_bold: bool = False
    is_italic: bool = False


@dataclasses.dataclass(frozen=True)
class TextSpan:
    """
    文本片段 — 同一 Block 内具有相同样式的连续文字。

    bbox 使用 PDF 标准坐标 (x0, y0, x1, y1)，
    y 轴向下增长，单位为 pt (1/72 inch)。
    """
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    style: Style = dataclasses.field(default_factory=Style)


@dataclasses.dataclass(frozen=True)
class Block:
    """
    页面内容块 — PDF 文档的最小结构单元。

    每个 Block 代表一个语义完整的内容区域:
    表格、标题、正文段落、图像或公式。
    """
    block_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4())[:8])
    block_type: Literal["text", "table", "image", "title", "key_value", "footer", "formula"] = "text"
    spans: Tuple[TextSpan, ...] = ()  # frozen 需要 tuple
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    reading_order: int = 0
    page: int = 0
    # 原始内容 — 按类型存储
    raw_content: Union[str, List[List[str]], Dict[str, str], bytes, None] = None
    # 标题层级 (1=h1, 2=h2, 3=h3)，仅 title Block 有值
    heading_level: Optional[int] = None
    # 图注文字，仅 image Block 有值
    caption: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class PageLayout:
    """
    单页版面结构 — 包含所有 Block 和语义区域划分。

    semantic_zones 示例:
        {"header": ["blk_a1"], "body": ["blk_b2", "blk_b3"], "footer": ["blk_c4"]}
    """
    page_number: int = 0
    width: float = 0.0
    height: float = 0.0
    blocks: Tuple[Block, ...] = ()  # frozen 需要 tuple
    semantic_zones: Dict[str, List[str]] = dataclasses.field(default_factory=dict)
    is_scanned: bool = False


@dataclasses.dataclass(frozen=True)
class BaseResult:
    """
    核心提取结果 — 不可变。

    这是 CoreExtractor 的唯一输出，代表对 PDF 最原始、最客观的结构化描述。
    一旦生成便不可修改，所有后续处理均作为"增强"存在于 EnhancedResult 中。
    """
    document_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    pages: Tuple[PageLayout, ...] = ()  # frozen 需要 tuple
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    full_text: str = ""

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def all_blocks(self) -> List[Block]:
        """按 reading_order 返回所有页面的 Block。"""
        blocks = []
        for page in self.pages:
            blocks.extend(page.blocks)
        return sorted(blocks, key=lambda b: (b.page, b.reading_order))

    @property
    def table_blocks(self) -> List[Block]:
        """仅返回 table 类型的 Block。"""
        return [b for b in self.all_blocks if b.block_type == "table"]

    @property
    def entities(self) -> Dict[str, str]:
        """合并所有 key_value Block 的数据。"""
        result: Dict[str, str] = {}
        for b in self.all_blocks:
            if b.block_type == "key_value" and isinstance(b.raw_content, dict):
                result.update(b.raw_content)
        return result
