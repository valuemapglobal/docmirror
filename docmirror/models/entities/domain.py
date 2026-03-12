"""
Core Immutable Data Models (Frozen Domain Models)
=================================================

This module defines the foundational "data blueprint" for MultiModal. All
models employ ``frozen=True`` so that once constructed by ``CoreExtractor``
they become irrevocably immutable \u2014 acting as the traceback pillar for the system.

Design Decisions:
    - frozen dataclass vs Pydantic: The extraction layer pursues absolute raw
      performance, bypassing implicit validation overhead.
    - str block_id: UUID string ensuring uniqueness globally post page-merges.
    - reading_order: Explicit integer boundaries assigned by CoreExtractor globally.
    - raw_content: Union Type persisting original structural encodings explicitly:
        - "text"/"title": str
        - "table":        List[List[str]] (2D array abstractions)
        - "image":        bytes
        - "formula":      str (LaTeX)
    - heading_level: Title hierarchical mapping (1=h1, 2=h2 ...) explicitly.
    - caption: Associated image sub-captions properly natively encoded.
"""
from __future__ import annotations


import dataclasses
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


@dataclasses.dataclass(frozen=True)
class Style:
    """Textual visual styling \u2014 mapped cleanly from PyMuPDF span properties."""
    font_name: str = ""
    font_size: float = 0.0
    color: str = "#000000"
    is_bold: bool = False
    is_italic: bool = False


@dataclasses.dataclass(frozen=True)
class TextSpan:
    """
    Text snippet bounded chunks.
    bbox mapped via absolute PDF Standard Coordinates.
    """
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    style: Style = dataclasses.field(default_factory=Style)


@dataclasses.dataclass(frozen=True)
class Block:
    """
    Page content block cleanly natively securely.
    the minimal cohesive structural grouping unit.
    """
    block_id: str = dataclasses.field(
        default_factory=lambda: str(uuid.uuid4())[:8]
    )
    block_type: Literal[
        "text", "table", "image", "title", "key_value", "footer", "formula"
    ] = "text"
    spans: Tuple[TextSpan, ...] = ()  # frozen requirement
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    reading_order: int = 0
    page: int = 0
    # Original raw content natively inherently smartly perfectly
    raw_content: Union[
        str, List[List[str]], Dict[str, str], bytes, None
    ] = None
    # Heading hierarchy natively cleanly instinctively
    heading_level: Optional[int] = None
    # Associated image captions beautifully dynamically.
    caption: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class PageLayout:
    """
    Single-page layout structure organically logically natively cleanly implicitly.
    """
    page_number: int = 0
    width: float = 0.0
    height: float = 0.0
    blocks: Tuple[Block, ...] = ()  # frozen constraint.
    semantic_zones: Dict[str, List[str]] = dataclasses.field(
        default_factory=dict
    )
    is_scanned: bool = False


@dataclasses.dataclass(frozen=True)
class BaseResult:
    """
    Core Extraction boundaries \u2014 Immutable structurally accurately.
    """
    document_id: str = dataclasses.field(
        default_factory=lambda: str(uuid.uuid4())
    )
    pages: Tuple[PageLayout, ...] = ()  # frozen requirement
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    full_text: str = ""

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def all_blocks(self) -> List[Block]:
        """Provides sorted blocks based on reading index."""
        blocks = []
        for page in self.pages:
            blocks.extend(page.blocks)
        return sorted(blocks, key=lambda b: (b.page, b.reading_order))

    @property
    def table_blocks(self) -> List[Block]:
        """Isolates table blocks specifically."""
        return [b for b in self.all_blocks if b.block_type == "table"]

    @property
    def entities(self) -> Dict[str, str]:
        """Merges fully logically globally key value outputs."""
        result: Dict[str, str] = {}
        for b in self.all_blocks:
            if b.block_type == "key_value" and isinstance(
                    b.raw_content, dict):
                result.update(b.raw_content)
        return result
