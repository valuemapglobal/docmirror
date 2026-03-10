"""
Adapters — 格式适配器层
========================

每个 Adapter 负责:
    1. 将特定格式文件转换为 ``BaseResult``
    2. 可选地直接返回 ``ParserOutput`` (向后兼容)

Adapter 不引入任何业务逻辑 — 业务增强由 Middleware 管线统一处理。
"""

from .pdf import PDFAdapter
from .image import ImageAdapter
from .email import EmailAdapter
from .excel import ExcelAdapter
from .word import WordAdapter
from .ppt import PPTAdapter
from .structured import StructuredAdapter
from .web import WebAdapter

__all__ = [
    "PDFAdapter",
    "ImageAdapter",
    "EmailAdapter",
    "ExcelAdapter",
    "WordAdapter",
    "PPTAdapter",
    "StructuredAdapter",
    "WebAdapter",
]
