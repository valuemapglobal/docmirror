"""
Adapters — Format-specific document converters.
================================================

Each adapter is responsible for:
    1. Converting a specific file format into an immutable ``BaseResult``.
    2. Optionally returning a ``ParserOutput`` for backward compatibility.

Adapters contain NO business logic — all domain-specific enhancement
is handled by the middleware pipeline downstream.

Supported formats:
    - PDF      → PDFAdapter
    - Image    → ImageAdapter (VLM + OCR fallback)
    - Word     → WordAdapter (.docx via python-docx)
    - Excel    → ExcelAdapter (.xlsx via openpyxl)
    - PPT      → PPTAdapter (.pptx via python-pptx)
    - Email    → EmailAdapter (.eml via stdlib email)
    - HTML     → WebAdapter (raw text extraction)
    - JSON/CSV → StructuredAdapter
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
