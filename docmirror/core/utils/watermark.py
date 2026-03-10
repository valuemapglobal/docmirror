"""
预处理与水印过滤 (Preprocessing & Watermark Filter)
=====================================================

从 layout_analysis.py 拆分的 PDF 预处理功能。
包含 preprocess_pdf、filter_watermark_page、_dedup_overlapping_chars。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

def preprocess_pdf(file_path: Path) -> Path:
    """Layer 0 物理清洗: 用 pikepdf 剔除标注层。"""
    try:
        import pikepdf
        pdf = pikepdf.open(str(file_path))
        modified = False
        for page in pdf.pages:
            if '/Annots' in page:
                del page['/Annots']
                modified = True
        if modified:
            temp_path = file_path.parent / f"{file_path.stem}_cleaned.pdf"
            pdf.save(str(temp_path))
            logger.info(f"[v2] preprocess: removed annotations → {temp_path.name}")
            return temp_path
    except Exception as e:
        logger.debug(f"[v2] preprocess: pikepdf skip ({e})")
    return file_path


def is_watermark_char(obj: Dict) -> bool:
    """
    判断 pdfplumber 字符对象是否为水印。
    三重检测: 旋转/矩阵/颜色。
    """
    if not obj.get("upright", True):
        return True
    m = obj.get("matrix")
    if m and (abs(m[1]) > 0.1 or abs(m[2]) > 0.1):
        return True
    nsc = obj.get("non_stroking_color")
    if isinstance(nsc, (list, tuple)) and all(c > 0.5 for c in nsc):
        return True
    return False


def filter_watermark_page(page):
    """过滤 pdfplumber 页面中的水印字符。"""
    return page.filter(
        lambda obj: obj.get("object_type") != "char"
        or not is_watermark_char(obj)
    )


def _dedup_overlapping_chars(page):
    """去除伪加粗重复字符。"""
    seen = set()
    dedup_ids = set()
    bucket = 3
    for i, c in enumerate(page.chars):
        bx = int(c["x0"] / bucket)
        by = int(c["top"] / bucket)
        key = (bx, by, c["text"])
        if key in seen:
            dedup_ids.add(i)
        else:
            seen.add(key)

    if not dedup_ids:
        return page

    keep_chars = {id(page.chars[i]) for i in range(len(page.chars)) if i not in dedup_ids}
    return page.filter(
        lambda obj: obj.get("object_type") != "char" or id(obj) in keep_chars
    )
