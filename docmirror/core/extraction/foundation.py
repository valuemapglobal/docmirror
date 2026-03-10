"""
基础引擎封装 (Foundation Engine Wrappers)
==========================================

对底层 PDF 库的统一封装，隔离第三方依赖：
    - FitzEngine:       PyMuPDF 快速文本/字体/元数据提取
    - PDFPlumberEngine: pdfplumber 高精度表格识别
    - OCREngine:        PaddleOCR/RapidOCR 懒加载封装

上游只通过这些引擎类访问底层能力，便于未来替换。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PyMuPDF 引擎
# ═══════════════════════════════════════════════════════════════════════════════

class FitzEngine:
    """
    PyMuPDF 封装 — 极速文本提取 + 字体分析。

    主要用途:
        1. 文本层预检 (判断电子/扫描)
        2. 全文提取 + 文本坐标
        3. 字体/颜色/加粗等视觉特征提取
    """

    @staticmethod
    def open(file_path: Path):
        """打开 PDF 返回 fitz.Document。"""
        import fitz
        return fitz.open(str(file_path))

    @staticmethod
    def has_text_layer(fitz_doc) -> bool:
        """
        快速检查 PDF 是否包含文本层。

        策略: 检查前 3 页，任一页有 >20 字符的文本即认为有文本层。
        """
        for page_idx in range(min(3, len(fitz_doc))):
            text = fitz_doc[page_idx].get_text()
            if text and len(text.strip()) > 20:
                return True
        return False

    @staticmethod
    def extract_page_text(fitz_page) -> str:
        """提取单页全文。"""
        return fitz_page.get_text()

    @staticmethod
    def extract_page_words(fitz_page) -> List[Tuple]:
        """
        提取单页 word 列表。

        每个 word: (x0, y0, x1, y1, text, block_no, line_no, word_no)
        """
        return fitz_page.get_text("words")

    @staticmethod
    def extract_page_blocks_with_style(fitz_page) -> List[Dict[str, Any]]:
        """
        提取单页文本块，附带字体/颜色信息。

        Returns:
            List of {
                "text": str,
                "bbox": (x0, y0, x1, y1),
                "font_name": str,
                "font_size": float,
                "color": int,
                "flags": int,  # bit 0=superscript, 1=italic, 2=serif, 3=monospace, 4=bold
            }
        """
        result = []
        text_dict = fitz_page.get_text("dict", flags=11)
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # 只处理文本块
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    result.append({
                        "text": span.get("text", ""),
                        "bbox": (
                            span.get("bbox", (0, 0, 0, 0))[0],
                            span.get("bbox", (0, 0, 0, 0))[1],
                            span.get("bbox", (0, 0, 0, 0))[2],
                            span.get("bbox", (0, 0, 0, 0))[3],
                        ),
                        "font_name": span.get("font", ""),
                        "font_size": span.get("size", 0.0),
                        "color": span.get("color", 0),
                        "flags": span.get("flags", 0),
                    })
        return result

    @staticmethod
    def get_page_dimensions(fitz_page) -> Tuple[float, float]:
        """返回页面 (width, height)。"""
        rect = fitz_page.rect
        return rect.width, rect.height

    @staticmethod
    def extract_raw_text_from_bbox(fitz_page, bbox: Tuple[float, float, float, float]) -> str:
        """
        提取指定 bounding box 内的 100% 准确底层文本。
        用途：作为 Hybrid Text-Vision Prior 注入给多模态大模型，防止数字/错别字幻觉。
        """
        import fitz
        x0, y0, x1, y1 = bbox
        rect = fitz.Rect(x0, y0, x1, y1)
        # flags=0 保持按阅读顺序提取纯文本
        return fitz_page.get_text("text", clip=rect).strip()

    @staticmethod
    def extract_multicrop_payload(fitz_page, rois: List[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
        """
        DeepSeek-OCR2 多裁剪启发 (Multi-Crop Tokenization):
        构造 Global Base (低分辨率全局) + Local Focus (高分辨率局部区块) 的多图载荷。
        
        Args:
            fitz_page: PyMuPDF 页面对象
            rois: 重点关注区域序列 (Region of Interest) 格式为 [(x0,y0,x1,y1), ...]
                  例如表格或密集数据区的 bounding boxes
                  
        Returns:
            Dict 包含:
               'global_img': 150 DPI 全局图 (base64 或 bytes, 此处返回 numpy RGB 以便外层调用)
               'local_patches': List of 300 DPI 局部图 RGB
        """
        import numpy as np
        import cv2
        
        payload = {"global_img": None, "local_patches": []}
        
        # 1. 生成 Global Base (低分辨率, 比如长边 <= 1024 相当于 150 DPI)
        pix_global = fitz_page.get_pixmap(dpi=150)
        img_global = np.frombuffer(pix_global.samples, dtype=np.uint8).reshape(pix_global.h, pix_global.w, pix_global.n)
        if pix_global.n == 4:
            img_global = cv2.cvtColor(img_global, cv2.COLOR_RGBA2RGB)
            
        payload["global_img"] = img_global
        
        # 2. 如果没有提供 ROIs, 直接返回
        if not rois:
            return payload
            
        # 3. 生成 Local Focus Patches (高分辨率 300 DPI)
        # 为了不全页面渲染 300DPI 占内存，利用 fitz 的 clip 参数只渲染 ROI 区域
        import fitz
        for roi in rois:
            x0, y0, x1, y1 = roi
            # 扩展边界 5px 防止吃字
            rect = fitz.Rect(max(0, x0 - 5), max(0, y0 - 5), x1 + 5, y1 + 5)
            
            pix_patch = fitz_page.get_pixmap(dpi=300, clip=rect)
            img_patch = np.frombuffer(pix_patch.samples, dtype=np.uint8).reshape(pix_patch.h, pix_patch.w, pix_patch.n)
            if pix_patch.n == 4:
                img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGBA2RGB)
                
            payload["local_patches"].append(img_patch)
            
        return payload


# ═══════════════════════════════════════════════════════════════════════════════
# pdfplumber 引擎
# ═══════════════════════════════════════════════════════════════════════════════

class PDFPlumberEngine:
    """
    pdfplumber 封装 — 高精度表格结构识别。

    主要用途:
        1. 表格检测与提取 (线框/文本两种策略)
        2. 字符级坐标提取
    """

    @staticmethod
    def open(file_path: Path):
        """打开 PDF 返回 pdfplumber.PDF。"""
        import pdfplumber
        return pdfplumber.open(str(file_path))

    @staticmethod
    def extract_tables(page_plum, **kwargs) -> List[List[List[str]]]:
        """
        从单页提取所有表格。

        Returns:
            List of tables, 每个 table 是 List[List[str]]。
        """
        try:
            tables = page_plum.extract_tables(kwargs) if kwargs else page_plum.extract_tables()
            if not tables:
                return []
            # 清理 None 值
            result = []
            for tbl in tables:
                if tbl:
                    cleaned = [
                        [str(cell) if cell is not None else "" for cell in row]
                        for row in tbl
                    ]
                    result.append(cleaned)
            return result
        except Exception as e:
            logger.debug(f"pdfplumber table extraction error: {e}")
            return []

    @staticmethod
    def get_page_chars(page_plum) -> List[Dict[str, Any]]:
        """返回单页所有字符的坐标信息。"""
        return page_plum.chars if hasattr(page_plum, 'chars') else []


# ═══════════════════════════════════════════════════════════════════════════════
# OCR 引擎
# ═══════════════════════════════════════════════════════════════════════════════

class OCREngine:
    """
    OCR 引擎封装 — 代理到 engines.vision.rapidocr_engine 统一单例。

    仅在扫描件处理时首次调用才加载模型，避免启动开销。
    """

    _instance: Optional[Any] = None

    @classmethod
    def get_engine(cls):
        """获取 OCR 引擎单例 (代理到 rapidocr_engine)。"""
        if cls._instance is None:
            try:
                from docmirror.core.ocr.vision.rapidocr_engine import get_ocr_engine
                cls._instance = get_ocr_engine()
            except ImportError:
                logger.warning("RapidOCR not available, OCR features disabled")
                return None
        return cls._instance

    @classmethod
    def is_available(cls) -> bool:
        """检查 OCR 引擎是否已就绪。"""
        engine = cls.get_engine()
        return engine is not None and hasattr(engine, '_engine') and engine._engine is not None

