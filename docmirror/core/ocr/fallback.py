"""
扫描件 OCR 回退 (Scanned Page OCR Fallback)
=============================================

从 layout_analysis.py 拆分的扫描件处理模块。
包含 analyze_scanned_page 及其辅助函数。
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def _preprocess_image_for_ocr(img_bgr):
    """OCR 前图像预处理 (增强版 v2):
    1. DPI 检测 + 低分辨率自动放大
    2. 自适应 CLAHE 对比度增强
    3. 双边滤波 (边缘保留降噪)
    4. Unsharp Mask 锐化
    5. 形态学去噪 (移除小斑点)
    6. 自适应二值化
    """
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]

    # ── Step 0: 低分辨率检测 + 自动放大 ──
    # 如果图片太小 (短边 < 1000px), 2x 放大以提升 OCR 精度
    min_dim = min(h, w)
    if min_dim < 1000:
        scale = 2.0
        img_bgr = cv2.resize(
            img_bgr, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        h, w = img_bgr.shape[:2]
        logger.debug(f"[OCR] 低分辨率图片已放大 2x → {w}x{h}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Step 1: 自适应 CLAHE (根据对比度动态调整) ──
    contrast = gray.std()
    clip_limit = 3.0 if contrast < 40 else 2.0 if contrast < 80 else 1.5
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ── Step 2: 双边滤波 (边缘保留降噪, 优于高斯) ──
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # ── Step 3: Unsharp Mask 锐化 ──
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # ── Step 4: 自适应二值化 ──
    # blockSize=21 对扫描件表格效果更好 (比 15 更抗背景渐变)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 8,
    )

    # ── Step 5: 形态学去噪 (移除 <3px 的小斑点噪声) ──
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _deskew_image(img_bgr):
    """页面纠偏: ±20° 以内。"""
    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        if abs(angle) < 20:
            angles.append(angle)

    if not angles:
        return img_bgr, 0.0

    median_angle = sorted(angles)[len(angles) // 2]

    if abs(median_angle) < 0.5:
        return img_bgr, 0.0

    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, median_angle


def _group_chars_into_rows(
    chars: List[dict], y_tolerance: float = 8.0
) -> List[Tuple[float, List[dict]]]:
    """将 OCR 字符按 Y 坐标分组为行。"""
    if not chars:
        return []

    # 按 top 排序
    sorted_chars = sorted(chars, key=lambda c: c.get("top", 0))
    
    rows: List[Tuple[float, List[dict]]] = []
    current_y = sorted_chars[0].get("top", 0)
    current_row: List[dict] = [sorted_chars[0]]

    for ch in sorted_chars[1:]:
        ch_y = ch.get("top", 0)
        if abs(ch_y - current_y) <= y_tolerance:
            current_row.append(ch)
        else:
            # 行内按 x 排序
            current_row.sort(key=lambda c: c.get("x0", 0))
            rows.append((current_y, current_row))
            current_y = ch_y
            current_row = [ch]

    if current_row:
        current_row.sort(key=lambda c: c.get("x0", 0))
        rows.append((current_y, current_row))

    return rows


def _chars_to_text(chars: List[dict]) -> str:
    """将字符列表合并为文本字符串。"""
    return " ".join(c.get("text", "") for c in chars).strip()


def _cluster_x_positions(
    x_positions: List[float], gap_multiplier: float = 2.0
) -> List[Tuple[float, float]]:
    """从 x 坐标聚类检测列边界。"""
    if not x_positions:
        return []

    sorted_x = sorted(set(x_positions))
    if len(sorted_x) < 2:
        return [(sorted_x[0], sorted_x[0] + 100)]

    # 计算间距
    gaps = [sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 10

    # 按大间距分割为列
    col_starts = [sorted_x[0]]
    for i, gap in enumerate(gaps):
        if gap > median_gap * gap_multiplier:
            col_starts.append(sorted_x[i + 1])

    # 构建列边界 (start, end)
    bounds = []
    for i, start in enumerate(col_starts):
        if i + 1 < len(col_starts):
            end = col_starts[i + 1]
        else:
            end = max(x_positions) + 10
        bounds.append((start, end))

    return bounds


def _assign_chars_to_columns(
    chars: List[dict], col_bounds: List[Tuple[float, float]]
) -> List[str]:
    """将一行字符分配到各列中。"""
    cols: List[List[dict]] = [[] for _ in col_bounds]
    
    for ch in chars:
        cx = (ch.get("x0", 0) + ch.get("x1", 0)) / 2
        assigned = False
        for i, (start, end) in enumerate(col_bounds):
            if start <= cx < end:
                cols[i].append(ch)
                assigned = True
                break
        if not assigned and cols:
            # 分配到最近的列
            min_dist = float("inf")
            min_idx = 0
            for i, (start, end) in enumerate(col_bounds):
                mid = (start + end) / 2
                dist = abs(cx - mid)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            cols[min_idx].append(ch)

    return [_chars_to_text(col) for col in cols]


def _split_tables_by_y_gap(
    rows_by_y: List[Tuple[float, List[dict]]], page_h: float
) -> List[List[Tuple[float, List[dict]]]]:
    """按 y 间隙将行分割为多个表格。"""
    if len(rows_by_y) < 4:
        return [rows_by_y]

    gap_threshold = page_h * 0.05
    tables: List[List[Tuple[float, List[dict]]]] = []
    current: List[Tuple[float, List[dict]]] = [rows_by_y[0]]

    for i in range(1, len(rows_by_y)):
        if rows_by_y[i][0] - rows_by_y[i - 1][0] > gap_threshold:
            tables.append(current)
            current = []
        current.append(rows_by_y[i])
    tables.append(current)

    return [t for t in tables if len(t) >= 2]


def _detect_table_lines_hough(
    img_bgr, page_h: int, page_w: int
) -> Optional[List[Tuple[float, float]]]:
    """Hough 变换检测扫描件中的表格垂直线。"""
    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_line_len = int(page_h * 0.15)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=3.14159 / 180, threshold=80,
        minLineLength=min_line_len, maxLineGap=10,
    )
    if lines is None:
        return None

    vertical_x = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 5:
            vertical_x.append((x1 + x2) / 2)

    if len(vertical_x) < 2:
        return None

    vertical_x.sort()
    clusters = [vertical_x[0]]
    for x in vertical_x[1:]:
        if x - clusters[-1] > 10:
            clusters.append(x)
        else:
            clusters[-1] = (clusters[-1] + x) / 2

    if len(clusters) < 2:
        return None

    col_bounds = []
    for i in range(len(clusters) - 1):
        col_bounds.append((clusters[i], clusters[i + 1]))

    col_bounds = [(a, b) for a, b in col_bounds if b - a > 20]

    return col_bounds if len(col_bounds) >= 2 else None


def analyze_scanned_page(
    fitz_page, page_idx: int, min_confidence: float = 0.3, table_bbox: Optional[Tuple[float, float, float, float]] = None
) -> Optional[Dict[str, Any]]:
    """对扫描件页面执行 OCR 提取。"""
    try:
        import numpy as np
        import cv2

        from ..extraction.foundation import FitzEngine
        
        # ── 优化: 图文混合先验 (Hybrid Text-Vision Prompt Prior) ──
        text_prior = ""
        if table_bbox:
            text_prior = FitzEngine.extract_raw_text_from_bbox(fitz_page, table_bbox)
        else:
            text_prior = FitzEngine.extract_page_text(fitz_page)
            
        if len(text_prior) > 1000:
            text_prior = text_prior[:1000]

        # 尝试导入 OCR 引擎
        ocr_engine = None
        try:
            from rapidocr_onnxruntime import RapidOCR as _RapidOCR
            ocr_engine = _RapidOCR()
        except ImportError:
            try:
                from docmirror.core.ocr.vision.rapidocr_engine import get_ocr_engine
                _eng = get_ocr_engine()
                if _eng and _eng._engine:
                    ocr_engine = _eng._engine
            except ImportError:
                pass

        if ocr_engine is None:
            logger.debug("[v2] OCR skipped: no OCR engine available")
            return None

        all_words = []
        page_h = 0
        img = None
        for dpi in [200, 300]:  # M3: 首次 200 DPI (原 150), 重试 300
            pix = fitz_page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            page_h = pix.h

            img_processed = _preprocess_image_for_ocr(img)
            img_processed, skew_angle = _deskew_image(img_processed)

            result, _ = ocr_engine(img_processed)
            if not result:
                if dpi == 150:
                    continue
                return None

            all_words = []
            for box, text, conf in result:
                if conf < min_confidence:
                    continue
                text = text.strip()
                if not text:
                    continue
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                all_words.append((
                    min(x_coords), min(y_coords),
                    max(x_coords), max(y_coords),
                    text,
                ))

            if len(all_words) >= 10 or dpi == 300:
                break

        if len(all_words) < 3:
            return None

        header_y = page_h * 0.12
        footer_y = page_h * 0.90

        header_words = [w for w in all_words if w[3] < header_y]
        footer_words = [w for w in all_words if w[1] > footer_y]
        table_words = [w for w in all_words
                       if w[1] >= header_y and w[3] <= footer_y]

        header_text = " ".join(
            w[4] for w in sorted(header_words, key=lambda w: (w[1], w[0]))
        )
        footer_text = " ".join(
            w[4] for w in sorted(footer_words, key=lambda w: (w[1], w[0]))
        )

        if len(table_words) < 2:
            return None

        chars = []
        for x0, y0, x1, y1, text in table_words:
            chars.append({
                "x0": float(x0), "x1": float(x1),
                "top": float(y0), "bottom": float(y1),
                "text": str(text),
                "upright": True,
            })

        rows_by_y = _group_chars_into_rows(chars, y_tolerance=8.0)
        if len(rows_by_y) < 2:
            return None

        table_groups = _split_tables_by_y_gap(rows_by_y, page_h)

        col_bounds = _detect_table_lines_hough(img, page_h, img.shape[1] if img is not None else 0)
        if not col_bounds:
            all_x0 = [c["x0"] for c in chars]
            col_bounds = _cluster_x_positions(all_x0, gap_multiplier=2.0)

        def _build_table(group_rows):
            if len(col_bounds) < 2:
                return [[_chars_to_text(rc)] for _, rc in group_rows]
            return [_assign_chars_to_columns(rc, col_bounds) for _, rc in group_rows]

        tables = [_build_table(g) for g in table_groups]
        tables = [t for t in tables if len(t) >= 2]

        if not tables:
            return None

        main_table = max(tables, key=len)

        raw_result = {
            "table": main_table,
            "tables": tables if len(tables) > 1 else None,
            "header_text": header_text,
            "footer_text": footer_text,
        }

        # OCR 后处理纠正 (金额/日期/领域词典)
        from .ocr_postprocess import postprocess_ocr_result
        return postprocess_ocr_result(raw_result)

    except ImportError:
        logger.debug("[v2] OCR skipped: required libraries not installed")
        return None
    except Exception as e:
        logger.warning(f"[v2] OCR error on page {page_idx}: {e}")
        return None

