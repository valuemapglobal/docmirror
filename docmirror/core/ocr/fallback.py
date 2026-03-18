# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Scanned Page OCR Fallback
=========================

Processes scanned document pages where no text layer is available.
Extracted from layout_analysis.py as an independent module.

Pipeline:
    1. **Image preprocessing** — adaptive CLAHE, bilateral filtering,
       unsharp mask sharpening, morphological denoising, adaptive
       binarisation.  Low-resolution images are automatically upscaled 2×.
    2. **Page deskewing** — contour-based median angle detection (±20°).
    3. **RapidOCR recognition** — two-pass DPI strategy (200 first, 300
       if too few words).
    4. **Region segmentation** — header (top 12 %), footer (bottom 10 %),
       and table body.
    5. **Column detection** — Hough line transform for scanned tables,
       with x-coordinate clustering fallback.
    6. **Table construction** — characters grouped into rows, split into
       tables by y-gap, assigned to columns.
    7. **OCR post-processing** — amount/date/domain-term correction via
       ``ocr_postprocess.postprocess_ocr_result``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple, Union

from .image_preprocessing import (
    deskew_image as _deskew_image,
)
from .image_preprocessing import (
    preprocess_image_for_ocr as _preprocess_image_for_ocr,
)

# ── Delegated sub-modules ──
from .image_preprocessing import (
    preprocess_minimal as _preprocess_minimal,
)
from .table_reconstruction import (
    assign_chars_to_columns as _assign_chars_to_columns,
)
from .table_reconstruction import (
    chars_to_text as _chars_to_text,
)
from .table_reconstruction import (
    cluster_x_positions as _cluster_x_positions,
)
from .table_reconstruction import (
    detect_has_table as _detect_has_table,
)
from .table_reconstruction import (
    detect_table_lines_hough as _detect_table_lines_hough,
)
from .table_reconstruction import (
    group_chars_into_rows as _group_chars_into_rows,
)
from .table_reconstruction import (
    group_words_into_lines as _group_words_into_lines,
)
from .table_reconstruction import (
    reconstruct_table_grid_2d as _reconstruct_table_grid_2d,
)
from .table_reconstruction import (
    split_tables_by_y_gap as _split_tables_by_y_gap,
)

logger = logging.getLogger(__name__)


def _resolve_external_ocr_provider(provider_spec: str | None):
    """Resolve 'module:callable' string to a callable, or return None.

    The callable is expected to have signature:
        (image_bgr, *, page_idx=0, dpi=200, **kwargs)
    and return either:
        - list of (x0, y0, x1, y1, text, confidence), or
        - dict with content_type ("table" | "general") and table/header_text/footer_text
          or lines/page_h/page_w (same as ocr_extract_universal).
    """
    if not provider_spec or ":" not in provider_spec:
        return None
    try:
        mod_name, attr = provider_spec.strip().rsplit(":", 1)
        import importlib

        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    except Exception as e:
        logger.warning(f"[external_ocr] Failed to resolve provider {provider_spec!r}: {e}")
        return None


def _render_page_to_bgr(fitz_page, dpi: int = 200):
    """Render a fitz page to BGR numpy array. Returns (img_bgr, page_h, page_w)."""
    import cv2
    import numpy as np

    pix = fitz_page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = img
    return img_bgr, img_bgr.shape[0], img_bgr.shape[1]


def assess_image_quality_from_bgr(img_bgr) -> int:
    """Assess image quality from a BGR numpy array (0–100).

    Uses Laplacian variance as a blur/sharpness proxy. Same scale as
    PreAnalyzer._assess_image_quality for consistency. Used to decide
    whether to delegate to external OCR when quality is below threshold.
    """
    import numpy as np

    try:
        import cv2

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = np.mean(img_bgr[:, :, :3], axis=2).astype(np.uint8)
    try:
        from scipy.ndimage import laplace  # type: ignore

        lap = laplace(gray.astype(np.float64))
        lap_var = float(np.var(lap))
        if lap_var >= 200:
            score = 100
        elif lap_var >= 50:
            score = 60 + int((lap_var - 50) / 150 * 40)
        else:
            score = int(lap_var / 50 * 60)
        return max(0, min(100, score))
    except Exception:
        std = float(np.std(gray))
        return max(0, min(100, int(std * 1.5)))


def _read_exif_orientation(fitz_page) -> int:
    """Read EXIF Orientation tag and return rotation angle in degrees.

    Phone cameras always record orientation in EXIF.  This is 100% reliable
    and has zero computational overhead compared to OCR-based probing.

    Returns:
        int: rotation angle (0, 90, 180, or 270).
    """
    try:
        doc = fitz_page.parent
        if doc is None:
            return 0
        # fitz exposes image metadata; check for rotation
        # Method 1: page rotation property
        rot = fitz_page.rotation
        if rot in (90, 180, 270):
            return rot
        # Method 2: for single-image PDFs, check the image EXIF
        images = fitz_page.get_images(full=True)
        if len(images) == 1:
            xref = images[0][0]
            import fitz as _fitz

            pix = _fitz.Pixmap(doc, xref)
            # fitz Pixmap doesn't expose EXIF directly, but the
            # _image_to_virtual_pdf path already handles this via
            # fitz.open() which auto-applies EXIF rotation.
            del pix
    except Exception as exc:
        logger.debug(f"operation: suppressed {exc}")
    return 0


def _preprocess_minimal(img_bgr):
    """Minimal preprocessing (Strategy B): preserves maximum information.

    Steps:
        0.  Adaptive upscale (Lanczos4 + sharpening).
        1.  Gamma correction for dark images.
        2.  Histogram stretch for low contrast.
        3.  Red seal removal.
        4.  CLAHE contrast enhancement.
        5.  Light bilateral filtering.

    Output: grayscale image (NOT binarised) — lets the OCR engine use
    gradient information for character recognition.
    """
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]

    # ── Adaptive upscale ──
    min_dim = min(h, w)
    if min_dim < 500:
        scale = 4.0
    elif min_dim < 1000:
        scale = 2.0
    else:
        scale = 0
    if scale > 0:
        img_bgr = cv2.resize(
            img_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LANCZOS4,
        )
        sharp_kernel = np.array(
            [
                [0, -0.5, 0],
                [-0.5, 3, -0.5],
                [0, -0.5, 0],
            ],
            dtype=np.float32,
        )
        img_bgr = cv2.filter2D(img_bgr, -1, sharp_kernel)
        h, w = img_bgr.shape[:2]

    # ── Gamma for dark images ──
    gray_check = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_br = float(gray_check.mean())
    contrast = float(gray_check.std())
    if mean_br < 80:
        gamma = 0.6
        lut = np.array([min(255, int(((i / 255.0) ** gamma) * 255)) for i in range(256)], dtype=np.uint8)
        img_bgr = cv2.LUT(img_bgr, lut)

    # ── Histogram stretch for low contrast ──
    if contrast < 25:
        for c in range(3):
            ch = img_bgr[:, :, c]
            p_lo, p_hi = np.percentile(ch, (1, 99))
            if p_hi - p_lo > 10:
                img_bgr[:, :, c] = np.clip(
                    (ch.astype(np.float32) - p_lo) / (p_hi - p_lo) * 255,
                    0,
                    255,
                ).astype(np.uint8)

    # ── Red seal removal (same as full pipeline) ──
    try:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
        red_mask = cv2.bitwise_or(mask1, mask2)
        kernel_seal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_mask = cv2.dilate(red_mask, kernel_seal, iterations=1)
        img_bgr[red_mask > 0] = (255, 255, 255)
    except Exception as exc:
        logger.debug(f"operation: suppressed {exc}")

    # ── CLAHE on grayscale ──
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ── Light bilateral smoothing ──
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # Return as BGR (3-channel grayscale) — NO binarisation
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _ocr_upscale_and_normalize(img_bgr, cv2, np):
    """Stage 1: Upscale low-res images, gamma correct, histogram stretch, edge pad."""
    h, w = img_bgr.shape[:2]

    # ── Adaptive upscaling for low-resolution images ──
    min_dim = min(h, w)
    if min_dim < 500:
        scale = 4.0
    elif min_dim < 1000:
        scale = 2.0
    else:
        scale = 0
    if scale > 0:
        img_bgr = cv2.resize(
            img_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LANCZOS4,
        )
        sharp_kernel = np.array(
            [
                [0, -0.5, 0],
                [-0.5, 3, -0.5],
                [0, -0.5, 0],
            ],
            dtype=np.float32,
        )
        img_bgr = cv2.filter2D(img_bgr, -1, sharp_kernel)
        h, w = img_bgr.shape[:2]
        logger.debug(f"[OCR] Upscaled {scale:.0f}\u00d7 (Lanczos+sharp) \u2192 {w}x{h}")

    # ── Gamma correction for dark images ──
    gray_check = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray_check.mean())
    img_contrast = float(gray_check.std())
    if mean_brightness < 80:
        gamma = 0.6
        lut = np.array([min(255, int(((i / 255.0) ** gamma) * 255)) for i in range(256)], dtype=np.uint8)
        img_bgr = cv2.LUT(img_bgr, lut)
        logger.debug(
            f"[OCR] Gamma correction (\u03b3={gamma}): "
            f"brightness {mean_brightness:.0f} \u2192 {cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean():.0f}"
        )

    # ── Histogram stretch for very low contrast ──
    if img_contrast < 25:
        for c in range(3):
            channel = img_bgr[:, :, c]
            p_lo, p_hi = np.percentile(channel, (1, 99))
            if p_hi - p_lo > 10:
                channel = np.clip(
                    (channel.astype(np.float32) - p_lo) / (p_hi - p_lo) * 255,
                    0,
                    255,
                ).astype(np.uint8)
                img_bgr[:, :, c] = channel
        logger.debug(
            f"[OCR] Histogram stretch: contrast {img_contrast:.0f} \u2192 "
            f"{cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).std():.0f}"
        )

    # ── Edge padding to prevent border text clipping ──
    pad = 20
    img_bgr = cv2.copyMakeBorder(
        img_bgr,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return img_bgr


def _ocr_remove_interference(img_bgr, cv2, np):
    """Stage 2: Remove seals, watermarks, borders; correct perspective and BG lighting."""
    h, w = img_bgr.shape[:2]

    # ── KMeans Color Slicing ──
    try:
        small_for_km = cv2.resize(img_bgr, (0, 0), fx=0.5, fy=0.5)
        hsv_for_km = cv2.cvtColor(small_for_km, cv2.COLOR_BGR2HSV)
        pixels = hsv_for_km.reshape((-1, 3)).astype(np.float32)
        k = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)

        bg_cluster_idx = -1
        max_bg_score = -1
        for i, center in enumerate(centers):
            h_val, s_val, v_val = center
            score = float(v_val) - float(s_val)
            if score > max_bg_score:
                max_bg_score = score
                bg_cluster_idx = i

        text_cluster_idx = -1
        min_v = 256
        for i, center in enumerate(centers):
            if i == bg_cluster_idx:
                continue
            if center[2] < min_v:
                min_v = center[2]
                text_cluster_idx = i

        full_labels = cv2.resize(labels.reshape(small_for_km.shape[:2]), (w, h), interpolation=cv2.INTER_NEAREST)
        img_bgr[full_labels == bg_cluster_idx] = (255, 255, 255)
        img_bgr[full_labels == text_cluster_idx] = (0, 0, 0)

        for i in range(k):
            if i != bg_cluster_idx and i != text_cluster_idx:
                h_val, s_val, v_val = centers[i]
                is_red = (h_val < 15 or h_val > 165) and s_val > 50
                is_watermark = s_val < 40 and v_val > 200
                if is_red or is_watermark:
                    img_bgr[full_labels == i] = (255, 255, 255)
                else:
                    img_bgr[full_labels == i] = (0, 0, 0)
    except Exception as e:
        logger.debug(f"[OCR] KMeans color slicing failed: {e}")

    # ── Watermark removal ──
    try:
        hsv_wm = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        wm_mask = cv2.inRange(hsv_wm, (0, 0, 200), (180, 40, 255))
        wm_mask2 = cv2.inRange(hsv_wm, (0, 0, 180), (180, 60, 255))
        wm_combined = cv2.bitwise_or(wm_mask, wm_mask2)
        wm_ratio = cv2.countNonZero(wm_combined) / (h * w)
        if 0.01 < wm_ratio < 0.40:
            kernel_wm = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            wm_combined = cv2.morphologyEx(wm_combined, cv2.MORPH_CLOSE, kernel_wm, iterations=1)
            img_bgr[wm_combined > 0] = (255, 255, 255)
    except Exception as e:
        logger.debug(f"[OCR] Watermark removal skipped: {e}")

    # ── Decorative border cropping ──
    try:
        gray_border = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh_border = cv2.threshold(gray_border, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area_ratio = cv2.contourArea(largest) / (h * w)
            if 0.20 < area_ratio < 0.95:
                x, y, cw, ch = cv2.boundingRect(largest)
                margin = 10
                x = max(0, x - margin)
                y = max(0, y - margin)
                cw = min(w - x, cw + 2 * margin)
                ch = min(h - y, ch + 2 * margin)
                img_bgr = img_bgr[y : y + ch, x : x + cw]
                h, w = img_bgr.shape[:2]
    except Exception as e:
        logger.debug(f"[OCR] Border crop skipped: {e}")

    # ── Perspective correction ──
    try:
        gray_persp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred_persp = cv2.GaussianBlur(gray_persp, (5, 5), 0)
        edges = cv2.Canny(blurred_persp, 50, 150)
        kernel_persp = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel_persp, iterations=1)
        contours_p, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_p:
            largest_p = max(contours_p, key=cv2.contourArea)
            peri = cv2.arcLength(largest_p, True)
            approx = cv2.approxPolyDP(largest_p, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 0.30 * h * w:
                pts = approx.reshape(4, 2).astype(np.float32)
                s = pts.sum(axis=1)
                d = np.diff(pts, axis=1).ravel()
                ordered = np.array(
                    [
                        pts[np.argmin(s)],
                        pts[np.argmin(d)],
                        pts[np.argmax(s)],
                        pts[np.argmax(d)],
                    ],
                    dtype=np.float32,
                )
                w_top = np.linalg.norm(ordered[1] - ordered[0])
                w_bot = np.linalg.norm(ordered[2] - ordered[3])
                h_left = np.linalg.norm(ordered[3] - ordered[0])
                h_right = np.linalg.norm(ordered[2] - ordered[1])
                out_w = int(max(w_top, w_bot))
                out_h = int(max(h_left, h_right))
                dst = np.array(
                    [
                        [0, 0],
                        [out_w, 0],
                        [out_w, out_h],
                        [0, out_h],
                    ],
                    dtype=np.float32,
                )
                M = cv2.getPerspectiveTransform(ordered, dst)
                img_bgr = cv2.warpPerspective(
                    img_bgr,
                    M,
                    (out_w, out_h),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                h, w = img_bgr.shape[:2]
    except Exception as e:
        logger.debug(f"[OCR] Perspective correction skipped: {e}")

    # ── Background lighting equalisation ──
    try:
        gray_bg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg = cv2.GaussianBlur(gray_bg, (0, 0), sigmaX=51)
        bg[bg < 1] = 1
        normalised = (gray_bg / bg * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(normalised, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        logger.debug(f"[OCR] Background equalisation skipped: {e}")

    return img_bgr


def _ocr_binarize_and_clean(img_bgr, cv2, np):
    """Stage 3: CLAHE + bilateral + binarisation voting + line removal + morph repair."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Adaptive CLAHE ──
    contrast = gray.std()
    clip_limit = 3.0 if contrast < 40 else 2.0 if contrast < 80 else 1.5
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ── Bilateral filtering ──
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # ── Unsharp mask sharpening ──
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # ── Multi-method binarisation voting ──
    _, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_gauss = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        8,
    )
    bin_mean = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )
    vote = bin_otsu.astype(np.uint16) + bin_gauss.astype(np.uint16) + bin_mean.astype(np.uint16)
    binary = np.where(vote >= 2 * 255, 255, 0).astype(np.uint8)

    # ── Table/grid line removal ──
    try:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1))
        h_lines = cv2.morphologyEx(cv2.bitwise_not(binary), cv2.MORPH_OPEN, h_kernel, iterations=1)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 8))
        v_lines = cv2.morphologyEx(cv2.bitwise_not(binary), cv2.MORPH_OPEN, v_kernel, iterations=1)
        all_lines = cv2.bitwise_or(h_lines, v_lines)
        kernel_line_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        all_lines = cv2.dilate(all_lines, kernel_line_d, iterations=1)
        line_px = int(cv2.countNonZero(all_lines))
        if line_px > 0:
            binary[all_lines > 0] = 255
    except Exception as e:
        logger.debug(f"[OCR] Line removal skipped: {e}")

    # ── Morphological text repair ──
    inv_binary = cv2.bitwise_not(binary)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    inv_binary = cv2.morphologyEx(inv_binary, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    inv_binary = cv2.morphologyEx(inv_binary, cv2.MORPH_OPEN, noise_kernel, iterations=1)
    binary = cv2.bitwise_not(inv_binary)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _preprocess_image_for_ocr(img_bgr):
    """Enhanced image preprocessing for OCR (v4 — 3-stage pipeline).

    Stage 1: Upscale + normalise (gamma, histogram, padding).
    Stage 2: Remove interference (colour slicing, watermarks, borders, perspective, BG equalisation).
    Stage 3: Binarise + clean (CLAHE, bilateral, binarisation voting, line removal, morph repair).
    """
    import cv2
    import numpy as np

    # Stage 1: Upscale and normalise
    img_bgr = _ocr_upscale_and_normalize(img_bgr, cv2, np)
    # Stage 2: Remove interference patterns
    img_bgr = _ocr_remove_interference(img_bgr, cv2, np)
    # Stage 3: Binarise and clean
    return _ocr_binarize_and_clean(img_bgr, cv2, np)


def _deskew_image(img_bgr):
    """Deskew a page image by detecting and correcting rotation (±20°).

    Uses contour-based minimum-area-rectangle angle estimation with
    median filtering for robustness.
    """
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
        img_bgr,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, median_angle


def _group_chars_into_rows(chars: list[dict], y_tolerance: float = 8.0) -> list[tuple[float, list[dict]]]:
    """Group OCR character dicts into rows by y-coordinate proximity."""
    if not chars:
        return []

    sorted_chars = sorted(chars, key=lambda c: c.get("top", 0))

    rows: list[tuple[float, list[dict]]] = []
    current_y = sorted_chars[0].get("top", 0)
    current_row: list[dict] = [sorted_chars[0]]

    for ch in sorted_chars[1:]:
        ch_y = ch.get("top", 0)
        if abs(ch_y - current_y) <= y_tolerance:
            current_row.append(ch)
        else:
            # Sort within-row characters by x position
            current_row.sort(key=lambda c: c.get("x0", 0))
            rows.append((current_y, current_row))
            current_y = ch_y
            current_row = [ch]

    if current_row:
        current_row.sort(key=lambda c: c.get("x0", 0))
        rows.append((current_y, current_row))

    return rows


def _chars_to_text(chars: list[dict]) -> str:
    """Merge a list of character dicts into a single text string."""
    return " ".join(c.get("text", "") for c in chars).strip()


def _cluster_x_positions(x_positions: list[float], gap_multiplier: float = 2.0) -> list[tuple[float, float]]:
    """Detect column boundaries by clustering x-coordinates."""
    if not x_positions:
        return []

    sorted_x = sorted(set(x_positions))
    if len(sorted_x) < 2:
        return [(sorted_x[0], sorted_x[0] + 100)]

    # Compute inter-position gaps
    gaps = [sorted_x[i + 1] - sorted_x[i] for i in range(len(sorted_x) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 10

    # Split into columns at large gaps
    col_starts = [sorted_x[0]]
    for i, gap in enumerate(gaps):
        if gap > median_gap * gap_multiplier:
            col_starts.append(sorted_x[i + 1])

    # Build column boundary intervals (start, end)
    bounds = []
    for i, start in enumerate(col_starts):
        if i + 1 < len(col_starts):
            end = col_starts[i + 1]
        else:
            end = max(x_positions) + 10
        bounds.append((start, end))

    return bounds


def _assign_chars_to_columns(chars: list[dict], col_bounds: list[tuple[float, float]]) -> list[str]:
    """Assign a row's characters to column bins."""
    cols: list[list[dict]] = [[] for _ in col_bounds]

    for ch in chars:
        cx = (ch.get("x0", 0) + ch.get("x1", 0)) / 2
        assigned = False
        for i, (start, end) in enumerate(col_bounds):
            if start <= cx < end:
                cols[i].append(ch)
                assigned = True
                break
        if not assigned and cols:
            # Assign to the nearest column by midpoint distance
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
    rows_by_y: list[tuple[float, list[dict]]], page_h: float
) -> list[list[tuple[float, list[dict]]]]:
    """Split grouped rows into multiple tables based on vertical gaps."""
    if len(rows_by_y) < 4:
        return [rows_by_y]

    gap_threshold = page_h * 0.05
    tables: list[list[tuple[float, list[dict]]]] = []
    current: list[tuple[float, list[dict]]] = [rows_by_y[0]]

    for i in range(1, len(rows_by_y)):
        if rows_by_y[i][0] - rows_by_y[i - 1][0] > gap_threshold:
            tables.append(current)
            current = []
        current.append(rows_by_y[i])
    tables.append(current)

    return [t for t in tables if len(t) >= 2]


def _reconstruct_table_grid_2d(
    chars: list[dict], hough_lines: list[tuple[float, float]] | None = None
) -> list[list[str]]:
    """Robust 2D Table Grid Reconstruction (Virtual Grid Alignment).

    Replaces 1D x-coordinate clustering with a 2D spatial alignment algorithm.
    It builds a virtual grid by finding strong alignment edges and snapping characters
    to the optimal (row, col) coordinates, handling jagged cell contents and
    misaligned headers.

    Algorithm:
        1. Base Row Clustering: Group chars by y-overlap (IoU).
        2. Base Col Clustering: Group chars by x-overlap (IoU) or Hough lines.
        3. Grid Snapping: Assign each char to a (row_idx, col_idx) bucket.
        4. Output Generation: Build a dense 2D list of strings.
    """
    if not chars:
        return []

    # 1. Base Row Clustering (Robust y-projection)
    # Sort by top coordinate
    sorted_chars = sorted(chars, key=lambda c: c["top"])

    rows_y = []  # list of (min_y, max_y, chars)

    for c in sorted_chars:
        c_min_y, c_max_y = c["top"], c["bottom"]
        matched = False
        # Try to match with existing row (look at last few rows to handle slight overlaps)
        for i in range(len(rows_y) - 1, max(-1, len(rows_y) - 4), -1):
            r_min_y, r_max_y, r_chars = rows_y[i]

            # Calculate vertical IoU or significant overlap
            overlap = max(0, min(c_max_y, r_max_y) - max(c_min_y, r_min_y))
            c_height = c_max_y - c_min_y

            # If overlap is > 40% of character height, it belongs to this row
            if overlap > 0.4 * c_height or (c_min_y >= r_min_y and c_max_y <= r_max_y):
                # Update row boundaries
                rows_y[i] = (min(r_min_y, c_min_y), max(r_max_y, c_max_y), r_chars + [c])
                matched = True
                break

        if not matched:
            rows_y.append((c_min_y, c_max_y, [c]))

    # Sort rows by their physical y-position
    rows_y.sort(key=lambda x: x[0])
    row_chars_list = [r[2] for r in rows_y]

    # 2. Base Col Clustering
    col_bounds = []  # list of (min_x, max_x)

    if hough_lines and len(hough_lines) >= 2:
        col_bounds = hough_lines
    else:
        # Fallback to robust X-clustering using all characters
        x_spans = [(c["x0"], c["x1"]) for c in chars]
        x_spans.sort(key=lambda x: x[0])

        merged_cols = []
        for span in x_spans:
            if not merged_cols:
                merged_cols.append([span[0], span[1]])
                continue

            last_col = merged_cols[-1]
            # If x overlaps or gap is very small (< 10px), merge into same column
            if span[0] <= last_col[1] + 10:
                last_col[1] = max(last_col[1], span[1])
            else:
                merged_cols.append([span[0], span[1]])

        # If we merged everything into 1 column, fallback to K-Means/Gap logic
        if len(merged_cols) < 2:
            all_x0 = [c["x0"] for c in chars]
            # Reuse 1D clustering as a last resort
            col_bounds = _cluster_x_positions(all_x0, gap_multiplier=2.0)
        else:
            col_bounds = [(c[0], c[1]) for c in merged_cols]

    # Ensure at least 1 column
    if not col_bounds:
        col_bounds = [(0, 9999)]

    # 3. Grid Snapping
    num_rows = len(row_chars_list)
    num_cols = len(col_bounds)

    table_grid: list[list[list[dict]]] = [[[] for _ in range(num_cols)] for _ in range(num_rows)]

    for r_idx, r_chars in enumerate(row_chars_list):
        for c in r_chars:
            cx = (c["x0"] + c["x1"]) / 2

            # Find best column index
            best_c_idx = 0
            min_dist = float("inf")

            for c_idx, (start, end) in enumerate(col_bounds):
                if start <= cx <= end:
                    best_c_idx = c_idx
                    break

                # Calculate distance to mid-point if outside
                mid = (start + end) / 2
                dist = abs(cx - mid)
                if dist < min_dist:
                    min_dist = dist
                    best_c_idx = c_idx

            table_grid[r_idx][best_c_idx].append(c)

    # 4. Output Generation (Merge characters in each cell to string)
    final_table = []
    for r_idx in range(num_rows):
        row_str = []
        for c_idx in range(num_cols):
            cell_chars = table_grid[r_idx][c_idx]
            # Sort characters in cell left-to-right
            cell_chars.sort(key=lambda c: c["x0"])
            row_str.append(_chars_to_text(cell_chars))
        final_table.append(row_str)

    return final_table


def _detect_table_lines_hough(img_bgr, page_h: int, page_w: int) -> list[tuple[float, float]] | None:
    """Detect vertical table lines in a scanned image using Hough transform.

    Returns column boundary intervals derived from clustered vertical
    line x-coordinates, or ``None`` if too few lines are found.
    """
    import cv2

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_line_len = int(page_h * 0.15)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=3.14159 / 180,
        threshold=80,
        minLineLength=min_line_len,
        maxLineGap=10,
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


def _probe_best_orientation(img_bgr, ocr_engine=None):
    """Try OCR at 0°/90°/180°/270° on a downscaled image; return best angle.

    Uses a fast, low-resolution probe (~800px max dimension) to determine
    the correct document orientation.  Applies gamma correction only for
    very dark images to avoid washing out orientation signal.

    Returns:
        int: best rotation angle (0, 90, 180, or 270).
    """
    import cv2
    import numpy as np

    if ocr_engine is None:
        return 0

    h, w = img_bgr.shape[:2]
    max_probe = 800
    if max(h, w) > max_probe:
        scale = max_probe / max(h, w)
        small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr.copy()

    # Only apply gamma for very dark images — preserve natural signal otherwise
    gray_check = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if gray_check.mean() < 100:
        gamma = 0.5
        lut = np.array([min(255, int(((i / 255.0) ** gamma) * 255)) for i in range(256)], dtype=np.uint8)
        small = cv2.LUT(small, lut)

    best_angle = 0
    best_score = -1.0

    for angle in [0, 90, 180, 270]:
        if angle == 0:
            probe_img = small
        elif angle == 90:
            probe_img = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            probe_img = cv2.rotate(small, cv2.ROTATE_180)
        else:
            probe_img = cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)

        try:
            words = ocr_engine.detect_image_words(probe_img)
        except Exception as exc:
            logger.debug(f"operation: suppressed {exc}")
            continue

        if not words:
            continue

        # conf is at index 8
        score = sum(w[8] for w in words if len(w) > 8 and w[8] >= 0.5)
        logger.debug(f"[OCR] Orientation probe {angle}°: score={score:.1f}")

        if score > best_score:
            best_score = score
            best_angle = angle

    if best_angle != 0:
        logger.info(f"[OCR] Auto-orient: best angle = {best_angle}°")

    return best_angle


def analyze_scanned_page(
    fitz_page,
    page_idx: int,
    min_confidence: float = 0.3,
    table_bbox: tuple[float, float, float, float] | None = None,
    target_dpi: int = 200,
) -> dict[str, Any] | None:
    """Perform OCR-based extraction on a scanned document page.

    Pipeline:
        1. Optionally extract a text prior from the native text layer
           (hybrid text–vision prompt).
        2. Initialise an OCR engine (RapidOCR preferred).
        3. Render the page at ``target_dpi`` (retry at next tier if too few words).
        4. Preprocess the image and deskew.
        5. Run OCR; filter by confidence threshold.
        6. Segment into header / footer / table-body regions.
        7. Group characters into rows, detect columns (Hough or clustering),
           and build the table grid.
        8. Apply OCR post-processing corrections.

    Args:
        target_dpi: Rendering DPI for OCR.  Defaults to 200.
            The AdaptiveQualityRouter may pass 300 for dense/low-quality zones.

    Returns:
        A dict with keys ``table``, ``tables``, ``header_text``,
        ``footer_text``, or ``None`` on failure.
    """
    try:
        import cv2
        import numpy as np

        from ..extraction.foundation import FitzEngine

        # ── Hybrid text–vision prompt prior ──
        text_prior = ""
        if table_bbox:
            text_prior = FitzEngine.extract_raw_text_from_bbox(fitz_page, table_bbox)
        else:
            text_prior = FitzEngine.extract_page_text(fitz_page)

        if len(text_prior) > 1000:
            text_prior = text_prior[:1000]

        # Try to import an OCR engine
        ocr_engine = None
        try:
            from docmirror.core.ocr.vision.rapidocr_engine import get_ocr_engine

            ocr_engine = get_ocr_engine()
        except ImportError:
            pass

        if ocr_engine is None or not ocr_engine._engine:
            logger.debug("OCR skipped: no OCR engine available")
            return None

        # ── Auto-orientation probe ──
        # Render at low DPI for fast probe, detect best rotation
        probe_pix = fitz_page.get_pixmap(dpi=100)
        probe_img = np.frombuffer(probe_pix.samples, dtype=np.uint8).reshape(probe_pix.h, probe_pix.w, probe_pix.n)
        if probe_pix.n == 3:
            probe_img = cv2.cvtColor(probe_img, cv2.COLOR_RGB2BGR)
        elif probe_pix.n == 4:
            probe_img = cv2.cvtColor(probe_img, cv2.COLOR_RGBA2BGR)
        best_angle = _probe_best_orientation(probe_img, ocr_engine)

        all_words = []
        page_h = 0
        img = None
        # Adaptive DPI passes: start at target_dpi, escalate if needed
        dpi_passes = [target_dpi]
        if target_dpi < 300:
            dpi_passes.append(300)  # escalation pass
        for dpi in dpi_passes:
            pix = fitz_page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            # Apply orientation correction if needed
            if best_angle == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif best_angle == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif best_angle == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            page_h = img.shape[0]

            img_processed = _preprocess_image_for_ocr(img)
            img_processed, skew_angle = _deskew_image(img_processed)

            words = ocr_engine.detect_image_words(img_processed, multi_scale=(dpi >= 300))
            if not words:
                if dpi == 150:
                    continue
                return None

            all_words = []
            for w in words:
                conf = w[8] if len(w) > 8 else 1.0
                if conf < min_confidence:
                    continue
                text = w[4].strip()
                if not text:
                    continue
                all_words.append((w[0], w[1], w[2], w[3], text))

            if len(all_words) >= 10 or dpi == 300:
                break

        if len(all_words) < 3:
            return None

        # ── Region segmentation ──
        header_y = page_h * 0.12
        footer_y = page_h * 0.90

        header_words = [w for w in all_words if w[3] < header_y]
        footer_words = [w for w in all_words if w[1] > footer_y]
        table_words = [w for w in all_words if w[1] >= header_y and w[3] <= footer_y]

        header_text = " ".join(w[4] for w in sorted(header_words, key=lambda w: (w[1], w[0])))
        footer_text = " ".join(w[4] for w in sorted(footer_words, key=lambda w: (w[1], w[0])))

        if len(table_words) < 2:
            return None

        chars = []
        for x0, y0, x1, y1, text in table_words:
            chars.append(
                {
                    "x0": float(x0),
                    "x1": float(x1),
                    "top": float(y0),
                    "bottom": float(y1),
                    "text": str(text),
                    "upright": True,
                }
            )

        rows_by_y = _group_chars_into_rows(chars, y_tolerance=8.0)
        if len(rows_by_y) < 2:
            return None

        _split_tables_by_y_gap(rows_by_y, page_h)

        # Detect column boundaries — Hough lines first
        col_bounds_hough = _detect_table_lines_hough(img, page_h, img.shape[1] if img is not None else 0)

        # ── Advanced 2D Grid Reconstruction ──
        # Group characters into independent tables first
        rows_by_y_raw = _group_chars_into_rows(chars, y_tolerance=8.0)
        table_groups_raw = _split_tables_by_y_gap(rows_by_y_raw, page_h)

        tables = []
        for group in table_groups_raw:
            # Flatten group chars
            grp_chars = []
            for _, r_chars in group:
                grp_chars.extend(r_chars)

            # Reconstruct 2D grid
            tb = _reconstruct_table_grid_2d(grp_chars, hough_lines=col_bounds_hough)

            # Clean up empty rows and single-column tables
            tb_clean = [row for row in tb if any(cell.strip() for cell in row)]
            if len(tb_clean) >= 2 and len(tb_clean[0]) >= 2:
                tables.append(tb_clean)

        if not tables:
            return None

        main_table = max(tables, key=len)

        raw_result = {
            "table": main_table,
            "tables": tables if len(tables) > 1 else None,
            "header_text": header_text,
            "footer_text": footer_text,
        }

        # Apply OCR post-processing corrections (amount / date / domain terms)
        from .ocr_postprocess import postprocess_ocr_result

        return postprocess_ocr_result(raw_result)

    except ImportError:
        logger.debug("OCR skipped: required libraries not installed")
        return None
    except Exception as e:
        logger.warning(f"OCR error on page {page_idx}: {e}")
        return None


# ===============================================================================
# Universal OCR Extraction (content-type aware)
# ===============================================================================


def _merge_line_fragments(words):
    """Merge OCR word fragments that belong to the same text line.

    Conservative rules:
        - Vertical overlap > 50% of the shorter word's height.
        - Horizontal gap < 1.5× average character height.
    Sorts by reading order (top→bottom, left→right) after merging.

    Args:
        words: list of (x0, y0, x1, y1, text, conf) tuples.
    Returns:
        Merged list in the same format.
    """
    if not words or len(words) < 2:
        return words

    # Sort by y-centre then x
    ws = sorted(words, key=lambda w: (((w[1] + w[3]) / 2), w[0]))
    merged = [list(ws[0])]

    for w in ws[1:]:
        last = merged[-1]
        # Heights
        h_last = last[3] - last[1]
        h_curr = w[3] - w[1]
        min_h = min(h_last, h_curr)
        if min_h <= 0:
            merged.append(list(w))
            continue

        # Vertical overlap
        overlap_top = max(last[1], w[1])
        overlap_bot = min(last[3], w[3])
        v_overlap = max(0, overlap_bot - overlap_top)

        # Horizontal gap
        h_gap = w[0] - last[2]

        avg_char_h = (h_last + h_curr) / 2

        if v_overlap > 0.5 * min_h and h_gap < 1.5 * avg_char_h:
            # Merge: extend bounding box, concatenate text
            last[0] = min(last[0], w[0])
            last[1] = min(last[1], w[1])
            last[2] = max(last[2], w[2])
            last[3] = max(last[3], w[3])
            last[4] = last[4] + w[4]
            # Weighted average confidence
            last[5] = (last[5] * len(last[4]) + w[5] * len(w[4])) / (len(last[4]) + len(w[4]))
        else:
            merged.append(list(w))

    return [tuple(m) for m in merged]


def _merge_multi_scale_words(all_scale_words: list[tuple[int, list[tuple]]]) -> list[tuple]:
    """Fuse words from multiple DPI scales using Non-Maximum Suppression (NMS).

    Args:
        all_scale_words: List of (dpi, words) tuples.
            words are (x0, y0, x1, y1, text, conf) in the scale's coordinate space.
    Returns:
        Fused list of words in the 72 DPI (standard PDF) coordinate space.
    """
    if not all_scale_words:
        return []

    BASE_DPI = 72.0
    projected_words = []

    # Project all words to 72 DPI space
    for dpi, words in all_scale_words:
        scale = BASE_DPI / float(dpi)
        for w in words:
            px0, py0 = w[0] * scale, w[1] * scale
            px1, py1 = w[2] * scale, w[3] * scale
            # Store tuple: (x0, y0, x1, y1, text, conf, dpi, area)
            area = (px1 - px0) * (py1 - py0)
            if area > 0:
                projected_words.append((px0, py0, px1, py1, w[4], w[5], dpi, area))

    if not projected_words:
        return []

    # Sort primarily by confidence (descending), secondarily by area (descending)
    projected_words.sort(key=lambda x: (x[5], x[7]), reverse=True)

    kept_words = []

    def _compute_iou(b1, b2):
        # b = (x0, y0, x1, y1)
        ix0 = max(b1[0], b2[0])
        iy0 = max(b1[1], b2[1])
        ix1 = min(b1[2], b2[2])
        iy1 = min(b1[3], b2[3])

        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        intersection = iw * ih

        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0
        return intersection / union

    def _compute_intersection_over_min_area(b1, b2):
        # Stricter overlap for tiny text inside big text boxes
        ix0 = max(b1[0], b2[0])
        iy0 = max(b1[1], b2[1])
        ix1 = min(b1[2], b2[2])
        iy1 = min(b1[3], b2[3])

        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        intersection = iw * ih

        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        min_area = min(area1, area2)

        if min_area <= 0:
            return 0.0
        return intersection / min_area

    # NMS Loop
    for p_word in projected_words:
        b1 = p_word[0:4]
        is_suppressed = False

        for k_word in kept_words:
            b2 = k_word[0:4]
            # If overlap is massive (>60% of the smaller box), they represent the same text
            overlap_ratio = _compute_intersection_over_min_area(b1, b2)
            if overlap_ratio > 0.6:
                is_suppressed = True
                break

        if not is_suppressed:
            kept_words.append(p_word)

    # Re-scale kept words back to the coordinate space of the highest DPI we processed
    # (Because the rest of the pipeline expects coordinates in the rendered image space)
    target_dpi = max(dpi for dpi, _ in all_scale_words)
    inv_scale = target_dpi / BASE_DPI

    final_output = []
    for w in kept_words:
        fx0, fy0 = w[0] * inv_scale, w[1] * inv_scale
        fx1, fy1 = w[2] * inv_scale, w[3] * inv_scale
        final_output.append((fx0, fy0, fx1, fy1, w[4], w[5]))

    # Sort by reading order
    return sorted(final_output, key=lambda w: (((w[1] + w[3]) / 2), w[0]))


def _run_ocr(fitz_page, min_confidence: float = 0.3, *, dpi_list: list[int] | None = None):
    """Run OCR on a fitz page with adaptive strategy escalation.

    Performance-optimized approach:
      - Try Strategy B (minimal) first — fastest (~100ms)
      - Escalate to A (full) only if B score is insufficient
      - Try C (KMeans color slice) only as last resort
      - Early exit when composite score exceeds threshold
      - Rescue (DET/REC decoupling) only when word count < 5

    Args:
        dpi_list: DPI values to try (from PreAnalyzer). Defaults to [150, 200, 300].

    Returns:
        (all_words, img, page_h) where each word is
        (x0, y0, x1, y1, text, confidence).
        Returns (None, None, 0) on failure.
    """
    import cv2
    import numpy as np

    if dpi_list is None:
        dpi_list = [150, 200, 300]

    # Locate OCR engine
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
        logger.debug("[universal] OCR skipped: no OCR engine available")
        return None, None, 0

    # ── Auto-orientation ──
    # Try EXIF first (reliable), fall back to probe
    best_angle = _read_exif_orientation(fitz_page)
    if best_angle == 0:
        probe_pix = fitz_page.get_pixmap(dpi=100)
        probe_img = np.frombuffer(probe_pix.samples, dtype=np.uint8).reshape(probe_pix.h, probe_pix.w, probe_pix.n)
        if probe_pix.n == 3:
            probe_img = cv2.cvtColor(probe_img, cv2.COLOR_RGB2BGR)
        elif probe_pix.n == 4:
            probe_img = cv2.cvtColor(probe_img, cv2.COLOR_RGBA2BGR)
        best_angle = _probe_best_orientation(probe_img, ocr_engine)

    # ── Helper: single OCR pass ──
    def _ocr_pass(img_input, preprocess_fn, label):
        img_pp = preprocess_fn(img_input.copy())
        img_pp, _ = _deskew_image(img_pp)
        try:
            result, _ = ocr_engine(img_pp)
        except Exception as exc:
            logger.debug(f"operation: suppressed {exc}")
            return [], 0.0
        if not result:
            return [], 0.0
        words = []
        for box, text, conf in result:
            if conf < min_confidence:
                continue
            text = text.strip()
            if not text:
                continue
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            words.append(
                (
                    min(x_coords),
                    min(y_coords),
                    max(x_coords),
                    max(y_coords),
                    text,
                    conf,
                )
            )
        score = sum(w[5] for w in words)
        logger.debug(f"[OCR] {label}: {len(words)} words, score={score:.1f}")
        return words, score

    # ── Helper: Dynamic Color Slice (Strategy C: HSV/YcbCr KMeans) ──
    def _ocr_dynamic_color_slice(img_input):
        """Extract dominant ink layers using HSV KMeans, and run OCR on YCbCr Luminance."""
        import cv2
        import numpy as np

        best_slice_words, best_slice_score = [], 0.0

        # 1. YCbCr Luminance (Y channel) - best for human/OCR perception of detail
        ycbcr = cv2.cvtColor(img_input, cv2.COLOR_BGR2YCrCb)
        y_channel = ycbcr[:, :, 0]
        y_bgr = cv2.cvtColor(y_channel, cv2.COLOR_GRAY2BGR)
        w, s = _ocr_pass(y_bgr, _preprocess_minimal, "Ch-Y(Luminance)")
        if s > best_slice_score:
            best_slice_words, best_slice_score = w, s

        # 2. Dynamic HSV Hue extraction for colored overlays (e.g. red seals)
        hsv = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]

        # Subsample for fast KMeans (e.g., max 500x500 points)
        scale_k = min(1.0, 500.0 / max(img_input.shape[0:2]))
        small_h = cv2.resize(h_channel, (0, 0), fx=scale_k, fy=scale_k)
        pixels = np.float32(small_h.reshape(-1))

        # Find 3 dominant hues (Background, Text, Overlay/Seal)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

        centers = np.uint8(centers.flatten())

        # Generate a mask for each dominant hue layer and run OCR on it
        for i, center_val in enumerate(centers):
            # Create mask for pixels close to this hue
            lower_bound = max(0, int(center_val) - 15)
            upper_bound = min(179, int(center_val) + 15)

            mask = cv2.inRange(h_channel, lower_bound, upper_bound)

            # Apply mask to original luminance channel
            # We want to keep the text (dark) in the masked region
            masked_y = np.full_like(y_channel, 255)  # White background
            masked_y[mask > 0] = y_channel[mask > 0]

            slice_bgr = cv2.cvtColor(masked_y, cv2.COLOR_GRAY2BGR)
            w, s = _ocr_pass(slice_bgr, _preprocess_minimal, f"HSV-Slice-{i}(H={center_val})")

            if s > best_slice_score:
                best_slice_words, best_slice_score = w, s

        return best_slice_words, best_slice_score

    # ── Helper: DET/REC Decoupling Rescue (Strategy D) ──
    def _rescue_missing_regions(img_input, existing_words):
        """Force REC on regions that DET missed using OpenCV morphology."""
        import cv2
        import numpy as np

        from docmirror.core.ocr.vision.rapidocr_engine import get_ocr_engine

        # 1. Enhance and binarize for connected components
        gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Adaptive threshold to find dark blobs
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Connect nearby characters into text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Find contours of potential text blocks
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours
        candidate_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter noise (too small) and huge blocks (too big)
            if 15 < h < 100 and 15 < w < img_input.shape[1] * 0.8:
                # Add padding
                pad_x, pad_y = 5, 5
                cx0 = max(0, x - pad_x)
                cy0 = max(0, y - pad_y)
                cx1 = min(img_input.shape[1], x + w + pad_x)
                cy1 = min(img_input.shape[0], y + h + pad_y)

                # Check if this region is ALREADY covered by existing DET words
                is_covered = False
                for ex_w in existing_words:
                    ew_x0, ew_y0, ew_x1, ew_y1 = ex_w[0:4]
                    # Calculate IoU or partial overlap
                    ix0, iy0 = max(cx0, ew_x0), max(cy0, ew_y0)
                    ix1, iy1 = min(cx1, ew_x1), min(cy1, ew_y1)
                    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
                    if iw * ih > 0:
                        is_covered = True
                        break

                if not is_covered:
                    candidate_regions.append((cx0, cy0, cx1, cy1))

        # Force recognize these candidate regions
        rescued_words = []
        if candidate_regions:
            engine = get_ocr_engine()
            raw_rescued = engine.force_recognize_regions(img_input, candidate_regions)
            # Filter low confidence
            for rx0, ry0, rx1, ry1, text, conf in raw_rescued:
                if conf >= min_confidence:
                    rescued_words.append((rx0, ry0, rx1, ry1, text, conf))

        score = sum(w[5] for w in rescued_words)
        return rescued_words, score

    # ── Multi-Dimensional composite scoring ──
    def _composite_score(words, raw_score):
        if not words:
            return 0.0
        n = len(words)
        mean_conf = raw_score / n if n > 0 else 0.0
        unique_chars = len(set("".join(w[4] for w in words)))
        return (n * mean_conf * max(1, unique_chars)) ** (1.0 / 3.0)

    # ══════════════════════════════════════════════════════════════════════
    # Main loop: adaptive DPI + early-exit strategy escalation
    # ══════════════════════════════════════════════════════════════════════
    all_scale_results = []
    final_img = None
    final_page_h = 0

    # Early-exit threshold: skip remaining DPIs/strategies when score is good enough
    _GOOD_ENOUGH_SCORE = 8.0
    # Half-threshold: below this, try heavy strategies
    _ESCALATION_THRESHOLD = _GOOD_ENOUGH_SCORE * 0.5

    for dpi in dpi_list:
        pix = fitz_page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # Apply orientation correction
        if best_angle == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif best_angle == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif best_angle == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        page_h = img.shape[0]

        # ── Adaptive strategy escalation (lightest first) ──
        # Step 1: Try Strategy B (minimal — fastest, ~100ms)
        words_b, score_b = _ocr_pass(img, _preprocess_minimal, "Strategy-B(minimal)")
        cs_b = _composite_score(words_b, score_b)
        best_words, best_cs, winner = words_b, cs_b, "B"

        # Step 2: Only escalate to Strategy A if B is insufficient
        if cs_b < _GOOD_ENOUGH_SCORE:
            words_a, score_a = _ocr_pass(img, _preprocess_image_for_ocr, "Strategy-A(full)")
            cs_a = _composite_score(words_a, score_a)
            if cs_a > best_cs:
                best_words, best_cs, winner = words_a, cs_a, "A"

        # Step 3: Only try Strategy C (heavy KMeans) if both A and B are poor
        if best_cs < _ESCALATION_THRESHOLD:
            words_c, score_c = _ocr_dynamic_color_slice(img)
            cs_c = _composite_score(words_c, score_c)
            if cs_c > best_cs:
                best_words, best_cs, winner = words_c, cs_c, "C"

        logger.debug(f"[OCR] DPI={dpi}: winner=Strategy-{winner} (score={best_cs:.2f})")

        # Step 4: Rescue only when word count is suspiciously low
        if len(best_words) < 5:
            try:
                rescued_words, rescued_score = _rescue_missing_regions(img, best_words)
                if rescued_words:
                    best_words = list(best_words) + rescued_words
                    logger.debug(f"[OCR] DPI={dpi}: Rescued {len(rescued_words)} missing regions.")
            except Exception as exc:
                logger.debug(f"[OCR] Rescue skipped: {exc}")

        all_scale_results.append((dpi, best_words))
        final_img = img
        final_page_h = page_h

        # ── Early exit: if score is good enough, skip remaining DPIs ──
        if best_cs >= _GOOD_ENOUGH_SCORE and dpi < max(dpi_list):
            logger.debug(f"[OCR] Early exit at DPI={dpi} (score={best_cs:.2f} >= {_GOOD_ENOUGH_SCORE})")
            break

    # ── Multi-Scale NMS Fusion ──
    best_words = _merge_multi_scale_words(all_scale_results)

    if len(best_words) < 3:
        return None, None, 0

    # ── Text line merge: join fragments on the same line ──
    best_words = _merge_line_fragments(best_words)

    return best_words, final_img, final_page_h


def _detect_has_table(img, page_h: int) -> bool:
    """Check whether the page image has genuine table line structure.

    Rejects decorative border frames by verifying that detected columns
    have comparable widths (widest / median ≤ 5).
    """
    col_bounds = _detect_table_lines_hough(img, page_h, img.shape[1] if img is not None else 0)
    if not col_bounds or len(col_bounds) < 3:
        return False

    # Reject border-frame false positives: in a real table, columns
    # have roughly comparable widths.  A frame has very narrow border
    # columns flanking one huge content area.
    widths = sorted(b - a for a, b in col_bounds)
    median_w = widths[len(widths) // 2]
    max_w = widths[-1]
    if median_w > 0 and max_w / median_w > 5:
        return False

    return True


def _group_words_into_lines(words: list[tuple], y_tolerance: float = 12.0) -> list[dict]:
    """Group OCR words into text lines by y-proximity.

    Returns a list of line dicts sorted in reading order, each with:
        {"text": str, "bbox": (x0, y0, x1, y1)}
    """
    if not words:
        return []

    # Sort by y, then x
    sorted_w = sorted(words, key=lambda w: (w[1], w[0]))

    lines: list[dict] = []
    cur_words = [sorted_w[0]]
    cur_y = sorted_w[0][1]

    for w in sorted_w[1:]:
        if abs(w[1] - cur_y) <= y_tolerance:
            cur_words.append(w)
        else:
            # Finish current line
            cur_words.sort(key=lambda ww: ww[0])
            text = " ".join(ww[4] for ww in cur_words)
            x0 = min(ww[0] for ww in cur_words)
            y0 = min(ww[1] for ww in cur_words)
            x1 = max(ww[2] for ww in cur_words)
            y1 = max(ww[3] for ww in cur_words)
            lines.append({"text": text, "bbox": (x0, y0, x1, y1)})
            cur_words = [w]
            cur_y = w[1]

    # Last line
    if cur_words:
        cur_words.sort(key=lambda ww: ww[0])
        text = " ".join(ww[4] for ww in cur_words)
        x0 = min(ww[0] for ww in cur_words)
        y0 = min(ww[1] for ww in cur_words)
        x1 = max(ww[2] for ww in cur_words)
        y1 = max(ww[3] for ww in cur_words)
        lines.append({"text": text, "bbox": (x0, y0, x1, y1)})

    return lines


def ocr_extract_universal(
    fitz_page,
    page_idx: int,
    min_confidence: float = 0.3,
    *,
    page_quality: int | None = None,
    external_ocr_threshold: int | None = None,
    external_ocr_provider: Callable[..., list[tuple] | dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Universal OCR extraction — auto-detects document type.

    When ``page_quality`` is below ``external_ocr_threshold`` and
    ``external_ocr_provider`` is set, delegates to the external provider
    instead of built-in OCR (for 99% recognition targets on very poor scans).

    For table-dominant pages, delegates to ``analyze_scanned_page``
    (full backward compatibility).  For general documents (licenses,
    certificates, contracts, etc.), returns all text lines in reading
    order with real bounding boxes.

    Returns:
        dict with ``content_type`` ("table" or "general") plus:
        - table: same format as ``analyze_scanned_page``
        - general: ``{"lines": [{"text", "bbox"}, ...], "page_h", "page_w"}``
        Returns ``None`` on failure.
    """
    try:
        # ── External OCR handoff: quality too low for built-in ──
        if (
            page_quality is not None
            and external_ocr_threshold is not None
            and page_quality < external_ocr_threshold
            and external_ocr_provider is not None
        ):
            img_bgr, page_h, page_w = _render_page_to_bgr(fitz_page, dpi=200)
            try:
                out = external_ocr_provider(img_bgr, page_idx=page_idx, dpi=200, min_confidence=min_confidence)
            except Exception as e:
                logger.warning(f"[external_ocr] Provider failed on page {page_idx}: {e}")
                out = None
            if out is not None:
                if isinstance(out, dict) and out.get("content_type") in ("table", "general"):
                    if out.get("content_type") == "table":
                        from docmirror.core.ocr.ocr_postprocess import postprocess_ocr_result

                        postprocess_ocr_result(out)
                    logger.info(f"[DocMirror] Page {page_idx} delegated to external OCR (quality={page_quality})")
                    return out
                if isinstance(out, list) and out:
                    # List of (x0,y0,x1,y1,text,conf) → continue with our pipeline
                    all_words, img = out, img_bgr
                    page_w = img.shape[1]
                    has_table = _detect_has_table(img, page_h)
                    if has_table:
                        table_result = analyze_scanned_page(fitz_page, page_idx, min_confidence)
                        if table_result:
                            table_result["content_type"] = "table"
                            return table_result
                    lines = _group_words_into_lines(all_words, y_tolerance=12.0)
                    return {
                        "content_type": "general",
                        "lines": lines,
                        "page_h": page_h,
                        "page_w": page_w,
                    }
            # External failed or returned invalid → fall through to built-in

        all_words, img, page_h = _run_ocr(fitz_page, min_confidence)
        if all_words is None:
            return None

        page_w = img.shape[1] if img is not None else 0

        # Decide: table or general?
        has_table = _detect_has_table(img, page_h)

        if has_table:
            # Delegate to existing table-oriented pipeline
            table_result = analyze_scanned_page(fitz_page, page_idx, min_confidence)
            if table_result:
                table_result["content_type"] = "table"
                return table_result
            # If table pipeline fails, fall through to general

        # General document: output all text lines in reading order
        lines = _group_words_into_lines(all_words, y_tolerance=12.0)

        return {
            "content_type": "general",
            "lines": lines,
            "page_h": page_h,
            "page_w": page_w,
        }

    except Exception as e:
        logger.warning(f"[universal] OCR error on page {page_idx}: {e}")
        return None
