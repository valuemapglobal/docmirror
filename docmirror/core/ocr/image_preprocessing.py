# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Image Preprocessing for OCR
============================

Two preprocessing strategies for scanned document images:
  - **Strategy A (full)**: Heavy preprocessing with binarisation.
  - **Strategy B (minimal)**: Light preprocessing preserving gradients.
"""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# T3-2: Module-level import caching (avoids ~1-2ms per function call)
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore


def _base_preprocess(img_bgr):
    """Shared preprocessing base: upscale + gamma + histogram + seal removal.

    Both Strategy A and B share these steps. Extracting them avoids
    ~100-200ms of duplicated computation when both strategies are tried.

    Returns: (preprocessed_bgr, height, width)
    """

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

    # ── Red seal removal ──
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

    return img_bgr, h, w


def preprocess_minimal(img_bgr) -> np.ndarray:
    """Minimal preprocessing (Strategy B): preserves maximum information.

    Delegates shared steps to _base_preprocess, then applies CLAHE + bilateral.
    Output: grayscale image (NOT binarised).
    """

    img_bgr, h, w = _base_preprocess(img_bgr)

    # ── CLAHE on grayscale ──
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ── Light bilateral smoothing ──
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # Return as BGR (3-channel grayscale) — NO binarisation
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def preprocess_image_for_ocr(img_bgr) -> np.ndarray:
    """Enhanced image preprocessing for OCR (v4 — full pipeline).

    Delegates shared base steps to _base_preprocess(), then applies:
    edge pad → KMeans color slice → watermark removal → border crop →
    perspective correction → background equalization → CLAHE → bilateral →
    unsharp mask → binarisation voting → line removal → morph repair.
    """

    # ── Shared base: upscale + gamma + histogram + seal removal ──
    img_bgr, h, w = _base_preprocess(img_bgr)

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
    h, w = img_bgr.shape[:2]

    # ── Step A: Dynamic Color Slicing (KMeans HSV) ──
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

    # ── Step E: watermark removal ──
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

    # ── Step B: decorative border cropping ──
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

    # ── Step D: perspective correction ──
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

    # ── Step C: background lighting equalisation ──
    try:
        gray_bg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg = cv2.GaussianBlur(gray_bg, (0, 0), sigmaX=51)
        bg[bg < 1] = 1
        normalised = (gray_bg / bg * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(normalised, cv2.COLOR_GRAY2BGR)
        h, w = img_bgr.shape[:2]
    except Exception as e:
        logger.debug(f"[OCR] Background equalisation skipped: {e}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Step 1: adaptive CLAHE ──
    contrast = gray.std()
    clip_limit = 3.0 if contrast < 40 else 2.0 if contrast < 80 else 1.5
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ── Step 2: bilateral filtering ──
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # ── Step 3: unsharp mask sharpening ──
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # ── Step 4: multi-method binarisation voting ──
    _, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8)
    bin_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    vote = bin_otsu.astype(np.uint16) + bin_gauss.astype(np.uint16) + bin_mean.astype(np.uint16)
    binary = np.where(vote >= 2 * 255, 255, 0).astype(np.uint8)

    # ── Step F: table/grid line removal ──
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

    # ── Step 5: Morphological Text Repair ──
    inv_binary = cv2.bitwise_not(binary)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    inv_binary = cv2.morphologyEx(inv_binary, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    inv_binary = cv2.morphologyEx(inv_binary, cv2.MORPH_OPEN, noise_kernel, iterations=1)
    binary = cv2.bitwise_not(inv_binary)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def deskew_image(img_bgr) -> tuple:
    """Deskew a page image by detecting and correcting rotation (±20°).

    Uses contour-based minimum-area-rectangle angle estimation with
    median filtering for robustness.

    Returns:
        (corrected_image, skew_angle_degrees)
    """
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
