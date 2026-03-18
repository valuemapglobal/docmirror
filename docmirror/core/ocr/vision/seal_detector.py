# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Seal Detection & Polar Coordinate Unwarping
=============================================

Based on OpenCV (cv2).

Supports two detection modes:
    1. **Colour seal**: HSV red-channel segmentation (for colour scans).
    2. **Greyscale seal**: grey-level thresholding + circularity filtering
       (for B&W / greyscale scans).

Solves the problem of extremely curved seals (e.g. bank chops) that
cannot be recognised by standard OCR.  Uses ``cv2.warpPolar`` for
polar-coordinate transformation to "straighten" the curved text into
a horizontal image strip.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("OpenCV is not installed. Seal detection will be skipped.")


class SealDetector:
    """Seal detector & polar-coordinate straightener — supports both colour
    and greyscale scans."""

    def __init__(self):
        # Red occupies two disjoint hue ranges in HSV colour space
        self.lower_red1 = np.array([0, 50, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def detect_seal(self, image_bgr: np.ndarray) -> dict[str, Any]:
        """Detect a seal and return detection metadata (no polar unwarping).

        Returns:
            {
                "has_seal": bool,
                "center": (x, y) | None,
                "radius": int | None,
                "bbox": (x1, y1, x2, y2) | None,
                "mode": "color" | "gray" | None,
            }
        """
        if not _CV2_AVAILABLE:
            return {"has_seal": False, "center": None, "radius": None, "bbox": None, "mode": None}

        # 1. Try colour (red) seal detection first
        result = self._detect_color_seal(image_bgr)
        if result["has_seal"]:
            return result

        # 2. Fallback to greyscale (B&W scan) seal detection
        return self._detect_gray_seal(image_bgr)

    def unwarp_circular_seal(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Extract the seal from the image and flatten it via polar-coordinate
        transformation into a horizontal text strip."""
        info = self.detect_seal(image_bgr)
        if not info["has_seal"]:
            return None

        try:
            center = info["center"]
            radius = info["radius"]
            h, w = image_bgr.shape[:2]
            x1 = max(0, center[0] - radius)
            y1 = max(0, center[1] - radius)
            x2 = min(w, center[0] + radius)
            y2 = min(h, center[1] + radius)

            roi = image_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            local_center = (center[0] - x1, center[1] - y1)
            circumference = int(2 * np.pi * radius)
            unwarped = cv2.warpPolar(
                roi,
                dsize=(radius, circumference),
                center=local_center,
                maxRadius=radius,
                flags=cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR,
            )
            unwarped = cv2.rotate(unwarped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return unwarped

        except Exception as e:
            logger.error(f"Seal unwarping failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Colour seal detection (red HSV)
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_color_seal(self, image_bgr: np.ndarray) -> dict[str, Any]:
        """Detect a colour seal via HSV red-channel segmentation."""
        empty = {"has_seal": False, "center": None, "radius": None, "bbox": None, "mode": None}
        try:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            kernel = np.ones((3, 3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return empty

            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < 1000:
                return empty

            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            center = (int(cx), int(cy))
            r = int(radius)
            return {
                "has_seal": True,
                "center": center,
                "radius": r,
                "bbox": (center[0] - r, center[1] - r, center[0] + r, center[1] + r),
                "mode": "color",
            }
        except Exception as exc:
            logger.debug(f"operation: suppressed {exc}")
            return empty

    # ─────────────────────────────────────────────────────────────────────────
    # Greyscale seal detection (for B&W scans)
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_gray_seal(self, image_bgr: np.ndarray) -> dict[str, Any]:
        """Greyscale circular-contour detection.

        Algorithm:
          1. Convert to greyscale, apply Gaussian blur for denoising.
          2. Adaptive threshold + morphological operations to retain only
             mid-grey regions (excluding pure-black text and white background).
          3. Find contours in the thresholded image, filter by circularity.
          4. Select the largest contour with circularity > 0.5 as the seal.
        """
        empty = {"has_seal": False, "center": None, "radius": None, "bbox": None, "mode": None}
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Search only the top-right quadrant (seals are typically placed there)
            roi_y1, roi_y2 = 0, h // 3
            roi_x1, roi_x2 = w // 2, w
            gray_roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]

            # Gaussian blur for noise reduction
            blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

            # Extract mid-grey regions (exclude pure-black text < 80
            # and white background > 200)
            # Scanned seals typically fall in the ~80–200 grey range
            mask = cv2.inRange(blurred, 80, 200)

            # Morphological close to connect broken arcs, open to remove speckle
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return empty

            # Find the contour with the highest circularity and sufficient area
            best = None
            best_score = 0
            min_area = 2000  # Minimum area threshold
            min_circularity = 0.3

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter < 1:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < min_circularity:
                    continue

                # Combined score: area × circularity
                score = area * circularity
                if score > best_score:
                    best_score = score
                    best = cnt

            if best is None:
                return empty

            (cx, cy), radius = cv2.minEnclosingCircle(best)
            # Convert back to full-image coordinates
            abs_cx = int(cx) + roi_x1
            abs_cy = int(cy) + roi_y1
            r = int(radius)

            area = cv2.contourArea(best)
            perimeter = cv2.arcLength(best, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            logger.info(
                f"[SealDetector] Gray seal: center=({abs_cx},{abs_cy}), "
                f"r={r}, area={area:.0f}, circularity={circularity:.3f}"
            )

            return {
                "has_seal": True,
                "center": (abs_cx, abs_cy),
                "radius": r,
                "bbox": (abs_cx - r, abs_cy - r, abs_cx + r, abs_cy + r),
                "mode": "gray",
            }
        except Exception as e:
            logger.debug(f"Gray seal detection error: {e}")
            return empty


# Singleton accessor
_default_seal_detector: SealDetector | None = None


def get_seal_detector() -> SealDetector:
    global _default_seal_detector
    if _default_seal_detector is None:
        _default_seal_detector = SealDetector()
    return _default_seal_detector
