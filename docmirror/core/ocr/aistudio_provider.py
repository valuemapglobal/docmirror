# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
AI Studio Layout-Parsing API — External OCR Provider
====================================================

Integrates the AI Studio layout-parsing HTTP API as an external OCR provider.
Used when image quality is below ``external_ocr_quality_threshold`` and
``DOCMIRROR_EXTERNAL_OCR_PROVIDER`` is set to
``docmirror.core.ocr.aistudio_provider:call_aistudio_layout_ocr``.

Requires: ``requests`` (install with ``pip install docmirror[external-ocr]``).
Image encoding uses OpenCV when available (from the ``ocr`` extra).

Environment variables:
    DOCMIRROR_AISTUDIO_OCR_API_URL   — API endpoint (default from integration spec)
    DOCMIRROR_AISTUDIO_OCR_TOKEN    — Bearer token for Authorization
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default from integration spec; override with env
DEFAULT_API_URL = "https://j0g4k9d1a8x3mfx1.aistudio-app.com/layout-parsing"


def _encode_image_bgr_to_base64(image_bgr) -> str | None:
    """Encode BGR numpy array to PNG base64. Requires OpenCV."""
    try:
        import cv2

        _, buf = cv2.imencode(".png", image_bgr)
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except ImportError:
        logger.debug("OpenCV not available for aistudio provider image encode")
        return None
    except Exception as e:
        logger.warning(f"[aistudio] Image encode failed: {e}")
        return None


def call_aistudio_layout_ocr(
    image_bgr,
    *,
    page_idx: int = 0,
    dpi: int = 200,
    min_confidence: float = 0.3,
    file_type: int = 1,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_chart_recognition: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Call AI Studio layout-parsing API and return result in DocMirror OCR format.

    Args:
        image_bgr: Page/image as BGR numpy array (OpenCV convention).
        page_idx: Page index (for logging).
        dpi: Render DPI (informational; image is already rendered).
        min_confidence: Ignored by this API; kept for contract compatibility.
        file_type: 0 = PDF, 1 = image. This provider always receives an image, so 1.
        use_doc_orientation_classify: Optional API flag.
        use_doc_unwarping: Optional API flag.
        use_chart_recognition: Optional API flag.

    Returns:
        dict with ``content_type="general"``, ``lines`` (one item with full markdown text),
        ``page_h``, ``page_w``; or ``None`` on failure.
    """
    try:
        import requests
    except ImportError:
        logger.warning("[aistudio] requests not installed. pip install docmirror[external-ocr]")
        return None

    api_url = os.environ.get("DOCMIRROR_AISTUDIO_OCR_API_URL", DEFAULT_API_URL).strip()
    token = os.environ.get("DOCMIRROR_AISTUDIO_OCR_TOKEN", "").strip()
    if not token:
        logger.warning("[aistudio] DOCMIRROR_AISTUDIO_OCR_TOKEN not set")
        return None

    file_data = _encode_image_bgr_to_base64(image_bgr)
    if not file_data:
        return None

    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "file": file_data,
        "fileType": file_type,
        "useDocOrientationClassify": use_doc_orientation_classify,
        "useDocUnwarping": use_doc_unwarping,
        "useChartRecognition": use_chart_recognition,
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
    except requests.RequestException as e:
        logger.warning(f"[aistudio] Request failed on page {page_idx}: {e}")
        return None

    if response.status_code != 200:
        logger.warning(f"[aistudio] API returned {response.status_code} on page {page_idx}")
        return None

    try:
        data = response.json()
    except ValueError as e:
        logger.warning(f"[aistudio] Invalid JSON on page {page_idx}: {e}")
        return None

    result = data.get("result", data)
    layout_results = result.get("layoutParsingResults", [])
    if not layout_results:
        logger.debug(f"[aistudio] No layoutParsingResults on page {page_idx}")
        return None

    text_parts = []
    for res in layout_results:
        md = res.get("markdown") or {}
        text_parts.append((md.get("text") or "").strip())
    full_text = "\n\n".join(t for t in text_parts if t)

    try:
        h, w = image_bgr.shape[:2]
    except Exception:
        h, w = 1, 1

    return {
        "content_type": "general",
        "lines": [{"text": full_text, "bbox": (0, 0, w, h)}],
        "page_h": h,
        "page_w": w,
    }
