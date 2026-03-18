# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Image to Virtual PDF Converter
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def image_to_virtual_pdf(image_path: Path) -> fitz.Document:
    """Convert image to a virtual single-page PDF, pre-scaling large images to max 4096px."""
    import fitz

    # Read image
    img_doc = fitz.open(str(image_path))
    try:
        if len(img_doc) == 0:
            raise ValueError(f"Cannot open image: {image_path}")

        # Pre-scale large images (prevent memory explosion)
        page = img_doc[0]
        w, h = page.rect.width, page.rect.height
        max_dim = 4096
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            logger.info(f"[DocMirror] Image pre-scaled: {w:.0f}x{h:.0f} -> {w * scale:.0f}x{h * scale:.0f}")
            # use pixmap scaling
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)
            # Rebuild from scaled pixmap
            new_doc = fitz.open()
            new_page = new_doc.new_page(width=pix.width, height=pix.height)
            new_page.insert_image(new_page.rect, pixmap=pix)
            return new_doc

        # Normal size: convert directly to PDF
        pdf_bytes = img_doc.convert_to_pdf()
        return fitz.open("pdf", pdf_bytes)
    finally:
        img_doc.close()
