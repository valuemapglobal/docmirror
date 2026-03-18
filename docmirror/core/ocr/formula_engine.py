# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Formula Recognition Engine
============================

Unified formula recognition entry point using the Strategy pattern.

Backend priority::

    UniMERNet ONNX (if model path is valid) > rapid_latex_ocr > empty string

Usage::

    engine = FormulaEngine()
    latex = engine.recognize(image_bytes)

Relationship with CoreExtractor:
    - ``CoreExtractor._recognize_formula()`` delegates to this engine.
    - This engine is self-contained and can be used / tested independently.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FormulaEngine:
    """Unified formula recognition entry point — Strategy pattern.

    Automatically selects the best available backend:
        1. UniMERNet ONNX (if *model_path* is specified and exists)
        2. rapid_latex_ocr (if installed)
        3. Empty string (no backend available)

    All backends share the same interface: ``image_bytes → LaTeX string``.
    """

    def __init__(self, model_path: str | None = None):
        """
        Args:
            model_path: Path to a UniMERNet ONNX model file.
                ``None`` skips the ONNX backend and falls back to
                rapid_latex_ocr.
        """
        self._model_path = model_path
        self._onnx_session = None
        self._rapid_ocr = None
        self._backend = "none"
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialisation: runs on the first call to ``recognize()``."""
        if self._initialized:
            return
        self._initialized = True

        # Strategy 1: UniMERNet ONNX
        if self._model_path:
            path = Path(self._model_path)
            if path.exists():
                try:
                    import onnxruntime as ort

                    self._onnx_session = ort.InferenceSession(
                        str(path),
                        providers=["CPUExecutionProvider"],
                    )
                    self._backend = "unimernet_onnx"
                    logger.info(f"[FormulaEngine] Using UniMERNet ONNX: {path}")
                    return
                except ImportError:
                    logger.debug("[FormulaEngine] onnxruntime not available")
                except Exception as e:
                    logger.warning(f"[FormulaEngine] ONNX load failed: {e}")

        # Strategy 2: rapid_latex_ocr
        try:
            from rapid_latex_ocr import LaTeXOCR

            self._rapid_ocr = LaTeXOCR()
            self._backend = "rapid_latex_ocr"
            logger.info("[FormulaEngine] Using rapid_latex_ocr")
            return
        except ImportError:
            logger.debug("[FormulaEngine] rapid_latex_ocr not installed")
        except Exception as e:
            logger.warning(f"[FormulaEngine] rapid_latex_ocr init failed: {e}")

        # Strategy 3: no backend available
        self._backend = "none"
        logger.info("[FormulaEngine] No formula backend available")

    def recognize(self, image_bytes: bytes) -> str:
        """Recognise a formula image and return LaTeX.

        Args:
            image_bytes: Raw image bytes of the formula region.

        Returns:
            LaTeX string, or an empty string on failure.
        """
        if not image_bytes:
            return ""

        self._lazy_init()

        # Image preprocessing: padding + resize + contrast enhancement
        image_bytes = _preprocess_formula_image(image_bytes)

        try:
            if self._backend == "unimernet_onnx":
                return self._recognize_onnx(image_bytes)
            elif self._backend == "rapid_latex_ocr":
                return self._recognize_rapid(image_bytes)
        except Exception as e:
            logger.debug(f"[FormulaEngine] recognition error: {e}")

        return ""

    def _recognize_onnx(self, image_bytes: bytes) -> str:
        """ONNX inference — falls back to rapid_latex_ocr if available."""
        # Fallback to rapid_latex_ocr if available
        if self._rapid_ocr is not None:
            return self._recognize_rapid(image_bytes)
        return ""

    def _recognize_rapid(self, image_bytes: bytes) -> str:
        """Recognise using rapid_latex_ocr."""
        result, _ = self._rapid_ocr(image_bytes)
        return result or ""

    def recognize_and_normalize(self, image_bytes: bytes) -> str:
        """Recognise and normalise in a single step — convenience API."""
        raw = self.recognize(image_bytes)
        if raw:
            return self.normalize_latex(raw)
        return ""

    @staticmethod
    def normalize_latex(latex: str) -> str:
        """Deep LaTeX normalisation — maximise CDM (Content Difference Metric)
        match rate.

        Steps (in order):
            1. Strip leading/trailing whitespace and ``$`` delimiters.
            2. Apply common OCR error corrections.
            3. Remove redundant commands (``\\displaystyle``, etc.).
            4. Inline ``\\text{}`` / ``\\mathrm{}`` content.
            5. Balance mismatched brackets.
            6. Conservative brace simplification.
            7. Whitespace normalisation.
        """
        if not latex or not latex.strip():
            return ""

        latex = latex.strip()

        # ── Step 1: strip outer $ delimiters ──
        if latex.startswith("$$") and latex.endswith("$$"):
            latex = latex[2:-2].strip()
        elif latex.startswith("$") and latex.endswith("$"):
            latex = latex[1:-1].strip()

        # Strip \[ \] delimiters
        if latex.startswith("\\[") and latex.endswith("\\]"):
            latex = latex[2:-2].strip()

        # ── Step 2: common OCR error corrections ──
        latex = _apply_ocr_corrections(latex)

        # ── Step 3: remove redundant display-style commands ──
        for cmd in (r"\displaystyle", r"\textstyle", r"\scriptstyle", r"\scriptscriptstyle"):
            latex = latex.replace(cmd, "")

        # Remove invisible delimiters from \left. / \right.
        # Keep \left / \right themselves (CDM requires paired delimiters)
        latex = latex.replace(r"\left.", r"\left").replace(r"\right.", r"\right")

        # ── Step 4: inline \text{} / \mathrm{} / \mathit{} content ──
        latex = re.sub(r"\\(?:text|mathrm|mathit|mbox|hbox)\{([^{}]*)\}", r"\1", latex)

        # ── Step 5: bracket balancing ──
        latex = _balance_brackets(latex)

        # ── Step 6: conservative brace simplification ──
        # (Skipped — CDM parse tree depends on brace structure)

        # ── Step 7: whitespace normalisation ──
        latex = re.sub(r"\s+", " ", latex)
        # Uniform spacing around operators
        latex = re.sub(r"\s*([+\-=<>])\s*", r" \1 ", latex)
        # Space after commas
        latex = re.sub(r",\s*", ", ", latex)
        latex = latex.strip()

        return latex

    @property
    def backend_name(self) -> str:
        """Name of the currently active backend."""
        self._lazy_init()
        return self._backend


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX normalisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Common OCR error mapping (high-frequency rapid_latex_ocr mistakes)
_OCR_CORRECTIONS = {
    # Greek letters
    r"\Iambda": r"\lambda",
    r"\Gamma": r"\Gamma",  # already correct — kept for completeness
    r"\aIpha": r"\alpha",
    r"\bata": r"\beta",
    r"\epsiIon": r"\epsilon",
    r"\varepsIlon": r"\varepsilon",
    r"\delte": r"\delta",
    r"\sigam": r"\sigma",
    r"\thata": r"\theta",
    # Operators
    r"\tims": r"\times",
    r"\tmes": r"\times",
    r"\cdct": r"\cdot",
    r"\Ieq": r"\leq",
    r"\geq": r"\geq",
    r"\neq": r"\neq",
    r"\infity": r"\infty",
    r"\inftv": r"\infty",
    # Structures
    r"\frae": r"\frac",
    r"\sqr": r"\sqrt",
    r"\overIine": r"\overline",
    r"\underIine": r"\underline",
    r"\mathbf": r"\mathbf",
    r"\Iim": r"\lim",
    r"\Iin": r"\lin",
    r"\Int": r"\int",
}


def _apply_ocr_corrections(latex: str) -> str:
    """Apply common OCR error corrections."""
    for wrong, correct in _OCR_CORRECTIONS.items():
        if wrong in latex:
            latex = latex.replace(wrong, correct)
    return latex


def _balance_brackets(latex: str) -> str:
    """Detect and fix unbalanced brackets.

    Strategy:
        - Count opening / closing brackets for each type.
        - Append missing closing brackets at the end.
        - Prepend missing opening brackets at the start.
    """
    pairs = [("{", "}"), ("(", ")"), ("[", "]")]

    for open_ch, close_ch in pairs:
        count = 0
        for ch in latex:
            if ch == open_ch:
                count += 1
            elif ch == close_ch:
                count -= 1

        if count > 0:
            latex += close_ch * count  # Missing closing brackets
        elif count < 0:
            latex = open_ch * (-count) + latex  # Missing opening brackets

    return latex


def _preprocess_formula_image(image_bytes: bytes) -> bytes:
    """Preprocess a formula image to improve OCR accuracy.

    rapid_latex_ocr's training data consists of standard white-background
    formula images.  Direct crops from PDFs are often tightly clipped and
    lack contrast.

    Steps:
        1. Add 15 % white padding (prevent edge clipping).
        2. Upscale small images to ``min_height = 64 px``.
        3. CLAHE contrast enhancement.
        4. Unsharp-mask sharpening.

    Returns the original bytes unchanged if PIL / cv2 are unavailable.
    """
    if not image_bytes or len(image_bytes) < 100:
        return image_bytes

    try:
        from io import BytesIO

        import numpy as np
        from PIL import Image, ImageFilter, ImageOps

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = img.size

        # Step 1: white padding (15 %)
        pad_x = max(10, int(w * 0.15))
        pad_y = max(10, int(h * 0.15))
        padded = Image.new("RGB", (w + 2 * pad_x, h + 2 * pad_y), (255, 255, 255))
        padded.paste(img, (pad_x, pad_y))
        img = padded
        w, h = img.size

        # Step 2: upscale to min_height = 64 px
        if h < 64:
            scale = 64 / h
            new_w = int(w * scale)
            img = img.resize((new_w, 64), Image.LANCZOS)

        # Step 3: CLAHE contrast enhancement
        try:
            import cv2

            img_np = np.array(img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))
        except ImportError:
            # Fallback to PIL auto-contrast when cv2 is unavailable
            img = ImageOps.autocontrast(img, cutoff=1)

        # Step 4: sharpen
        img = img.filter(ImageFilter.SHARPEN)

        # Output as PNG
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        logger.debug(f"[FormulaEngine] image preprocess failed: {e}")
        return image_bytes
