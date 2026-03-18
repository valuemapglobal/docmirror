# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
DocMirror Global Settings
=========================

Centralized system-level configuration for the DocMirror parsing engine.

All settings have sensible defaults and can be overridden via environment
variables using the ``DOCMIRROR_`` prefix. The ``from_env()`` classmethod
reads the current environment and returns a configured instance.

Configuration groups:
    - **Enhancement**: Default pipeline mode.
    - **Performance**: Page limits, OCR resolution, and language detection.
    - **Validation**: Pass/fail thresholds for the quality validator.
    - **Model paths**: Optional paths to AI model weights (layout, reading
      order, formula recognition). When ``None``, rule-based fallbacks
      are used instead.
    - **Pipeline strategy**: How to handle middleware failures (skip vs abort).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OCRHyperParams:
    """
    Physically justified OCR preprocessing hyperparameters.

    Every value here has a documented rationale. Changing any value
    should require updating the justification — enforcing Deutsch's
    'hard to vary' principle.
    """

    # ── Adaptive Upscale ──
    # Rationale: RapidOCR's detection model (DBNet) is trained on 640×640 inputs.
    # Images below 500px on their shortest side produce feature maps too coarse
    # for the 3×3 convolution kernels to separate adjacent characters.
    upscale_threshold_low: int = 500  # Below this → 4× upscale
    upscale_threshold_mid: int = 1000  # Below this → 2× upscale
    upscale_factor_low: float = 4.0  # Matches DBNet's receptive field requirement
    upscale_factor_mid: float = 2.0

    # ── Gamma Correction ──
    # Rationale: Mean brightness < 80/255 (31%) indicates a severely underexposed
    # scan. Gamma=0.6 maps the [0,80] input range onto [0,148], restoring
    # discriminability in the dark region where OCR character edges live.
    dark_image_brightness_threshold: int = 80
    dark_image_gamma: float = 0.6

    # ── Histogram Stretch ──
    # Rationale: Standard deviation < 25 in an 8-bit image means the entire
    # dynamic range is compressed into ~10% of the [0,255] spectrum. Below
    # this, CLAHE alone cannot restore sufficient edge contrast.
    low_contrast_std_threshold: float = 25.0
    histogram_percentile_lo: float = 1.0  # Clip darkest 1% — removes sensor noise
    histogram_percentile_hi: float = 99.0  # Clip brightest 1% — removes highlights

    # ── Red Seal Removal (HSV bounds) ──
    # Rationale: Red hue wraps around 0° in HSV. These ranges (0–10° and 160–180°)
    # cover the full red lobe. S≥70, V≥50 excludes desaturated pinks and dark shadows.
    red_hue_range_1: tuple = (0, 10)
    red_hue_range_2: tuple = (160, 180)
    red_saturation_min: int = 70
    red_value_min: int = 50

    # ── Row Clustering ──
    # Rationale: 40% vertical overlap of character height is the empirical minimum
    # for CJK characters (which are square) to be considered "same line". Below
    # 40%, subscripts and superscripts get incorrectly merged.
    row_overlap_ratio: float = 0.4

    # ── Line Fragment Merging ──
    # Rationale: Two words with >50% vertical overlap AND horizontal gap <1.5×
    # average char height belong to the same line. 1.5× accommodates CJK
    # inter-character spacing (which is wider than Latin).
    line_merge_v_overlap_ratio: float = 0.5
    line_merge_h_gap_multiplier: float = 1.5

    # ── NMS Fusion ──
    # Rationale: 60% intersection-over-min-area means the smaller box is
    # substantially enclosed by the larger one — they represent the same text.
    nms_overlap_threshold: float = 0.6

    # ── Minimum Words for Valid OCR ──
    # Rationale: A page with fewer than 10 recognized words at the initial DPI
    # is likely a low-quality scan that needs higher resolution. 3 words is the
    # absolute minimum to produce any meaningful output (e.g., a stamp).
    min_words_initial_pass: int = 10
    min_words_final: int = 3

    # ── Multi-Scale DPI Passes ──
    dpi_passes: tuple = (150, 200, 300)

    # ── Dynamic Color Slicing ──
    kmeans_clusters: int = 3  # Background, Text, Overlay — 3 is sufficient for documents
    hue_tolerance: int = 15  # ±15° in HSV hue space covers a single color family


@dataclass
class DocMirrorSettings:
    """
    Global configuration for DocMirror.

    Instantiate directly with custom values, or use ``from_env()`` to
    load from environment variables. Use ``to_dict()`` to inject into
    the Orchestrator pipeline configuration.
    """

    # ── Enhancement settings ──
    default_enhance_mode: str = "standard"  # "raw" | "standard" | "full"

    # ── Performance limits ──
    max_pages: int = 200  # Maximum pages to process per document
    max_page_concurrency: int = 1  # Page-level concurrency (1=sequential; >1 enables layout + extraction parallel)
    ocr_dpi: int = 150  # Default DPI for rendering pages to images for OCR
    ocr_retry_dpi: int = 300  # Higher DPI used when initial OCR produces poor results
    ocr_language: str = "auto"  # "auto" = auto-detect; or specify e.g. "zh", "en"

    # ── Validation thresholds ──
    validator_pass_threshold: float = 0.7  # Minimum score to consider parsing successful

    # ── Logging ──
    log_level: str = "INFO"

    # ── Pipeline error handling ──
    fail_strategy: str = "skip"  # "skip" = ignore failed middlewares; "abort" = halt pipeline

    # ── Optional AI model file paths ──
    # When None, DocMirror uses rule-based fallbacks instead of AI models
    layout_model_path: str | None = None  # DocLayout-YOLO ONNX model path
    reading_order_model_path: str | None = None  # LayoutReader ONNX model path
    formula_model_path: str | None = None  # Pix2Tex / UniMERNet ONNX model path

    # ── Model inference parameters ──
    model_render_dpi: int = 200  # DPI for rendering pages before DocLayout-YOLO inference

    # ── OCR Hyperparameters (physically justified) ──
    ocr_params: OCRHyperParams = field(default_factory=OCRHyperParams)

    # ── Constructor Theory: Impossible Transformation Guards ──
    min_file_size: int = 512  # Below 512 bytes, no document can contain meaningful content
    max_file_size: int = 500_000_000  # 500MB hard limit — prevents OOM
    min_image_dimension: int = 50  # 50px — below this, OCR receptive fields cannot function

    # ── Table extraction: RapidTable layer (slow ~10s/page) ──
    # When document has more than this many pages, skip RapidTable entirely (G4).
    table_rapid_max_pages: int | None = None  # None = no limit
    # Only try RapidTable when upstream layer confidence is below this (0–1). Default 0.3.
    table_rapid_min_confidence_threshold: float = 0.3

    # ── External OCR (low-quality handoff) ──
    # When image quality is below this (0–100), delegate to external OCR instead of built-in.
    # 80 = use external OCR for quality < 80 (e.g. blur, skew, poor lighting).
    external_ocr_quality_threshold: int = 80
    # Optional: "module:callable" to invoke for external OCR (e.g. "myapp.ocr:call_cloud_ocr").
    # Callable(image_bgr, page_idx=0, dpi=200, **kwargs) -> list of (x0,y0,x1,y1,text,conf) or dict.
    external_ocr_provider: str | None = None

    @classmethod
    def from_env(cls) -> DocMirrorSettings:
        """
        Create a DocMirrorSettings instance from environment variables.

        Reads ``DOCMIRROR_*`` environment variables and falls back to
        default values when variables are not set.

        Supported env vars:
            DOCMIRROR_ENHANCE_MODE       → default_enhance_mode
            DOCMIRROR_MAX_PAGES          → max_pages
            DOCMIRROR_VALIDATOR_THRESHOLD → validator_pass_threshold
            DOCMIRROR_LOG_LEVEL          → log_level
            DOCMIRROR_FAIL_STRATEGY      → fail_strategy
        """
        instance = cls(
            default_enhance_mode=os.getenv("DOCMIRROR_ENHANCE_MODE", "standard"),
            max_pages=int(os.getenv("DOCMIRROR_MAX_PAGES", "200")),
            max_page_concurrency=int(os.getenv("DOCMIRROR_MAX_PAGE_CONCURRENCY", "1")),
            validator_pass_threshold=float(os.getenv("DOCMIRROR_VALIDATOR_THRESHOLD", "0.7")),
            log_level=os.getenv("DOCMIRROR_LOG_LEVEL", "INFO"),
            fail_strategy=os.getenv("DOCMIRROR_FAIL_STRATEGY", "skip"),
            table_rapid_max_pages=(int(v) if (v := os.getenv("DOCMIRROR_TABLE_RAPID_MAX_PAGES", "").strip()) else None),
            table_rapid_min_confidence_threshold=float(
                os.getenv("DOCMIRROR_TABLE_RAPID_MIN_CONFIDENCE_THRESHOLD", "0.3")
            ),
            external_ocr_quality_threshold=int(os.getenv("DOCMIRROR_EXTERNAL_OCR_QUALITY_THRESHOLD", "80")),
            external_ocr_provider=((v := os.getenv("DOCMIRROR_EXTERNAL_OCR_PROVIDER", "").strip()) or None),
        )
        logger.info(
            f"[Config] Initialized global settings: enhance_mode='{instance.default_enhance_mode}', "
            f"max_concurrency={instance.max_page_concurrency}, fail_strategy='{instance.fail_strategy}'"
        )
        return instance

    def to_dict(self) -> dict[str, Any]:
        """
        Convert settings to a dict suitable for Orchestrator config injection.

        Returns a dict keyed by middleware class name, with each value
        containing the relevant configuration subset for that middleware.
        """
        return {
            "enhance_mode": self.default_enhance_mode,
            "SceneDetector": {},
            "Validator": {"pass_threshold": self.validator_pass_threshold},
        }


# Module-level singleton: initialized once from environment variables
# at import time. Can be overridden by creating a new instance.
default_settings = DocMirrorSettings.from_env()
