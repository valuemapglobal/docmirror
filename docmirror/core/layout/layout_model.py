"""
Layout Detection Model
======================

Wrapper around RapidLayout (ONNX) for model-based page layout detection.
Falls back gracefully when the model or its dependencies are unavailable.

Source: https://github.com/RapidAI/RapidLayout
Built-in models: DocLayout-YOLO, PP-Layout-CDLA, PP-DocLayoutV3, and 12 others.

Usage::

    detector = LayoutDetector()           # Default: DOCLAYOUT_DOCSTRUCTBENCH
    detector = LayoutDetector("cdla")     # Chinese document layout
    regions = detector.detect(page_image)

Requirements:
    - rapid-layout  (pip install rapid-layout)
    - onnxruntime   (already present as a RapidOCR dependency)
"""
from __future__ import annotations


import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

try:
    from rapid_layout import RapidLayout, RapidLayoutInput, ModelType
    HAS_RAPID_LAYOUT = True
except ImportError:
    HAS_RAPID_LAYOUT = False

# ---------------------------------------------------------------------------
# RapidLayout class_name → DocMirror zone type mapping
# ---------------------------------------------------------------------------
_CATEGORY_MAP = {
    # DocLayout-YOLO (DOCLAYOUT_DOCSTRUCTBENCH) — 10 classes
    "title": "title",
    "plain text": "text",
    "abandon": "abandon",
    "figure": "data_table",           # figures are treated as standalone regions too
    "figure_caption": "text",
    "table": "data_table",
    "table_caption": "text",
    "table_footnote": "text",
    "isolate_formula": "formula",
    "formula_caption": "text",
    # PP-Layout-CDLA — additional classes
    "text": "text",
    "header": "title",
    "footer": "footer",
    "reference": "text",
    "equation": "formula",
}

# Model type alias map (user-friendly string → ModelType enum name)
_MODEL_ALIASES = {
    "doclayout": "DOCLAYOUT_DOCSTRUCTBENCH",
    "doclayout_docstructbench": "DOCLAYOUT_DOCSTRUCTBENCH",
    "cdla": "PP_LAYOUT_CDLA",
    "publaynet": "PP_LAYOUT_PUBLAYNET",
    "table": "PP_LAYOUT_TABLE",
    "paper": "YOLOV8N_LAYOUT_PAPER",
    "report": "YOLOV8N_LAYOUT_REPORT",
    "general6": "YOLOV8N_LAYOUT_GENERAL6",
    "docstructbench": "DOCLAYOUT_DOCSTRUCTBENCH",
    "d4la": "DOCLAYOUT_D4LA",
    "docsynth": "DOCLAYOUT_DOCSYNTH",
    "layoutv2": "PP_DOC_LAYOUTV2",
    "layoutv3": "PP_DOC_LAYOUTV3",
}


@dataclass
class DetectedRegion:
    """A region detected by the layout model.

    Attributes:
        category:        Mapped DocMirror zone type (e.g. "text", "data_table").
        bbox:            Bounding box as (x0, y0, x1, y1).
        confidence:      Detection confidence score in [0, 1].
        raw_category_id: Original class name reported by the model.
    """
    category: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    raw_category_id: str


class LayoutDetector:
    """RapidLayout-based page layout detector.

    The underlying ONNX model is loaded lazily on the first ``detect()``
    call.  Supports 12 built-in models selectable via *model_type*.
    """

    def __init__(self, model_type: str = "doclayout_docstructbench"):
        """
        Args:
            model_type: Model type name.  Accepts short aliases such as
                ``"doclayout"``, ``"cdla"``, ``"layoutv3"``, ``"paper"``,
                or ``"general6"``.
        """
        self._model_type_str = model_type
        self._engine = None
        self._available = HAS_RAPID_LAYOUT

    def _ensure_engine(self) -> bool:
        """Lazily load the ONNX engine.  Returns ``True`` if ready."""
        if not self._available:
            return False
        if self._engine is not None:
            return True
        try:
            alias = _MODEL_ALIASES.get(self._model_type_str.lower(), self._model_type_str.upper())
            mt = ModelType[alias]
            cfg = RapidLayoutInput(model_type=mt)
            self._engine = RapidLayout(cfg)
            logger.info(f"[LayoutDetector] RapidLayout loaded: {mt.name}")
            return True
        except Exception as e:
            logger.warning(f"[LayoutDetector] Init failed: {e}")
            self._available = False
            return False

    def detect(
        self,
        page_image,
        confidence_threshold: float = 0.5,
    ) -> List[DetectedRegion]:
        """Detect layout regions in a page image.

        Args:
            page_image: Page image as a numpy ndarray (H×W×3, RGB/BGR)
                        or a file path.
            confidence_threshold: Minimum confidence to keep a detection.

        Returns:
            List of ``DetectedRegion`` objects sorted by y-coordinate
            (top-to-bottom reading order).
        """
        if not self._ensure_engine():
            return []

        try:
            result = self._engine(page_image)
        except Exception as e:
            logger.debug(f"[LayoutDetector] Detect error: {e}")
            return []

        if result.boxes is None or len(result.boxes) == 0:
            return []

        regions: List[DetectedRegion] = []
        for box, cls_name, score in zip(result.boxes, result.class_names, result.scores):
            if score < confidence_threshold:
                continue

            category = _CATEGORY_MAP.get(cls_name.lower(), "text")

            # Skip regions classified as "abandon" (decorative / irrelevant)
            if category == "abandon":
                continue

            bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

            # Filter out detections with negligibly small area
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < 100:
                continue

            regions.append(DetectedRegion(
                category=category,
                bbox=bbox,
                confidence=float(score),
                raw_category_id=cls_name,
            ))

        # Sort by y-coordinate for natural reading order
        regions.sort(key=lambda r: r.bbox[1])

        logger.debug(f"[LayoutDetector] Detected {len(regions)} regions (threshold={confidence_threshold})")
        return regions

    @property
    def is_available(self) -> bool:
        """Whether the layout detection model is available."""
        return self._available
