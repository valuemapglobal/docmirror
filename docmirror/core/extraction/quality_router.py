"""
Adaptive Quality Router
========================

Per-zone extraction strategy router that consumes PreAnalyzer's
``strategy_params`` and ``Zone`` metadata to recommend the optimal
extraction path for each zone — without modifying any existing
function signatures.

Design principles:
    - **Stateless**: Pure function calls, no internal state.
    - **Non-breaking**: Returns current-behavior defaults when
      ``strategy_params`` is empty or absent.
    - **CPU-only**: All logic is algorithmic; no VLM or GPU required.

Usage::

    from docmirror.core.extraction.quality_router import (
        AdaptiveQualityRouter, ZoneStrategy,
    )

    router = AdaptiveQualityRouter(strategy_params)
    strategy = router.recommend(zone, page_has_text=True)

    if strategy.extract_method == "ocr_enhanced":
        # re-extract at higher DPI
        ...
"""
from __future__ import annotations


import dataclasses
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ZoneStrategy:
    """Extraction strategy recommendation for a single zone.

    Attributes:
        extract_method: One of ``"rule"``, ``"ocr_standard"``,
            ``"ocr_enhanced"``.

            - ``"rule"``: use the native text-layer extraction
              (pdfplumber / PyMuPDF) — fast, zero OCR cost.
            - ``"ocr_standard"``: standard OCR pipeline at default DPI.
            - ``"ocr_enhanced"``: high-DPI OCR for dense / low-quality zones.

        ocr_dpi: DPI to render the page region at before OCR.
            Only meaningful when ``extract_method`` is ``"ocr_*"``.
        enable_watermark_separation: If True, run deep watermark separation
            before extraction on this zone.
        skip_extraction: If True, the zone should be skipped entirely
            (e.g., pagination footers in homogeneous table documents).
        reason: Human-readable explanation of the routing decision
            (logged at DEBUG level for observability).
    """

    extract_method: str = "rule"
    ocr_dpi: int = 200
    enable_watermark_separation: bool = False
    skip_extraction: bool = False
    reason: str = ""


class AdaptiveQualityRouter:
    """Per-zone extraction strategy router.

    Instantiated once per ``_extract_page()`` call with the document-level
    ``strategy_params`` from ``PreAnalysisResult``.  For each zone,
    ``recommend()`` returns a ``ZoneStrategy`` describing how to extract it.

    When ``strategy_params`` is empty the router returns the exact same
    behavior as the current (pre-router) pipeline — guaranteed backward
    compatibility.
    """

    # Zone types that benefit from high-DPI OCR re-extraction
    _DENSE_ZONE_TYPES = frozenset({"data_table", "formula"})

    # Zone types that can be safely skipped for speed in homogeneous docs
    _SKIPPABLE_FOOTER_TYPES = frozenset({"footer"})

    def __init__(self, strategy_params: Optional[Dict[str, Any]] = None):
        self._params = strategy_params or {}

        # Pre-compute thresholds from strategy_params with safe defaults
        self._ocr_dpis = self._params.get("ocr_dpi", [200])
        self._high_dpi = max(self._ocr_dpis) if self._ocr_dpis else 200
        self._standard_dpi = self._ocr_dpis[0] if self._ocr_dpis else 200
        self._skip_watermark = self._params.get("skip_watermark_filter", False)
        self._reuse_structure = self._params.get(
            "reuse_first_page_structure", False
        )

    def recommend(
        self,
        zone,
        *,
        page_has_text: bool = True,
        page_quality: int = 100,
        is_scanned_page: bool = False,
        zone_index: int = 0,
    ) -> ZoneStrategy:
        """Recommend extraction strategy for a zone.

        Args:
            zone: A ``Zone`` object with ``.type``, ``.confidence``,
                and ``.bbox`` attributes.
            page_has_text: Whether this page has a usable text layer.
            page_quality: Image quality score (0-100) for this page.
            is_scanned_page: Whether this page was classified as scanned.
            zone_index: Index of the zone on the page (for logging).

        Returns:
            ZoneStrategy with extraction recommendations.
        """
        zone_type = getattr(zone, "type", "unknown")
        confidence = getattr(zone, "confidence", 1.0)

        # ── Fast path: high-confidence text zone on digital page ──
        if (
            page_has_text
            and not is_scanned_page
            and zone_type not in self._DENSE_ZONE_TYPES
            and confidence >= 0.8
        ):
            return ZoneStrategy(
                extract_method="rule",
                reason=f"digital page, high-confidence {zone_type} zone",
            )

        # ── Dense zone on digital page: optional high-DPI re-extraction ──
        if zone_type in self._DENSE_ZONE_TYPES:
            if confidence < 0.7 or page_quality < 70:
                return ZoneStrategy(
                    extract_method="ocr_enhanced",
                    ocr_dpi=self._high_dpi,
                    reason=(
                        f"{zone_type} zone with low confidence "
                        f"({confidence:.2f}) or quality ({page_quality})"
                    ),
                )
            # Even on digital pages, tables still go through the normal
            # layered extraction — no change from current behavior.
            return ZoneStrategy(
                extract_method="rule",
                ocr_dpi=self._standard_dpi,
                reason=f"{zone_type} zone with adequate confidence",
            )

        # ── Scanned page routing ──
        if is_scanned_page:
            # Determine DPI based on page quality
            if page_quality < 60:
                dpi = self._high_dpi
                method = "ocr_enhanced"
                reason = f"scanned, low quality ({page_quality})"
            elif page_quality < 85:
                dpi = self._standard_dpi
                method = "ocr_standard"
                reason = f"scanned, medium quality ({page_quality})"
            else:
                dpi = self._standard_dpi
                method = "ocr_standard"
                reason = f"scanned, high quality ({page_quality})"

            # Watermark separation for scanned pages when not explicitly
            # skipped and quality is in the mid-range (watermarks degrade
            # OCR most when image quality is borderline).
            enable_wm = (
                not self._skip_watermark
                and 40 <= page_quality <= 85
            )

            return ZoneStrategy(
                extract_method=method,
                ocr_dpi=dpi,
                enable_watermark_separation=enable_wm,
                reason=reason,
            )

        # ── Default: current behavior ──
        return ZoneStrategy(
            extract_method="rule",
            reason="default (no special routing needed)",
        )

    def should_enhance_table(
        self, table_data: list, extraction_confidence: float
    ) -> bool:
        """Determine if a table should be re-extracted at higher DPI.

        Called after initial table extraction to decide whether the
        result quality warrants a second pass.

        Args:
            table_data: Extracted table rows (list of lists).
            extraction_confidence: Confidence from the table engine.

        Returns:
            True if re-extraction is recommended.
        """
        if not table_data or extraction_confidence >= 0.8:
            return False

        # Check for signs of poor extraction
        total_cells = sum(len(row) for row in table_data)
        empty_cells = sum(
            1 for row in table_data
            for cell in row
            if not (cell or "").strip()
        )
        if total_cells == 0:
            return False

        empty_ratio = empty_cells / total_cells
        # High empty ratio + low confidence → likely needs re-extraction
        return empty_ratio > 0.3 and extraction_confidence < 0.6
