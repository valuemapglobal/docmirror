# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Core Extractor
============================

The core extraction engine responsible for parsing raw PDF files into
immutable ``BaseResult`` data structures.

=== Extraction Flow ===

  Step 1: Pre-processing + pre-check
          PyMuPDF fast text layer check → mark digital/scanned

  Step 2: Page iteration + layout analysis
          Call segment_page_into_zones per page to partition semantic zones

  Step 3: TableExtract
          Call extract_tables_layered for each data_table zone (4-tier progressive).
          ExtractResult collected as table Blocks into PageLayout.
          Scanned documents use OCR fallback path (analyze_scanned_page).

  Step 4: Cross-page merge (_merge_cross_page_tables)
          Detect multi-page tables and merge them into single Block.
          Continuation page table header matching strategy:
            - First row is header + matches previous page header → merge data rows directly
            - First row is not header (summary row start) → merge after _strip_preamble

  Step 5: Table post-processing (_post_process_tables)
          Call post_process_table for each table Block:
            - VOCAB_BY_CATEGORY scan to find best header row
            - Summary rows before header → extract as KV via _extract_preamble_kv
            - Summary/repeat headers after header → strip via _strip_preamble
            - Data row cleanup (_is_junk_row / _is_data_row / cell alignment)
          After extraction, call get_and_clear_preamble_kv():
            - If KV exists, create key_value Block inserted before table

  Step 6: Assemble BaseResult
          Assemble per-page Block lists into frozen BaseResult

=== Output Structure ===

  Per-page blocks list (reading_order order):
    [title]      → PageTitle
    [title]      → Account information rows
    [key_value]  → Preamble summary KV (if any. e.g., total amount/count/start date)
    [table]      → Transaction detail table (header + data rows)
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid

# S2: Module-level alias — perf_counter avoids gettimeofday syscall
_clock = time.perf_counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models.entities.domain import (
    BaseResult,
    Block,
    PageLayout,
    Style,
    TextSpan,
)
from ..exceptions import ExtractionError
from ..layout.layout_analysis import (
    Zone,
    _dedup_overlapping_chars,
    _extract_summary_entities,
    _is_header_row,
    _reconstruct_rows_from_chars,
    _strip_preamble,
    analyze_document_layout,
    analyze_scanned_page,
    extract_tables_layered,
    filter_watermark_page,
    get_last_layer_timings,
    post_process_table,
    preprocess_document,
    segment_page_into_zones,
)
from .entity_collector import collect_kv_entities
from .foundation import FitzEngine
from .html_utils import parse_html_tables_to_key_value, strip_html_to_plain_text
from .image_converter import image_to_virtual_pdf
from .table_postprocessor import process_page_tables

logger = logging.getLogger(__name__)


# HTML utility functions moved to html_utils.py
# Backward-compatible aliases for internal references
_strip_html_to_plain_text = strip_html_to_plain_text
_parse_html_tables_to_key_value = parse_html_tables_to_key_value


@dataclass
class PageExtractionContext:
    """Encapsulates all parameters for single-page extraction.

    Reduces ``_extract_page()`` from 13 positional/keyword arguments
    to a single ``ctx`` object, improving readability and enabling
    easy extension without changing call signatures.
    """

    page_plum: Any  # pdfplumber page
    fitz_page: Any  # PyMuPDF page
    fitz_doc: Any  # PyMuPDF document
    page_idx: int  # 0-based page index
    layout_al: Any  # layout analysis result
    cleaned_path: Any  # pre-processed PDF path
    is_digital: bool = True  # digital vs scanned
    strategy_params: dict[str, Any] = field(default_factory=dict)
    page_quality: int = 100  # image quality 0-100
    content_type: str = "unknown"  # table_dominant/text_dominant/mixed/scanned
    zone_template: list | None = None  # zone template for homogeneous docs
    global_grid_x: list | None = None  # global x-coordinate grid
    global_table_template: Any = None  # golden page template


def _extract_single_page_digital_worker(
    args: tuple[Any, ...],
) -> tuple[int, PageLayout, list[str], str, float]:
    """
    Worker for thread-pool page extraction (Phase 2).
    Opens path in-thread, uses rule-based layout and no formula engine, returns (page_idx, page_layout, ocr_parts, layer, conf).
    """
    (
        path,
        page_idx,
        layout_al,
        strategy_params,
        page_quality,
        document_page_count,
        content_type,
        ext_ocr_thr,
        ext_ocr_prov,
        global_grid_x,
        global_table_template,
    ) = args
    import fitz
    import pdfplumber

    path = str(Path(path).resolve())
    extractor = CoreExtractor(layout_model_path=None)
    extractor._formula_engine = None  # no model in worker path
    fitz_doc = fitz.open(path)
    plum_doc = pdfplumber.open(path)
    try:
        page_plum = plum_doc.pages[page_idx]
        fitz_page = fitz_doc[page_idx]
        ctx = PageExtractionContext(
            page_plum=page_plum,
            fitz_page=fitz_page,
            fitz_doc=fitz_doc,
            page_idx=page_idx,
            layout_al=layout_al,
            cleaned_path=Path(path),
            is_digital=True,
            strategy_params=strategy_params or {},
            page_quality=page_quality,
            content_type=content_type,
            global_grid_x=global_grid_x,
            global_table_template=global_table_template,
        )
        page_layout, ocr_parts, extraction_layer, extraction_confidence = extractor._extract_page(ctx)
        return (page_idx, page_layout, ocr_parts, extraction_layer, extraction_confidence)
    finally:
        fitz_doc.close()
        plum_doc.close()


class CoreExtractor:
    """
    Core extractor — generates immutable BaseResult from PDF.

    Usage::

        extractor = CoreExtractor()
        result = await extractor.extract(Path("sample.pdf"))
        # result is frozen BaseResult, immutable

    All low-level functions come from MultiModal.core submodules (self-contained).
    """

    def __init__(
        self,
        seal_detector_fn=None,
        layout_model_path: str | None = None,
        max_page_concurrency: int = 1,
        formula_model_path: str | None = None,
        model_render_dpi: int = 200,
    ):
        """
        Args:
            seal_detector_fn: Optional seal detection callback function.
                Signature: (fitz_doc) -> Optional[Dict[str, Any]]
                When None, skips seal detection.
            layout_model_path: Optional DocLayout-YOLO ONNX model path.
                Set to "auto" to auto-download from HuggingFace.
                When None, uses rule-based fallback.
            max_page_concurrency: Page-level concurrency.
                pdfplumber/PyMuPDF shared doc object, current default is 1 (sequential).
                Set >1 to use ThreadPoolExecutor for parallel extraction.
            formula_model_path: Optional formula recognition ONNX model path (UniMERNet).
                When None, falls back to rapid_latex_ocr -> empty string.
            model_render_dpi: Page rendering DPI for DocLayout-YOLO model inference.
                Default 200; higher values improve layout detection precision but increase inference time.
        """
        self._seal_detector_fn = seal_detector_fn
        self._layout_detector = None
        self._max_page_concurrency = max_page_concurrency
        self._model_render_dpi = model_render_dpi

        # Formula recognition engine (Strategy pattern: UniMERNet ONNX > rapid_latex_ocr > empty)
        from ..ocr.formula_engine import FormulaEngine

        self._formula_engine = FormulaEngine(model_path=formula_model_path)

        if layout_model_path:
            try:
                from ..layout.layout_model import LayoutDetector

                # layout_model_path acts as model type name
                # "auto" -> default doclayout_docstructbench
                model_type = "doclayout_docstructbench" if layout_model_path == "auto" else layout_model_path
                self._layout_detector = LayoutDetector(model_type=model_type)
                logger.info("[DocMirror] Layout model enabled (RapidLayout)")
            except Exception as e:
                logger.warning(f"[DocMirror] Layout model init failed, falling back to rules: {e}")

    # Supported image formats
    _IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"})

    @staticmethod
    def _image_to_virtual_pdf(image_path: Path) -> fitz.Document:
        """Convert image to a virtual single-page PDF, pre-scaling large images to max 4096px."""
        return image_to_virtual_pdf(image_path)

    async def extract(self, file_path: Path) -> BaseResult:
        """
        Main entry point: extract BaseResult from PDF or image.

        Supported formats:
            - PDF File (.pdf)
            - ImageFile (.jpg, .png, .tiff, .bmp, .webp)

        Image files are automatically converted to virtual PDFs and run through the
        full parsing pipeline (layout analysis -> table extraction -> formula recognition -> reading order).

        Args:
            file_path: Path to PDF or image file.

        Returns:
            BaseResult: Immutable extraction result.
        """
        t0 = _clock()
        file_path = Path(file_path)
        doc_id = str(uuid.uuid4())
        is_image_input = file_path.suffix.lower() in self._IMAGE_SUFFIXES

        logger.info(f"[DocMirror] ▶ extract | file={file_path.name} | image={is_image_input}")

        # === [Main parsing logic (Heuristics)] ===
        fitz_doc = None
        try:
            import asyncio

            # === Step 0: Image -> virtual PDF ===
            if is_image_input:
                fitz_doc = await asyncio.to_thread(self._image_to_virtual_pdf, file_path)
                has_text = False  # images have no text layer
                logger.info("[DocMirror] Image input -> virtual PDF, marked as scanned document")
            else:
                # === Step 1: Pre-processing + pre-check ===
                cleaned_path = await asyncio.to_thread(preprocess_document, file_path)
                fitz_doc = await asyncio.to_thread(FitzEngine.open, cleaned_path)
                has_text = await asyncio.to_thread(FitzEngine.has_text_layer, fitz_doc)

            if not has_text:
                logger.info("[DocMirror] Text layer missing, marked as scanned document")

            # === Step 1.5: Pre-analysis (human cognition stage 2) ===
            from .pre_analyzer import PreAnalyzer

            pre_analysis = await asyncio.to_thread(PreAnalyzer().analyze, fitz_doc)

            # Define the heavy CPU-bound parsing block to run in a thread
            def _process_pdf_sync(fitz_doc, pre_analysis, has_text):
                return self._process_pdf_sync(
                    fitz_doc=fitz_doc,
                    pre_analysis=pre_analysis,
                    has_text=has_text,
                    is_image_input=is_image_input,
                    cleaned_path=cleaned_path if not is_image_input else None,
                    file_path=file_path,
                )

            # Execute the heavy synchronous block in a thread
            pages, full_text, extraction_layer, extraction_confidence, _perf, _page_perf = await asyncio.to_thread(
                _process_pdf_sync, fitz_doc, pre_analysis, has_text
            )

            # === Step 6: Assemble BaseResult ===
            elapsed = (_clock() - t0) * 1000

            total_blocks = sum(len(p.blocks) for p in pages)
            table_count = sum(1 for p in pages for b in p.blocks if b.block_type == "table")

            # Extract entities (regex + KV blocks)
            extracted_entities = self._collect_kv_entities(pages)

            # -- Optional: Seal detection (first page) -- via dependency injection --
            seal_info = None
            if self._seal_detector_fn:
                try:
                    seal_info = self._seal_detector_fn(fitz_doc)
                except Exception as e:
                    logger.debug(f"[DocMirror] Seal detection skip: {e}")
            if seal_info:
                pass  # merged into metadata below

            metadata = {
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size if file_path.exists() else 0,
                "page_count": len(pages),
                "parser": "DocMirror_CoreExtractor",
                "elapsed_ms": round(elapsed, 1),
                "block_count": total_blocks,
                "table_count": table_count,
                "has_text_layer": has_text,
                "scanned_pages": [p.page_number for p in pages if p.is_scanned],
                "pre_analysis": pre_analysis.to_dict(),
                "extracted_entities": extracted_entities,
                "perf_breakdown": _perf,
                "perf_per_page": _page_perf,
            }
            if seal_info:
                metadata["seal_info"] = seal_info

            # -- Extraction quality assessment metadata --
            if table_count > 0:
                # Find main table (largest table block)
                main_table = None
                for p in pages:
                    for b in p.blocks:
                        if b.block_type == "table" and isinstance(b.raw_content, list):
                            if main_table is None or len(b.raw_content) > len(main_table):
                                main_table = b.raw_content

                header_detected = False
                data_row_count = 0
                col_count_stable = True
                empty_cell_ratio = 0.0

                if main_table and len(main_table) >= 2:
                    header_detected = _is_header_row(main_table[0])
                    data_row_count = len(main_table) - 1
                    expected_cols = len(main_table[0])
                    col_count_stable = all(len(row) == expected_cols for row in main_table)
                    total_cells = sum(len(row) for row in main_table)
                    empty_cells = sum(1 for row in main_table for c in row if not (c or "").strip())
                    empty_cell_ratio = round(empty_cells / max(1, total_cells), 3)

                metadata["extraction_quality"] = {
                    "extraction_layer": extraction_layer,
                    "extraction_confidence": extraction_confidence,
                    "header_detected": header_detected,
                    "data_row_count": data_row_count,
                    "col_count_stable": col_count_stable,
                    "empty_cell_ratio": empty_cell_ratio,
                    "layer_timings_ms": get_last_layer_timings(),
                }

            result = BaseResult(
                document_id=doc_id,
                pages=tuple(pages),
                metadata=metadata,
                full_text=full_text,
            )

            logger.info(
                f"[DocMirror] ◀ extract | pages={len(pages)} | blocks={total_blocks} | "
                f"tables={table_count} | elapsed={elapsed:.0f}ms"
            )

            return result

        except ExtractionError as e:
            logger.error(f"[DocMirror] extraction error: {e}", exc_info=True)
            return BaseResult(
                document_id=doc_id,
                metadata={"error": str(e), "error_type": "ExtractionError", "parser": "DocMirror_CoreExtractor"},
                full_text="",
            )
        except Exception as e:
            logger.error(f"[DocMirror] unexpected error: {e}", exc_info=True)
            return BaseResult(
                document_id=doc_id,
                metadata={"error": str(e), "error_type": type(e).__name__, "parser": "DocMirror_CoreExtractor"},
                full_text="",
            )
        finally:
            try:
                if fitz_doc:
                    fitz_doc.close()
            except Exception as exc:
                logger.debug(f"operation: suppressed {exc}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Scanned Document OCR Extraction
    # ═══════════════════════════════════════════════════════════════════════════

    def _process_pdf_sync(
        self,
        fitz_doc,
        pre_analysis,
        has_text: bool,
        is_image_input: bool,
        cleaned_path,
        file_path,
    ):
        """CPU-bound core: layout analysis → per-page extraction → post-processing.

        Runs in a thread via ``asyncio.to_thread()`` to avoid blocking the
        event loop.  Extracted from the ``extract()`` method to de-nest the
        closure and enable independent testing.

        Returns:
            ``(pages, full_text, extraction_layer, extraction_confidence,
            _perf, _page_perf)`` 6-tuple.
        """
        # ── Per-step timing instrumentation ──
        _perf: dict[str, float] = {}
        _page_perf: list = []  # per-page timing breakdown

        # Derived: plumber_path for pdfplumber
        plumber_path = cleaned_path if cleaned_path else file_path

        # === Step 2: Layout analysis (parallel when max_page_concurrency > 1 and path available) ===
        _t = _clock()
        num_pages = len(fitz_doc)
        use_parallel_layout = (
            self._max_page_concurrency > 1 and not is_image_input and cleaned_path is not None and num_pages >= 4
        )
        if use_parallel_layout:
            from ..layout.layout_analysis import analyze_document_layout_parallel

            layout_path = str(Path(cleaned_path).resolve())
            env_layout = int(os.environ.get("DOCMIRROR_LAYOUT_MAX_WORKERS", "0"))
            layout_workers = (
                min(num_pages, env_layout)
                if env_layout > 0
                else min(self._max_page_concurrency, num_pages, os.cpu_count() or 4)
            )
            if layout_workers <= 1:
                page_layouts_al = analyze_document_layout(fitz_doc)
            else:
                page_layouts_al = analyze_document_layout_parallel(layout_path, num_pages, max_workers=layout_workers)
        else:
            page_layouts_al = analyze_document_layout(fitz_doc)
        _perf["layout_analysis_ms"] = (_clock() - _t) * 1000

        # Extract full text (with per-page caching to avoid redundant calls)
        _page_text_cache: dict[int, str] = {}
        full_text_parts = []
        for p_idx, page in enumerate(fitz_doc):
            txt = page.get_text()
            _page_text_cache[p_idx] = txt
            full_text_parts.append(txt)
        full_text_raw = "\n\n".join(full_text_parts)
        # Delay NFKC: only normalize at assembly time, not eagerly
        full_text = full_text_raw

        # === Step 3: Per-page extraction -- Zone -> Block ===
        # P0: Per-page hybrid routing — each page individually assessed
        # P1: Smart early-exit — honor DOCMIRROR_MAX_PAGES at core layer
        pages: list[PageLayout] = []
        ocr_text_parts: list[str] = []
        extraction_layer: str = "unknown"  # last strategy layer used by extract_tables_layered
        extraction_confidence: float = 0.0  # last extraction confidence

        max_pages = int(os.environ.get("DOCMIRROR_MAX_PAGES", "0"))

        # Per-page text presence detection (>50 chars = has text layer)
        # Use cached text from above
        _TEXT_THRESHOLD = 50
        page_has_text = []
        for p_idx in range(len(fitz_doc)):
            txt = _page_text_cache.get(p_idx, "")
            page_has_text.append(len(txt.strip()) > _TEXT_THRESHOLD)

        hybrid_doc = has_text and not all(page_has_text)
        if hybrid_doc:
            scanned_indices = [i for i, v in enumerate(page_has_text) if not v]
            logger.info(
                f"[DocMirror] Hybrid document detected: "
                f"{len(scanned_indices)} scanned pages out of {len(page_has_text)}"
            )

        # Open pdfplumber once for all digital pages
        import pdfplumber

        plumber_doc = None
        if has_text or hybrid_doc:
            plumber_doc = pdfplumber.open(str(plumber_path))

        # External OCR: resolve once for all scanned pages (quality < threshold → delegate)
        from docmirror.configs.settings import default_settings
        from docmirror.core.ocr.fallback import _resolve_external_ocr_provider

        _ext_ocr_threshold = getattr(default_settings, "external_ocr_quality_threshold", None)
        _ext_ocr_provider = _resolve_external_ocr_provider(getattr(default_settings, "external_ocr_provider", None))

        # -- Phase 1.5: Build Global Grid Tensor --
        global_grid_x = None
        if plumber_doc:
            try:
                from ..table.extraction.grid_tensor import build_global_grid_tensor

                _gt0 = _clock()
                # Extract chars from all digital pages (limit to first 50)
                max_tensor_pages = min(len(plumber_doc.pages), 50)
                all_chars = []
                max_w = 0
                for pid in range(max_tensor_pages):
                    if page_has_text[pid]:
                        p = plumber_doc.pages[pid]
                        all_chars.append(p.chars)
                        if getattr(p, "width", 0) > max_w:
                            max_w = p.width
                if all_chars and max_w > 0:
                    global_grid_x = build_global_grid_tensor(all_chars, max_w, resolution=1.0)
                logger.debug(f"[DocMirror] Global grid tensor built in {(_clock() - _gt0) * 1000:.1f}ms")
            except Exception as e:
                logger.warning(f"[DocMirror] Failed to build global grid tensor: {e}")

        # -- Phase 1.8: Golden Page Template Sampling (Graph-Propagated Injection) --
        global_table_template = None
        if plumber_doc and len(fitz_doc) >= 3:
            try:
                # extract_tables_layered already imported at module level (via layout_analysis re-export)
                from ..table.extraction.template_injector import build_global_template

                # Sample the very middle page
                sample_idx = len(fitz_doc) // 2
                if page_has_text[sample_idx]:
                    sp_t0 = _clock()
                    sample_plum = plumber_doc.pages[sample_idx]
                    sample_fitz = fitz_doc[sample_idx]

                    # Do a quick full layout segmentation on the sample page to get the table zone
                    # segment_page_into_zones already imported at module level
                    sample_zones = segment_page_into_zones(sample_plum, sample_idx)

                    table_zone = None
                    for z in sample_zones:
                        if z.type == "data_table":
                            table_zone = z
                            break

                    if table_zone:
                        stabs, slayer, sconf = extract_tables_layered(
                            sample_plum,
                            table_zone_bbox=table_zone.bbox,
                            document_page_count=len(fitz_doc),
                            fitz_page=sample_fitz,
                        )
                        if sconf >= 0.85 and stabs and len(stabs[0]) >= 2:
                            logger.info(
                                f"[DocMirror] Golden Page Sampling: page {sample_idx} yielded confidence {sconf:.2f}. Building GlobalTableTemplate."
                            )
                            crop_x0, crop_x1 = 0, sample_plum.width
                            y0, y1 = table_zone.bbox[1], table_zone.bbox[3]
                            work_page = sample_plum.crop((crop_x0, y0, crop_x1, y1))
                            global_table_template = build_global_template(work_page, stabs[0])

                    logger.debug(f"[DocMirror] Golden Page Sampling finished in {(_clock() - sp_t0) * 1000:.1f}ms")
            except Exception as e:
                logger.warning(f"[DocMirror] Golden Page Template Sampling failed: {e}")

        num_digital = sum(1 for i in range(len(page_has_text)) if page_has_text[i])
        use_page_concurrency = self._max_page_concurrency > 1 and plumber_doc is not None and num_digital >= 2
        max_workers = min(self._max_page_concurrency, 4, len(fitz_doc)) if use_page_concurrency else 1

        try:
            if use_page_concurrency:
                # Phase 2: parallel digital page extraction (thread pool)
                results_by_idx = {}  # page_idx -> future or (page_layout, ocr_parts)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for page_idx, layout_al in enumerate(page_layouts_al):
                        if max_pages > 0 and page_idx >= max_pages:
                            break
                        _pt = _clock()
                        try:
                            if page_has_text[page_idx] and plumber_doc:
                                future = executor.submit(
                                    _extract_single_page_digital_worker,
                                    (
                                        str(Path(plumber_path).resolve()),
                                        page_idx,
                                        layout_al,
                                        pre_analysis.strategy_params,
                                        dict(pre_analysis.page_quality_map).get(
                                            page_idx, pre_analysis.avg_image_quality
                                        ),
                                        len(fitz_doc),
                                        pre_analysis.content_type,
                                        _ext_ocr_threshold,
                                        _ext_ocr_provider,
                                        global_grid_x,
                                        global_table_template,
                                    ),
                                )
                                results_by_idx[page_idx] = future
                            else:
                                fitz_page = fitz_doc[page_idx]
                                page_qual = dict(pre_analysis.page_quality_map).get(
                                    page_idx, pre_analysis.avg_image_quality
                                )
                                page_layout = self._extract_scanned_page(
                                    fitz_page=fitz_page,
                                    page_idx=page_idx,
                                    page_quality=page_qual,
                                    external_ocr_threshold=_ext_ocr_threshold,
                                    external_ocr_provider=_ext_ocr_provider,
                                    global_grid_x=global_grid_x,
                                )
                                ocr_parts = []
                                for blk in page_layout.blocks:
                                    if blk.block_type == "text" and blk.raw_content:
                                        ocr_parts.append(str(blk.raw_content))
                                results_by_idx[page_idx] = (page_layout, ocr_parts)
                        except Exception as page_exc:
                            logger.error(
                                f"[DocMirror] ❌ Page {page_idx} extraction FAILED: {page_exc}",
                                exc_info=True,
                            )
                        _page_perf.append({"page": page_idx, "total": (_clock() - _pt) * 1000})
                # Collect results in page order
                last_layer, last_conf = "unknown", 0.0
                for page_idx in sorted(results_by_idx.keys()):
                    r = results_by_idx[page_idx]
                    if hasattr(r, "result"):
                        _, page_layout, ocr_parts, last_layer, last_conf = r.result()
                        pages.append(page_layout)
                        ocr_text_parts.extend(ocr_parts)
                    else:
                        page_layout, ocr_parts = r
                        pages.append(page_layout)
                        ocr_text_parts.extend(ocr_parts)
                extraction_layer = last_layer
                extraction_confidence = last_conf
            else:
                # Sequential path (original)
                _zone_template = None  # Perf #9: populated from page 0 via self
                self._zone_template = None  # Reset for each document
                # Perf #11: PageState for cross-page layer hint + header forwarding
                from ..table.page_state import PageState

                _page_state = PageState()
                self._page_state = _page_state
                for page_idx, layout_al in enumerate(page_layouts_al):
                    if max_pages > 0 and page_idx >= max_pages:
                        logger.info(f"[DocMirror] Early exit at page {page_idx} (DOCMIRROR_MAX_PAGES={max_pages})")
                        break
                    fitz_page = fitz_doc[page_idx]
                    _pt = _clock()
                    try:
                        if page_has_text[page_idx] and plumber_doc:
                            page_plum = plumber_doc.pages[page_idx]
                            # Perf #9: pass template for pages after page 0
                            _pass_template = (
                                self._zone_template if pre_analysis.layout_homogeneous and page_idx > 0 else None
                            )
                            ctx = PageExtractionContext(
                                page_plum=page_plum,
                                fitz_page=fitz_page,
                                fitz_doc=fitz_doc,
                                page_idx=page_idx,
                                layout_al=layout_al,
                                cleaned_path=plumber_path,
                                is_digital=True,
                                strategy_params=pre_analysis.strategy_params,
                                page_quality=dict(pre_analysis.page_quality_map).get(
                                    page_idx, pre_analysis.avg_image_quality
                                ),
                                content_type=pre_analysis.content_type,
                                zone_template=_pass_template,
                                global_grid_x=global_grid_x,
                                global_table_template=global_table_template,
                            )
                            page_layout, page_ocr_parts, extraction_layer, extraction_confidence = self._extract_page(
                                ctx
                            )
                            pages.append(page_layout)
                            ocr_text_parts.extend(page_ocr_parts)
                        else:
                            # Scanned page → OCR extraction (optional external handoff if quality too low)
                            page_qual = dict(pre_analysis.page_quality_map).get(
                                page_idx, pre_analysis.avg_image_quality
                            )
                            page_layout = self._extract_scanned_page(
                                fitz_page=fitz_page,
                                page_idx=page_idx,
                                page_quality=page_qual,
                                external_ocr_threshold=_ext_ocr_threshold,
                                external_ocr_provider=_ext_ocr_provider,
                                global_grid_x=global_grid_x,
                            )
                            pages.append(page_layout)
                            for blk in page_layout.blocks:
                                if blk.block_type == "text" and blk.raw_content:
                                    ocr_text_parts.append(str(blk.raw_content))
                    except Exception as page_exc:
                        logger.error(
                            f"[DocMirror] ❌ Page {page_idx} extraction FAILED: {page_exc}",
                            exc_info=True,
                        )
                    _page_perf.append({"page": page_idx, "total": (_clock() - _pt) * 1000})
        finally:
            if plumber_doc:
                plumber_doc.close()

            # Merge OCR text into full text
            if ocr_text_parts:
                full_text = full_text + "\n\n" + "\n\n".join(ocr_text_parts)

            # ═══ Step 4: Table post-processing (header detection + cleanup) ═══
            # Run BEFORE merge so each page's table has a confirmed header row
            _t = _clock()
            pages = self._post_process_tables(pages)
            _perf["table_postprocess_ms"] = (_clock() - _t) * 1000

            # ═══ Step 5: Cross-page merge ═══
            _t = _clock()
            pages = self._merge_cross_page_tables(pages)
            _perf["cross_page_merge_ms"] = (_clock() - _t) * 1000

            # ═══ Step 5.5: Table structure fix (post-merge) ═══
            pages = self._fix_table_structures(pages)

            # ═══ Step 5.6: Post-merge header inference ═══
            pages = self._infer_missing_headers(pages)

            # ── Log performance breakdown ──
            logger.info(
                "[DocMirror] ⏱ Pipeline timing breakdown:\n"
                + "\n".join(f"    {k}: {v:.0f}ms" for k, v in _perf.items())
            )
            if _page_perf:
                for pp in _page_perf:
                    logger.debug(
                        f"[DocMirror] ⏱ Page {pp['page']}: "
                        + " | ".join(f"{k}={v:.0f}ms" for k, v in pp.items() if k != "page")
                    )

            return pages, full_text, extraction_layer, extraction_confidence, _perf, _page_perf

    def _fix_table_structures(self, pages: list) -> list:
        """Step 5.5: Apply table structure fix to all table blocks.

        Run AFTER cross-page merge so column removal doesn't cause col-count
        mismatches that prevent merging.
        """
        from docmirror.core.table.table_structure_fix import fix_table_structure

        fixed_pages = []
        for pg in pages:
            new_blocks = []
            for block in pg.blocks:
                if block.block_type == "table" and isinstance(block.raw_content, list) and len(block.raw_content) >= 2:
                    fixed = fix_table_structure(block.raw_content)
                    new_blocks.append(
                        Block(
                            block_id=block.block_id,
                            block_type=block.block_type,
                            bbox=block.bbox,
                            reading_order=block.reading_order,
                            page=block.page,
                            raw_content=fixed,
                        )
                    )
                else:
                    new_blocks.append(block)
            fixed_pages.append(
                PageLayout(
                    page_number=pg.page_number,
                    width=pg.width,
                    height=pg.height,
                    blocks=tuple(new_blocks),
                    semantic_zones=pg.semantic_zones,
                    is_scanned=pg.is_scanned,
                )
            )
        return fixed_pages

    def _infer_missing_headers(self, pages: list) -> list:
        """Step 5.6: Infer and prepend missing table headers from text blocks.

        When a merged table starts with a data row (no header), scans
        text blocks from earlier pages for a vocabulary-matching header
        line and prepends it.
        """
        from ..utils.vocabulary import _is_data_row, _score_header_by_vocabulary

        for pg in pages:
            for block in pg.blocks:
                if block.block_type != "table" or not isinstance(block.raw_content, list):
                    continue
                rc = block.raw_content
                if not rc or len(rc) < 2:
                    continue
                if _is_header_row(rc[0]):
                    continue

                logger.warning(
                    f"[Extractor] Page {pg.page_number}: Table block missing header, searching for header in text..."
                )
                candidate_header = None
                best_vocab = 0
                for prev_pg in pages:
                    if prev_pg.page_number > pg.page_number:
                        break
                    for tb in prev_pg.blocks:
                        if tb.block_type == "text" and isinstance(tb.raw_content, str):
                            words = [w.strip() for w in tb.raw_content.split() if w.strip()]
                            if len(words) >= 3:
                                vs = _score_header_by_vocabulary(words)
                                if vs > best_vocab and vs >= 3:
                                    best_vocab = vs
                                    candidate_header = words
                if candidate_header:
                    logger.info(
                        f"[Merger] Page {pg.page_number}: Prepending inferred header (vocabulary score: {best_vocab})"
                    )
                    ncols = len(rc[0])
                    if len(candidate_header) == ncols:
                        aligned = candidate_header
                    elif len(candidate_header) > ncols:
                        aligned = candidate_header[:ncols]
                    else:
                        aligned = candidate_header + [""] * (ncols - len(candidate_header))
                    rc_new = [aligned] + list(rc)
                    new_blocks = []
                    for b in pg.blocks:
                        if b is block:
                            new_blocks.append(
                                Block(
                                    block_id=b.block_id,
                                    block_type=b.block_type,
                                    bbox=b.bbox,
                                    reading_order=b.reading_order,
                                    page=b.page,
                                    raw_content=rc_new,
                                )
                            )
                        else:
                            new_blocks.append(b)
                    idx = pages.index(pg)
                    pages[idx] = PageLayout(
                        page_number=pg.page_number,
                        width=pg.width,
                        height=pg.height,
                        blocks=tuple(new_blocks),
                        semantic_zones=pg.semantic_zones,
                        is_scanned=pg.is_scanned,
                    )
                    logger.info(
                        f"[DocMirror] header inferred from text: vocab={best_vocab}, words={len(candidate_header)}"
                    )
                    break
        return pages

    def _extract_scanned_page(
        self,
        *,
        fitz_page,
        page_idx: int,
        page_quality: int | None = None,
        external_ocr_threshold: int | None = None,
        external_ocr_provider: Any | None = None,
        global_grid_x: list[float] | None = None,
    ) -> PageLayout:
        """Single-page OCR extraction for scanned documents.

        Uses ``ocr_extract_universal`` which auto-detects content type:
        - Table documents → existing table block pipeline
        - General documents → text blocks in reading order with real bboxes

        When ``page_quality`` is below ``external_ocr_threshold`` and
        ``external_ocr_provider`` is set, OCR is delegated to the external
        provider (e.g. cloud OCR for 99% recognition on poor-quality scans).
        """
        from ..ocr.fallback import ocr_extract_universal

        width = fitz_page.rect.width
        height = fitz_page.rect.height
        blocks: list[Block] = []
        reading_order = 0

        try:
            ocr_result = ocr_extract_universal(
                fitz_page,
                page_idx,
                page_quality=page_quality,
                external_ocr_threshold=external_ocr_threshold,
                external_ocr_provider=external_ocr_provider,
            )

            if ocr_result and isinstance(ocr_result, dict):
                content_type = ocr_result.get("content_type", "table")

                if content_type == "table":
                    # ── Table document: existing logic ──
                    header_text = ocr_result.get("header_text", "").strip()
                    if header_text:
                        blocks.append(
                            Block(
                                block_id=f"blk_{page_idx}_{reading_order}",
                                block_type="text",
                                bbox=(0, 0, width, height * 0.1),
                                reading_order=reading_order,
                                page=page_idx + 1,
                                raw_content=header_text,
                            )
                        )
                        reading_order += 1

                    table_data = ocr_result.get("table", [])
                    if table_data and len(table_data) >= 2:
                        blocks.append(
                            Block(
                                block_id=f"blk_{page_idx}_{reading_order}",
                                block_type="table",
                                bbox=(0, height * 0.1, width, height * 0.9),
                                reading_order=reading_order,
                                page=page_idx + 1,
                                raw_content=table_data,
                            )
                        )
                        reading_order += 1
                    elif table_data:
                        text_lines = [" | ".join(str(c) for c in row if c) for row in table_data if any(c for c in row)]
                        if text_lines:
                            blocks.append(
                                Block(
                                    block_id=f"blk_{page_idx}_{reading_order}",
                                    block_type="text",
                                    bbox=(0, height * 0.1, width, height * 0.9),
                                    reading_order=reading_order,
                                    page=page_idx + 1,
                                    raw_content="\n".join(text_lines),
                                )
                            )
                            reading_order += 1

                    footer_text = ocr_result.get("footer_text", "").strip()
                    if footer_text:
                        blocks.append(
                            Block(
                                block_id=f"blk_{page_idx}_{reading_order}",
                                block_type="text",
                                bbox=(0, height * 0.9, width, height),
                                reading_order=reading_order,
                                page=page_idx + 1,
                                raw_content=footer_text,
                            )
                        )
                        reading_order += 1

                    logger.info(
                        f"[DocMirror] OCR page {page_idx} (table): "
                        f"header={bool(header_text)} "
                        f"table={len(table_data)}rows "
                        f"footer={bool(footer_text)}"
                    )

                else:
                    # ── General document: text blocks per line ──
                    lines = ocr_result.get("lines", [])
                    ocr_page_h = ocr_result.get("page_h", 1)
                    ocr_page_w = ocr_result.get("page_w", 1)

                    # Scale OCR pixel coords → PDF point coords
                    sx = width / max(ocr_page_w, 1)
                    sy = height / max(ocr_page_h, 1)

                    full_text_parts: list[str] = []
                    for line in lines:
                        text = line.get("text", "").strip()
                        if not text:
                            continue
                        full_text_parts.append(text)
                        ox0, oy0, ox1, oy1 = line.get("bbox", (0, 0, 0, 0))
                        bbox = (ox0 * sx, oy0 * sy, ox1 * sx, oy1 * sy)
                        # If content is HTML (e.g. external OCR), store plain text in block.
                        # Drop table segments so tables appear only in key_value block, not duplicated as text.
                        if "<table" in text.lower() or "<td" in text.lower():
                            text = _strip_html_to_plain_text(text, drop_tables=True)
                        blocks.append(
                            Block(
                                block_id=f"blk_{page_idx}_{reading_order}",
                                block_type="text",
                                bbox=bbox,
                                reading_order=reading_order,
                                page=page_idx + 1,
                                raw_content=text,
                            )
                        )
                        reading_order += 1

                    # When external OCR returns HTML <table>, parse to key_value block
                    full_text = "\n\n".join(full_text_parts)
                    kv = _parse_html_tables_to_key_value(full_text)
                    if kv:
                        blocks.append(
                            Block(
                                block_id=f"blk_{page_idx}_{reading_order}",
                                block_type="key_value",
                                bbox=(0, 0, width, height),
                                reading_order=reading_order,
                                page=page_idx + 1,
                                raw_content=kv,
                            )
                        )
                        reading_order += 1
                        logger.debug(
                            f"[DocMirror] OCR page {page_idx} (general): parsed table → key_value with {len(kv)} pairs"
                        )

                    logger.info(f"[DocMirror] OCR page {page_idx} (general): {len(lines)} text lines extracted")

        except Exception as e:
            logger.warning(f"[DocMirror] OCR failed on page {page_idx}: {e}")

        return PageLayout(
            page_number=page_idx + 1,
            blocks=tuple(blocks),
            is_scanned=True,
        )

    @staticmethod
    def _group_words_into_lines(words: list[dict], tolerance_ratio: float = 0.5) -> list[list[dict]]:
        """Group OCR words into lines by Y coordinate.

        Args:
            words: List of OCR word dicts, each with bbox and text fields.
            tolerance_ratio: Line spacing tolerance (relative to average character height).

        Returns:
            Word lists grouped by line.
        """
        if not words:
            return []

        # Calculate centre y of each word
        items = []
        for w in words:
            bbox = w.get("bbox", (0, 0, 0, 0))
            if len(bbox) >= 4:
                cy = (bbox[1] + bbox[3]) / 2
                h = bbox[3] - bbox[1]
                items.append((cy, h, w))

        if not items:
            return []

        # Sort by y
        items.sort(key=lambda x: x[0])

        # Estimate average character height
        avg_h = sum(h for _, h, _ in items) / len(items) if items else 10
        tolerance = avg_h * tolerance_ratio

        # Group into lines
        lines: list[list[dict]] = []
        current_line: list[dict] = [items[0][2]]
        current_y = items[0][0]

        for cy, h, w in items[1:]:
            if abs(cy - current_y) <= tolerance:
                current_line.append(w)
            else:
                # Sort line internally by x
                current_line.sort(key=lambda word: word.get("bbox", (0,))[0])
                lines.append(current_line)
                current_line = [w]
                current_y = cy

        if current_line:
            current_line.sort(key=lambda word: word.get("bbox", (0,))[0])
            lines.append(current_line)

        return lines

    # ═══════════════════════════════════════════════════════════════════════════
    # Zone → Block Handlers (extracted from _extract_page for readability)
    # ═══════════════════════════════════════════════════════════════════════════

    def _handle_formula_zone(
        self,
        zone,
        block_id: str,
        page_idx: int,
        width: float,
        height: float,
        content_type: str,
        reading_order: int,
    ) -> tuple[Block | None, float]:
        """Handle a formula-type zone → Block conversion.

        Returns:
            (block_or_none, formula_ms): The formula block (or None if no formula
            was detected) and the time spent in ms.
        """
        _fml_t = _clock()
        # Perf #10: Skip formula detection for text/table dominant docs
        _skip_formula = content_type in ("text_dominant", "table_dominant") or not self._formula_engine
        if _skip_formula:
            if zone.text:
                block = Block(
                    block_id=block_id,
                    block_type="text",
                    bbox=zone.bbox,
                    reading_order=reading_order,
                    page=page_idx + 1,
                    raw_content=zone.text,
                )
                return block, (_clock() - _fml_t) * 1000
            return None, (_clock() - _fml_t) * 1000

        # ── 3-tier formula zone gating (skip false-positive OCR) ──
        _skip_formula_ocr = False

        # Gate 1: YOLO confidence threshold
        if zone.confidence < 0.65:
            _skip_formula_ocr = True
            logger.debug(f"formula gate: skipped zone (confidence={zone.confidence:.2f} < 0.65)")

        # Gate 2: Zone area filter (formulas are small)
        if not _skip_formula_ocr:
            zone_area = (zone.bbox[2] - zone.bbox[0]) * (zone.bbox[3] - zone.bbox[1])
            page_area = width * height
            if page_area > 0 and zone_area > page_area * 0.3:
                _skip_formula_ocr = True
                logger.debug(f"formula gate: skipped zone (area={zone_area:.0f} > 30% page)")

        # Gate 3: Character content pre-check (must have math indicators)
        if not _skip_formula_ocr:
            _MATH_INDICATORS = set("∑∫∂√±≤≥≠∞∈∉∝∀∃αβγδεθλμπσφψω")
            zone_text = zone.text or ""
            has_math_chars = bool(set(zone_text) & _MATH_INDICATORS)
            has_operator_pattern = bool(
                re.search(
                    r"[≤≥±×÷∑∫]"  # Unambiguous math symbols
                    r"|[a-zA-Z]\^[\d{]"  # Superscript notation: x^2, x^{n}
                    r"|[a-zA-Z]_[\d{]"  # Subscript notation: a_1, a_{ij}
                    r"|\\\\[a-z]{3,}"  # LaTeX commands: \\frac, \\sqrt
                    r"|\{[^}]+\}",  # Braced expressions: {n+1}
                    zone_text,
                )
            )

            # True superscript detection: small chars that are ELEVATED
            has_superscript = False
            if zone.chars and len(zone.chars) >= 3:
                sizes = [c.get("size", 12) for c in zone.chars if c.get("size", 0) > 0]
                if sizes:
                    median_size = sorted(sizes)[len(sizes) // 2]
                    if median_size > 5:
                        tops = [c["top"] for c in zone.chars]
                        median_top = sorted(tops)[len(tops) // 2]
                        has_superscript = any(
                            c.get("size", 12) < median_size * 0.6 and c["top"] < median_top - 2 for c in zone.chars
                        )

            if not (has_math_chars or has_operator_pattern or has_superscript):
                _skip_formula_ocr = True
                logger.debug(f"formula gate: skipped zone (no math indicators in '{zone_text[:30]}')")

        # K1: prefer extracting from character stream (zero latency)
        latex_str = None
        try:
            from ..ocr.formula_chars import extract_formula_from_chars

            if zone.chars:
                latex_str = extract_formula_from_chars(zone.chars, zone.bbox)
        except Exception as exc:
            logger.debug(f"operation: suppressed {exc}")

        # K1 fallback: OCR cropped image recognition (ONLY if gates passed)
        if not latex_str and not _skip_formula_ocr:
            formula_img = self._crop_zone_image(fitz_page, zone.bbox)
            latex_str = self._recognize_formula(formula_img)
        _formula_ms = (_clock() - _fml_t) * 1000

        if latex_str:
            block = Block(
                block_id=block_id,
                block_type="formula",
                bbox=zone.bbox,
                reading_order=reading_order,
                page=page_idx + 1,
                raw_content=latex_str,
            )
            return block, _formula_ms
        elif _skip_formula_ocr and zone.text:
            block = Block(
                block_id=block_id,
                block_type="text",
                bbox=zone.bbox,
                reading_order=reading_order,
                page=page_idx + 1,
                raw_content=zone.text,
            )
            return block, _formula_ms
        return None, _formula_ms

    def _handle_data_table_zone(
        self,
        zone,
        block_id: str,
        page_idx: int,
        page_plum,
        fitz_page,
        fitz_doc,
        reading_order: int,
        is_digital: bool,
        _watermark_filtered: bool,
        _router,
        global_table_template,
    ) -> tuple[list[Block], str, float, float, bool]:
        """Handle a data_table-type zone → Block conversion + PID retry loop.

        Returns:
            (blocks, extraction_layer, extraction_confidence, table_ms, tables_extracted)
        """
        _tbl_t = _clock()
        result_blocks: list[Block] = []

        # P3-2: Detect merged cells
        merged_cells = []
        try:
            from ..table.extraction.engine import detect_merged_cells

            merged_cells = detect_merged_cells(page_plum, table_zone_bbox=zone.bbox)
        except Exception as exc:
            logger.debug(f"operation: suppressed {exc}")

        page_tables, extraction_layer, extraction_confidence = extract_tables_layered(
            page_plum,
            table_zone_bbox=zone.bbox,
            document_page_count=len(fitz_doc),
            fitz_page=fitz_page,
            watermark_filtered=_watermark_filtered,
            layer_hint=(
                self._page_state.winning_layer
                if hasattr(self, "_page_state") and self._page_state.should_use_hint()
                else None
            ),
            table_template=global_table_template,
        )
        _table_ms = (_clock() - _tbl_t) * 1000
        zone_tables_extracted = False
        ro = reading_order

        for tbl in page_tables:
            if tbl and len(tbl) >= 1:
                zone_tables_extracted = True
                tbl_id = f"blk_{page_idx}_{ro}"
                metadata = {}
                if merged_cells:
                    metadata["merged_cells"] = merged_cells
                metadata["extraction_layer"] = extraction_layer
                metadata["extraction_confidence"] = extraction_confidence
                block = Block(
                    block_id=tbl_id,
                    block_type="table",
                    bbox=zone.bbox,
                    reading_order=ro,
                    page=page_idx + 1,
                    raw_content=tbl,
                )
                result_blocks.append(block)
                ro += 1

        # Perf #11: Update PageState with extraction results
        if zone_tables_extracted and page_tables:
            first_tbl = page_tables[0]
            if first_tbl and len(first_tbl) >= 2:
                if hasattr(self, "_page_state"):
                    self._page_state.update(
                        header=first_tbl[0],
                        layer=extraction_layer,
                        confidence=extraction_confidence,
                    )

        if not zone_tables_extracted:
            fallback_rows = _reconstruct_rows_from_chars(zone.chars)
            if fallback_rows:
                fb_id = f"blk_{page_idx}_{ro}"
                block = Block(
                    block_id=fb_id,
                    block_type="table",
                    bbox=zone.bbox,
                    reading_order=ro,
                    page=page_idx + 1,
                    raw_content=fallback_rows,
                )
                result_blocks.append(block)
                ro += 1

        # ── PID Loop: Degradation Resampling ──
        logger.info(
            f"Trace PID Before Block: router={bool(_router)}, zone_tables_extracted={zone_tables_extracted}, page_tables=[{len(page_tables) if page_tables else 0}], extraction_confidence={extraction_confidence}"
        )

        if _router and zone_tables_extracted and page_tables and extraction_confidence < 0.85:
            original_conf = extraction_confidence
            best_table = page_tables[0]

            # Retry 1: Parameter Shift Resampling (Digital only)
            if is_digital:
                logger.info(
                    f"[DocMirror] PID Loop Retry 1: Triggering parameter shift resampling on page {page_idx} (conf={original_conf:.2f})"
                )
                re_tables, re_layer, re_conf = extract_tables_layered(
                    page_plum,
                    table_zone_bbox=zone.bbox,
                    document_page_count=len(fitz_doc),
                    fitz_page=fitz_page,
                    watermark_filtered=_watermark_filtered,
                    layer_hint=None,
                    table_template=global_table_template,
                    pid_resample=True,
                )
                if re_tables and re_conf > original_conf:
                    logger.info(
                        f"[DocMirror] PID Loop Retry 1 Success: conf boosted to {re_conf:.2f}. Adopting new parameters."
                    )
                    best_table = re_tables[0]
                    extraction_confidence = re_conf
                    for i in range(len(result_blocks) - 1, -1, -1):
                        if result_blocks[i].block_type == "table":
                            result_blocks[i] = Block(
                                block_id=result_blocks[i].block_id,
                                block_type="table",
                                bbox=zone.bbox,
                                reading_order=result_blocks[i].reading_order,
                                page=page_idx + 1,
                                raw_content=best_table,
                            )
                            break

            # Retry 2: Visual Optical Degradation
            if extraction_confidence < 0.85 and _router.should_enhance_table(best_table, extraction_confidence):
                try:
                    high_dpi = _router._high_dpi
                    logger.warning(
                        f"[DocMirror] PID Loop Retry 2: Total degradation to Vision/OCR "
                        f"on page {page_idx} at {high_dpi} DPI "
                        f"(confidence={extraction_confidence:.2f})"
                    )
                    re_result = analyze_scanned_page(
                        fitz_page,
                        page_idx,
                        table_bbox=zone.bbox,
                        target_dpi=high_dpi,
                    )
                    if re_result and re_result.get("table"):
                        re_table = re_result["table"]
                        if len(re_table) >= len(best_table):
                            for i in range(len(result_blocks) - 1, -1, -1):
                                if result_blocks[i].block_type == "table":
                                    result_blocks[i] = Block(
                                        block_id=result_blocks[i].block_id,
                                        block_type="table",
                                        bbox=zone.bbox,
                                        reading_order=result_blocks[i].reading_order,
                                        page=page_idx + 1,
                                        raw_content=re_table,
                                    )
                                    break
                            if hasattr(self, "_page_state"):
                                self._page_state.reset()
                except Exception as e:
                    logger.debug(f"[DocMirror] Quality Router: OCR Degradation skipped: {e}")

        return result_blocks, extraction_layer, extraction_confidence, _table_ms, zone_tables_extracted

    def _handle_text_zone(
        self,
        zone,
        block_id: str,
        page_idx: int,
        fitz_page,
        layout_al,
        style_map: dict[str, Style],
        reading_order: int,
    ) -> Block | None:
        """Handle an unknown/text-type zone → Block conversion.

        Returns:
            A text Block if content was extracted, otherwise None.
        """
        text_content = None
        try:
            from fitz import Rect

            text_content = fitz_page.get_textbox(Rect(*zone.bbox)).strip()
        except Exception as exc:
            logger.debug(f"fitz textbox extraction: suppressed {exc}")
            text_content = zone.text.strip() if zone.text else ""

        if not text_content and zone.text:
            text_content = zone.text.strip()

        # P5: DET/REC Decoupling — crop ROI and run OCR on scanned content
        if not text_content and layout_al.is_scanned:
            try:
                import cv2
                import fitz
                import numpy as np

                from docmirror.core.ocr.vision.rapidocr_engine import get_ocr_engine

                ocr_engine = get_ocr_engine()
                if ocr_engine and ocr_engine._engine:
                    bx0, by0, bx1, by1 = zone.bbox
                    rect = fitz.Rect(max(0, bx0 - 5), max(0, by0 - 5), bx1 + 5, by1 + 5)
                    pix_patch = fitz_page.get_pixmap(dpi=300, clip=rect)
                    img_patch = np.frombuffer(pix_patch.samples, dtype=np.uint8).reshape(
                        pix_patch.h, pix_patch.w, pix_patch.n
                    )
                    if pix_patch.n == 3:
                        img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)
                    elif pix_patch.n == 4:
                        img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGBA2BGR)
                    words = ocr_engine.detect_image_words(img_patch, multi_scale=False)
                    if words:
                        text_content = " ".join([w[4] for w in sorted(words, key=lambda w: (w[1], w[0]))])
            except Exception as e:
                logger.debug(f"[DocMirror] Zone OCR fallback/crop failed: {e}")

        if text_content:
            return Block(
                block_id=block_id,
                block_type="text",
                bbox=zone.bbox,
                reading_order=reading_order,
                page=page_idx + 1,
                raw_content=text_content,
                spans=self._build_spans(text_content, zone.bbox, style_map),
            )
        return None

    def _extract_page_images(
        self,
        fitz_page,
        fitz_doc,
        page_idx: int,
        blocks: list[Block],
        reading_order: int,
    ) -> tuple[list[Block], int]:
        """Extract images from a page and create image blocks.

        Returns:
            (new_blocks, updated_reading_order)
        """
        new_blocks: list[Block] = []
        try:
            for img_info in fitz_page.get_images(full=True):
                xref = img_info[0]
                try:
                    img_data = fitz_doc.extract_image(xref)
                    if not img_data or not img_data.get("image"):
                        continue
                    img_bytes = img_data["image"]
                    img_rects = fitz_page.get_image_rects(xref)
                    if not img_rects:
                        continue
                    img_rect = img_rects[0]
                    img_bbox = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)

                    if (img_rect.x1 - img_rect.x0) < 50 or (img_rect.y1 - img_rect.y0) < 50:
                        continue

                    caption = None
                    caption_y_range = (img_rect.y1, img_rect.y1 + 30)
                    for existing_block in blocks:
                        if existing_block.block_type == "text" and existing_block.raw_content:
                            bx0, by0, bx1, by1 = existing_block.bbox
                            if (
                                caption_y_range[0] <= by0 <= caption_y_range[1]
                                and bx0 < img_rect.x1
                                and bx1 > img_rect.x0
                            ):
                                caption = existing_block.raw_content
                                break

                    img_id = f"blk_{page_idx}_{reading_order}"
                    new_blocks.append(
                        Block(
                            block_id=img_id,
                            block_type="image",
                            bbox=img_bbox,
                            reading_order=reading_order,
                            page=page_idx + 1,
                            raw_content=img_bytes,
                            caption=caption,
                        )
                    )
                    reading_order += 1
                except Exception as exc:
                    logger.debug(f"image extraction: suppressed {exc}")
                    continue
        except Exception as e:
            logger.debug(f"[DocMirror] image extraction skip: {e}")
        return new_blocks, reading_order

    def _fallback_table_extraction(
        self,
        page_plum,
        fitz_page,
        fitz_doc,
        page_idx: int,
        layout_al,
        reading_order: int,
        is_digital: bool,
        _watermark_filtered: bool,
        _router,
        global_table_template,
    ) -> tuple[list[Block], str, float]:
        """Fallback path: layout analysis found table but zone detection didn't.

        Returns:
            (blocks, extraction_layer, extraction_confidence)
        """
        logger.info(
            f"[Extractor] Page {page_idx}: Detected Legacy fallback (Rule-based recovery for {len(layout_al.get('table', []))} layout zones)"
        )
        result_blocks: list[Block] = []
        page_tables, extraction_layer, extraction_confidence = extract_tables_layered(
            page_plum,
            document_page_count=len(fitz_doc),
            fitz_page=fitz_page,
            watermark_filtered=_watermark_filtered,
            layer_hint=(
                self._page_state.winning_layer
                if hasattr(self, "_page_state") and self._page_state.should_use_hint()
                else None
            ),
            table_template=global_table_template,
        )

        # ── PID Loop (Fallback Path) ──
        if page_tables and extraction_confidence < 0.85:
            # Retry 1: Parameter Shift Resampling (Digital only)
            if is_digital:
                logger.info(
                    f"[DocMirror] PID Loop Retry 1 (Fallback): Triggering parameter shift resampling on page {page_idx} (conf={extraction_confidence:.2f})"
                )
                re_tables, re_layer, re_conf = extract_tables_layered(
                    page_plum,
                    document_page_count=len(fitz_doc),
                    fitz_page=fitz_page,
                    watermark_filtered=_watermark_filtered,
                    layer_hint=None,
                    table_template=global_table_template,
                    pid_resample=True,
                )
                if re_tables and re_conf > extraction_confidence:
                    logger.info(
                        f"[DocMirror] PID Loop Retry 1 Success (Fallback): conf boosted to {re_conf:.2f}. Adopting new parameters."
                    )
                    page_tables = re_tables
                    extraction_confidence = re_conf
                    extraction_layer = re_layer

            # Retry 2: Visual Optical Degradation
            if (
                extraction_confidence < 0.85
                and _router
                and _router.should_enhance_table(page_tables[0] if page_tables else [], extraction_confidence)
            ):
                try:
                    high_dpi = _router._high_dpi
                    logger.warning(
                        f"[DocMirror] PID Loop Retry 2 (Fallback): Total degradation to Vision/OCR "
                        f"on page {page_idx} at {high_dpi} DPI "
                        f"(confidence={extraction_confidence:.2f})"
                    )
                    re_result = analyze_scanned_page(
                        fitz_page,
                        page_idx,
                        target_dpi=high_dpi,
                    )
                    if re_result and re_result.get("table"):
                        re_table = re_result["table"]
                        if len(re_table) >= len(page_tables[0] if page_tables else []):
                            page_tables = [re_table]
                            if hasattr(self, "_page_state"):
                                self._page_state.reset()
                except Exception as e:
                    logger.debug(f"[DocMirror] Quality Router: OCR Degradation (Fallback) skipped: {e}")

        ro = reading_order
        for tbl in page_tables:
            if tbl and len(tbl) >= 1:
                tbl_id = f"blk_{page_idx}_{ro}"
                result_blocks.append(
                    Block(
                        block_id=tbl_id,
                        block_type="table",
                        reading_order=ro,
                        page=page_idx + 1,
                        raw_content=tbl,
                    )
                )
                ro += 1

        return result_blocks, extraction_layer, extraction_confidence

    # ═══════════════════════════════════════════════════════════════════════════
    # Single-Page Extraction (extracted from extract())
    # ═══════════════════════════════════════════════════════════════════════════

    def _extract_page(
        self,
        ctx: PageExtractionContext,
    ) -> tuple[PageLayout, list[str], str, float]:
        """Single-page extraction: Zone -> Block conversion.

        Args:
            ctx: PageExtractionContext with all page-level parameters.

        Returns:
            (PageLayout, ocr_text_parts, extraction_layer, extraction_confidence)
        """
        # Unpack context for local use
        page_plum = ctx.page_plum
        fitz_page = ctx.fitz_page
        fitz_doc = ctx.fitz_doc
        page_idx = ctx.page_idx
        layout_al = ctx.layout_al
        is_digital = ctx.is_digital
        strategy_params = ctx.strategy_params or {}
        page_quality = ctx.page_quality
        content_type = ctx.content_type
        zone_template = ctx.zone_template
        global_table_template = ctx.global_table_template

        width, height = FitzEngine.get_page_dimensions(fitz_page)

        # -- Adaptive Quality Router initialization --
        try:
            from .quality_router import AdaptiveQualityRouter

            _router = AdaptiveQualityRouter(strategy_params)
        except Exception as exc:
            logger.debug(f"QualityRouter init: suppressed {exc}")
            _router = None

        # -- Character-level pre-processing --
        # Perf #10: Always run basic watermark filter (is_watermark_char is O(n),
        # <1ms, catches rotated/skewed watermark text).  Only skip the expensive
        # deep statistical separation for table_dominant documents.
        _use_deep_watermark = False
        if content_type != "table_dominant":
            if _router and strategy_params:
                _use_deep_watermark = not strategy_params.get("skip_watermark_filter", False) and page_quality < 85

        _chars_before_wm = len(page_plum.chars)
        if _use_deep_watermark:
            try:
                from ..utils.watermark import separate_watermark_layer

                page_plum = separate_watermark_layer(page_plum)
            except Exception as exc:
                logger.debug(f"deep watermark separation: suppressed {exc}")
                page_plum = filter_watermark_page(page_plum)
            _watermark_filtered = len(page_plum.chars) < _chars_before_wm
            page_plum = _dedup_overlapping_chars(page_plum)
        else:
            # S1: Fused single-pass watermark + dedup (2×O(N) → 1×O(N))
            from ..utils.watermark import fused_filter_and_dedup

            page_plum, _watermark_filtered = fused_filter_and_dedup(page_plum)

        # -- Spatial partitioning --
        _seg_t = _clock()
        _used_template = False
        zones = None

        # Perf #9: Try zone template first (skips full segmentation)
        if zone_template is not None:
            from ..layout.layout_analysis import apply_zone_template

            zones = apply_zone_template(zone_template, page_plum, page_idx)
            if zones is not None:
                _used_template = True

        # Fallback: full segmentation
        if zones is None:
            if self._layout_detector:
                zones = self._model_segmentation(fitz_page, page_plum, page_idx)
            else:
                zones = segment_page_into_zones(page_plum, page_idx)
        _seg_ms = (_clock() - _seg_t) * 1000
        if _used_template:
            logger.debug(f"[DocMirror] Perf #9: page {page_idx} segmentation via template ({_seg_ms:.0f}ms)")

        # Perf #9: Build/rebuild zone template when full segmentation was used.
        # This handles the common case where page 0 is a cover page with a
        # different layout — the template is rebuilt from the first page that
        # falls back to full segmentation (e.g., page 1).
        _should_build_template = (
            not _used_template  # Full segmentation was performed
            and zones  # Got valid zones
            and len(zones) >= 2  # Enough zones for a meaningful template
        )
        if _should_build_template:
            try:
                from ..layout.layout_analysis import build_zone_template

                width_plum = page_plum.width if hasattr(page_plum, "width") else width
                height_plum = page_plum.height if hasattr(page_plum, "height") else height
                new_template = build_zone_template(zones, width_plum, height_plum, page_idx)
                if new_template.zone_count > 0:
                    self._zone_template = new_template
                    logger.debug(
                        f"[DocMirror] Perf #9: zone template {'rebuilt' if page_idx > 0 else 'built'} "
                        f"({new_template.zone_count} zones) from page {page_idx}"
                    )
            except Exception as exc:
                logger.debug(f"[DocMirror] Perf #9: template build failed: {exc}")

        # -- Extract visual features (styles) --
        style_map = self._extract_page_styles(fitz_page)

        # -- Zone -> Block conversion --
        blocks: list[Block] = []
        reading_order = 0
        page_has_table = False
        extraction_layer = "unknown"
        extraction_confidence = 0.0
        ocr_text_parts: list[str] = []
        _formula_ms = 0.0  # timing accumulator: formula recognition
        _table_ms = 0.0  # timing accumulator: table extraction
        semantic_zones: dict[str, list[str]] = {
            "title_area": [],
            "metadata_area": [],
            "table_area": [],
            "text_area": [],
            "footer": [],
            "pagination": [],
        }

        for zone in zones:
            block_id = f"blk_{page_idx}_{reading_order}"

            if zone.type == "footer":
                block = Block(
                    block_id=block_id,
                    block_type="footer",
                    bbox=zone.bbox,
                    reading_order=reading_order,
                    page=page_idx + 1,
                    raw_content=zone.text,
                )
                blocks.append(block)
                if any(c.isdigit() for c in zone.text) and len(zone.text) < 10:
                    semantic_zones["pagination"].append(block_id)
                else:
                    semantic_zones["footer"].append(block_id)
                reading_order += 1
                continue

            if zone.type == "title":
                h_level = self._infer_heading_level(zone.text, style_map)
                block = Block(
                    block_id=block_id,
                    block_type="title",
                    bbox=zone.bbox,
                    reading_order=reading_order,
                    page=page_idx + 1,
                    raw_content=zone.text,
                    spans=self._build_spans(zone.text, zone.bbox, style_map),
                    heading_level=h_level,
                )
                blocks.append(block)
                semantic_zones["title_area"].append(block_id)
                reading_order += 1
                continue

            if zone.type == "summary":
                pairs: dict[str, str] = {}
                _extract_summary_entities(zone.chars, pairs)
                if pairs:
                    block = Block(
                        block_id=block_id,
                        block_type="key_value",
                        bbox=zone.bbox,
                        reading_order=reading_order,
                        page=page_idx + 1,
                        raw_content=pairs,
                    )
                    blocks.append(block)
                    semantic_zones["metadata_area"].append(block_id)
                    reading_order += 1
                continue

            if zone.type == "formula":
                fml_block, fml_ms = self._handle_formula_zone(
                    zone,
                    block_id,
                    page_idx,
                    width,
                    height,
                    content_type,
                    reading_order,
                )
                _formula_ms += fml_ms
                if fml_block:
                    blocks.append(fml_block)
                    reading_order += 1
                continue

            if zone.type == "data_table":
                tbl_blocks, extraction_layer, extraction_confidence, tbl_ms, zone_tables_extracted = (
                    self._handle_data_table_zone(
                        zone,
                        block_id,
                        page_idx,
                        page_plum,
                        fitz_page,
                        fitz_doc,
                        reading_order,
                        is_digital,
                        _watermark_filtered,
                        _router,
                        global_table_template,
                    )
                )
                _table_ms += tbl_ms
                if zone_tables_extracted:
                    page_has_table = True
                for b in tbl_blocks:
                    blocks.append(b)
                    if b.block_type == "table":
                        semantic_zones["table_area"].append(b.block_id)
                reading_order += len(tbl_blocks)
                continue

            # unknown or text → text block
            text_block = self._handle_text_zone(
                zone,
                block_id,
                page_idx,
                fitz_page,
                layout_al,
                style_map,
                reading_order,
            )
            if text_block:
                blocks.append(text_block)
                semantic_zones["text_area"].append(block_id)
                reading_order += 1

        # -- Image extraction --
        img_blocks, reading_order = self._extract_page_images(
            fitz_page,
            fitz_doc,
            page_idx,
            blocks,
            reading_order,
        )
        blocks.extend(img_blocks)

        # Fallback: layout analysis found table but zone did not detect any
        if not page_has_table and layout_al.has_table and not layout_al.is_scanned:
            fb_blocks, extraction_layer, extraction_confidence = self._fallback_table_extraction(
                page_plum,
                fitz_page,
                fitz_doc,
                page_idx,
                layout_al,
                reading_order,
                is_digital,
                _watermark_filtered,
                _router,
                global_table_template,
            )
            for b in fb_blocks:
                blocks.append(b)
                semantic_zones["table_area"].append(b.block_id)
            reading_order += len(fb_blocks)

        # OCR Fallback: scanned document without explicit layout zones
        if layout_al.is_scanned and not page_has_table and not zones:
            ocr_result = analyze_scanned_page(fitz_doc[page_idx], page_idx)
            if ocr_result:
                ocr_id = f"blk_{page_idx}_{reading_order}"
                blocks.append(
                    Block(
                        block_id=ocr_id,
                        block_type="table",
                        reading_order=reading_order,
                        page=page_idx + 1,
                        raw_content=ocr_result["table"],
                    )
                )
                semantic_zones["table_area"].append(ocr_id)
                reading_order += 1
                if ocr_result.get("header_text"):
                    ocr_text_parts.append(ocr_result["header_text"])

        page_layout = PageLayout(
            page_number=page_idx + 1,
            width=width,
            height=height,
            blocks=tuple(blocks),
            semantic_zones=semantic_zones,
            is_scanned=layout_al.is_scanned,
        )
        # ── Per-page timing log ──
        logger.debug(
            f"[DocMirror] ⏱ Page {page_idx}: "
            f"segmentation={_seg_ms:.0f}ms | "
            f"table_extraction={_table_ms:.0f}ms | "
            f"formula_ocr={_formula_ms:.0f}ms | "
            f"zones={len(zones)} | blocks={len(blocks)}"
        )
        return page_layout, ocr_text_parts, extraction_layer, extraction_confidence

    # ═══════════════════════════════════════════════════════════════════════════
    # Internal Methods
    # ═══════════════════════════════════════════════════════════════════════════

    # _run_seal_detection_if_enabled removed -- seal detection via __init__(seal_detector_fn=...) dependency injection

    def _model_segmentation(self, fitz_page, page_plum, page_idx: int) -> list:
        """Model-based layout analysis: render page image -> DocLayout-YOLO inference -> Zone list.

        Falls back to rule-based method on failure.
        """
        try:
            import numpy as np

            # Render page as image (configurable DPI, default 200)
            render_dpi = self._model_render_dpi
            pixmap = fitz_page.get_pixmap(dpi=render_dpi)
            img_data = pixmap.samples
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
            # RGBA → RGB
            if pixmap.n == 4:
                img = img[:, :, :3]

            # Model inference
            regions = self._layout_detector.detect(img, confidence_threshold=0.4)

            if not regions:
                logger.debug(f"[DocMirror] model detected 0 regions on page {page_idx}, falling back")
                return segment_page_into_zones(page_plum, page_idx)

            # Coordinate conversion: pixel space -> PDF point space (72 DPI)
            scale = 72.0 / render_dpi
            zones = []
            for region in regions:
                rx0, ry0, rx1, ry1 = region.bbox
                zone_bbox = (rx0 * scale, ry0 * scale, rx1 * scale, ry1 * scale)

                # Get text within region
                zone_text = ""
                zone_chars = []
                try:
                    for char in page_plum.chars:
                        cx = float(char.get("x0", 0))
                        cy = float(char.get("top", 0))
                        if zone_bbox[0] <= cx <= zone_bbox[2] and zone_bbox[1] <= cy <= zone_bbox[3]:
                            zone_chars.append(char)
                            zone_text += char.get("text", "")
                except Exception as exc:
                    logger.debug(f"operation: suppressed {exc}")

                zones.append(
                    Zone(
                        type=region.category,
                        bbox=zone_bbox,
                        text=zone_text.strip(),
                        chars=zone_chars,
                        confidence=region.confidence,
                    )
                )

            logger.info(f"[DocMirror] model segmentation: {len(zones)} zones on page {page_idx}")
            return zones

        except Exception as e:
            logger.warning(f"[DocMirror] model segmentation failed: {e}, falling back to rules")
            return segment_page_into_zones(page_plum, page_idx)

    def _crop_zone_image(self, fitz_page, bbox) -> bytes:
        """Crop an image region from a page at the given bbox."""
        try:
            import fitz as pymupdf

            rect = pymupdf.Rect(*bbox)
            clip = fitz_page.get_pixmap(clip=rect, dpi=300)
            return clip.tobytes("png")
        except Exception as exc:
            logger.debug(f"crop_image: suppressed {exc}")
            return b""

    def _recognize_formula(self, image_bytes: bytes) -> str:
        """Formula image -> LaTeX (delegated to FormulaEngine).

        FormulaEngine internally selects backend by strategy:
            UniMERNet ONNX > rapid_latex_ocr > empty string
        """
        return self._formula_engine.recognize(image_bytes)

    def _extract_page_styles(self, fitz_page) -> dict[str, Style]:
        """Extract visual features of text within the page."""
        style_map: dict[str, Style] = {}
        try:
            spans = FitzEngine.extract_page_blocks_with_style(fitz_page)
            for span_info in spans:
                text = span_info["text"].strip()
                if not text:
                    continue
                key = text[:20]
                flags = span_info["flags"]
                color_int = span_info["color"]
                style = Style(
                    font_name=span_info["font_name"],
                    font_size=round(span_info["font_size"], 1),
                    color=f"#{color_int:06x}" if isinstance(color_int, int) else "#000000",
                    is_bold=bool(flags & 16),
                    is_italic=bool(flags & 2),
                )
                style_map[key] = style
        except Exception as e:
            logger.debug(f"[DocMirror] style extraction error: {e}")
        return style_map

    def _build_spans(
        self,
        text: str,
        bbox: tuple[float, float, float, float],
        style_map: dict[str, Style],
    ) -> tuple[TextSpan, ...]:
        """Build TextSpan sequence from text + coordinates + style_map."""
        if not text:
            return ()

        key = text[:20]
        style = style_map.get(key, Style())
        return (TextSpan(text=text, bbox=bbox, style=style),)

    def _infer_heading_level(
        self,
        text: str,
        style_map: dict[str, Style],
    ) -> int | None:
        """Infer heading level based on font size and bold attributes.

        Strategy:
            - Collect all font sizes from style_map
            - Calculate median body text font size as baseline
            - Determine h1/h2/h3 based on relative size + bold:
              * font_size >= baseline * 1.6 and bold -> h1
              * font_size >= baseline * 1.2 and bold -> h2
              * bold only -> h3
              * not bold but notably larger font -> h2

        Returns:
            1, 2, 3 or None (when indeterminate)
        """
        if not text:
            return None

        key = text[:20]
        style = style_map.get(key)
        if not style:
            return None

        # Collect all font sizes within the page as context
        all_sizes = [s.font_size for s in style_map.values() if s.font_size > 0]
        if not all_sizes:
            return 3 if style.is_bold else None

        all_sizes.sort()
        # Median as body text baseline
        mid = len(all_sizes) // 2
        baseline = all_sizes[mid] if all_sizes else 10.0

        fs = style.font_size
        is_bold = style.is_bold

        if baseline <= 0:
            return 3 if is_bold else None

        ratio = fs / baseline

        if is_bold and ratio >= 1.6:
            return 1
        elif is_bold and ratio >= 1.2:
            return 2
        elif is_bold:
            return 3
        elif ratio >= 1.6:
            return 2  # large font but not bold -> h2
        else:
            return None

    def _collect_kv_entities(self, pages: list[PageLayout]) -> dict[str, str]:
        """Collect entities from extracted key_value blocks — delegated to entity_collector module."""
        return collect_kv_entities(pages)

    def _merge_cross_page_tables(self, pages: list[PageLayout]) -> list[PageLayout]:
        """Cross-page table merge -- delegated to table_merger module."""
        from ..table.merger import merge_cross_page_tables

        return merge_cross_page_tables(pages)

    def _post_process_tables(self, pages: list[PageLayout]) -> list[PageLayout]:
        """Table post-processing — delegated to table_postprocessor module."""
        return process_page_tables(pages)
