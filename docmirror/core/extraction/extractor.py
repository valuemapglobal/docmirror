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
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...models.domain import (
    BaseResult,
    Block,
    PageLayout,
    Style,
    TextSpan,
)
from .foundation import FitzEngine
from ..layout.layout_analysis import analyze_document_layout, segment_page_into_zones, Zone, extract_tables_layered, get_last_layer_timings, analyze_scanned_page, filter_watermark_page, _dedup_overlapping_chars, _reconstruct_rows_from_chars, _extract_summary_entities, post_process_table, _strip_preamble, preprocess_document, _is_header_row
from ..exceptions import ExtractionError

logger = logging.getLogger(__name__)


class CoreExtractor:
    """
    Core extractor — generates immutable BaseResult from PDF.

    Usage::

        extractor = CoreExtractor()
        result = await extractor.extract(Path("sample.pdf"))
        # result is frozen BaseResult, immutable

    All low-level functions come from MultiModal.core submodules (self-contained).
    """

    def __init__(self, seal_detector_fn=None, layout_model_path: Optional[str] = None,
                 max_page_concurrency: int = 1,
                 formula_model_path: Optional[str] = None,
                 model_render_dpi: int = 200):
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
    def _image_to_virtual_pdf(image_path: Path) -> "fitz.Document":
        """Convert image to a virtual single-page PDF, pre-scaling large images to max 4096px."""
        import fitz

        # Read image
        img_doc = fitz.open(str(image_path))
        if len(img_doc) == 0:
            raise ValueError(f"Cannot open image: {image_path}")

        # Pre-scale large images (prevent memory explosion)
        page = img_doc[0]
        w, h = page.rect.width, page.rect.height
        max_dim = 4096
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            logger.info(f"[DocMirror] Image pre-scaled: {w:.0f}x{h:.0f} -> {w*scale:.0f}x{h*scale:.0f}")
            # use pixmap scaling
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)
            img_doc.close()
            # Rebuild from scaled pixmap
            new_doc = fitz.open()
            new_page = new_doc.new_page(width=pix.width, height=pix.height)
            new_page.insert_image(new_page.rect, pixmap=pix)
            return new_doc

        # Normal size: convert directly to PDF
        pdf_bytes = img_doc.convert_to_pdf()
        img_doc.close()
        return fitz.open("pdf", pdf_bytes)

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
        t0 = time.time()
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
                logger.info(f"[DocMirror] Text layer missing, marked as scanned document")

            # === Step 1.5: Pre-analysis (human cognition stage 2) ===
            from .pre_analyzer import PreAnalyzer
            pre_analysis = await asyncio.to_thread(PreAnalyzer().analyze, fitz_doc)

            # Define the heavy CPU-bound parsing block to run in a thread
            def _process_pdf_sync(fitz_doc, pre_analysis, has_text):
                # ── Per-step timing instrumentation ──
                _perf: Dict[str, float] = {}
                _page_perf: list = []  # per-page timing breakdown

                # === Step 2: Layout analysis ===
                _t = time.time()
                page_layouts_al = analyze_document_layout(fitz_doc)
                _perf['layout_analysis_ms'] = (time.time() - _t) * 1000

                # Extract full text
                full_text_parts = []
                for page in fitz_doc:
                    full_text_parts.append(page.get_text())
                full_text_raw = "\n\n".join(full_text_parts)
                full_text = unicodedata.normalize("NFKC", full_text_raw)
    
                # === Step 3: Per-page extraction -- Zone -> Block ===
                # P0: Per-page hybrid routing — each page individually assessed
                # P1: Smart early-exit — honor DOCMIRROR_MAX_PAGES at core layer
                pages: List[PageLayout] = []
                ocr_text_parts: List[str] = []
                extraction_layer: str = "unknown"  # last strategy layer used by extract_tables_layered
                extraction_confidence: float = 0.0  # last extraction confidence
    
                max_pages = int(os.environ.get("DOCMIRROR_MAX_PAGES", "0"))
    
                # Per-page text presence detection (>50 chars = has text layer)
                _TEXT_THRESHOLD = 50
                page_has_text = []
                for p_idx in range(len(fitz_doc)):
                    chars = len(fitz_doc[p_idx].get_text().strip())
                    page_has_text.append(chars > _TEXT_THRESHOLD)
    
                hybrid_doc = has_text and not all(page_has_text)
                if hybrid_doc:
                    scanned_indices = [i for i, v in enumerate(page_has_text) if not v]
                    logger.info(
                        f"[DocMirror] Hybrid document detected: "
                        f"{len(scanned_indices)} scanned pages out of {len(page_has_text)}"
                    )
    
                # Open pdfplumber once for all digital pages
                import pdfplumber
                plumber_path = cleaned_path if not is_image_input else file_path
                plumber_doc = None
                if has_text or hybrid_doc:
                    plumber_doc = pdfplumber.open(str(plumber_path))
    
                try:
                    for page_idx, layout_al in enumerate(page_layouts_al):
                        # P1: Smart early-exit
                        if max_pages > 0 and page_idx >= max_pages:
                            logger.info(
                                f"[DocMirror] Early exit at page {page_idx} "
                                f"(DOCMIRROR_MAX_PAGES={max_pages})"
                            )
                            break
    
                        fitz_page = fitz_doc[page_idx]
    
                        # P0: Per-page routing decision
                        _pt = time.time()
                        try:
                            if page_has_text[page_idx] and plumber_doc:
                                # Digital page → pdfplumber + fitz joint extraction
                                page_plum = plumber_doc.pages[page_idx]
                                page_layout, page_ocr_parts, extraction_layer, extraction_confidence = (
                                    self._extract_page(
                                        page_plum=page_plum,
                                        fitz_page=fitz_page,
                                        fitz_doc=fitz_doc,
                                        page_idx=page_idx,
                                        layout_al=layout_al,
                                        cleaned_path=plumber_path,
                                        is_digital=True,
                                        strategy_params=pre_analysis.strategy_params,
                                        page_quality=dict(
                                            pre_analysis.page_quality_map
                                        ).get(page_idx, pre_analysis.avg_image_quality),
                                    )
                                )
                                pages.append(page_layout)
                                ocr_text_parts.extend(page_ocr_parts)
                            else:
                                # Scanned page → OCR extraction
                                page_layout = self._extract_scanned_page(
                                    fitz_page=fitz_page,
                                    page_idx=page_idx,
                                )
                                pages.append(page_layout)
                                # Collect OCR text
                                for blk in page_layout.blocks:
                                    if blk.block_type == "text" and blk.raw_content:
                                        ocr_text_parts.append(str(blk.raw_content))
                        except Exception as page_exc:
                            logger.error(
                                f"[DocMirror] ❌ Page {page_idx} extraction FAILED: {page_exc}",
                                exc_info=True,
                            )
                        _page_perf.append({'page': page_idx, 'total': (time.time() - _pt) * 1000})
                finally:
                    if plumber_doc:
                        plumber_doc.close()
    
                    # Merge OCR text into full text
                    if ocr_text_parts:
                        full_text = full_text + "\n\n" + "\n\n".join(ocr_text_parts)
    
                    # ═══ Step 4: Cross-page merge ═══
                    _t = time.time()
                    pages = self._merge_cross_page_tables(pages)
                    _perf['cross_page_merge_ms'] = (time.time() - _t) * 1000
    
                    # ═══ Step 5: Table post-processing ═══
                    _t = time.time()
                    pages = self._post_process_tables(pages)
                    _perf['table_postprocess_ms'] = (time.time() - _t) * 1000

                    # ── Log performance breakdown ──
                    logger.info(
                        "[DocMirror] ⏱ Pipeline timing breakdown:\n"
                        + "\n".join(f"    {k}: {v:.0f}ms" for k, v in _perf.items())
                    )
                    if _page_perf:
                        for pp in _page_perf:
                            logger.info(
                                f"[DocMirror] ⏱ Page {pp['page']}: "
                                + " | ".join(f"{k}={v:.0f}ms" for k, v in pp.items() if k != 'page')
                            )
                    
                    return pages, full_text, extraction_layer, extraction_confidence, _perf, _page_perf

            # Execute the heavy synchronous block in a thread
            pages, full_text, extraction_layer, extraction_confidence, _perf, _page_perf = await asyncio.to_thread(_process_pdf_sync, fitz_doc, pre_analysis, has_text)

            # === Step 6: Assemble BaseResult ===
            elapsed = (time.time() - t0) * 1000

            total_blocks = sum(len(p.blocks) for p in pages)
            table_count = sum(
                1 for p in pages for b in p.blocks if b.block_type == "table"
            )

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
                "scanned_pages": [
                    p.page_number for p in pages if p.is_scanned
                ],
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
                    col_count_stable = all(
                        len(row) == expected_cols for row in main_table
                    )
                    total_cells = sum(len(row) for row in main_table)
                    empty_cells = sum(
                        1 for row in main_table
                        for c in row if not (c or "").strip()
                    )
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

    def _extract_scanned_page(
        self, *, fitz_page, page_idx: int,
    ) -> "PageLayout":
        """Single-page OCR extraction for scanned documents.

        Uses ``ocr_extract_universal`` which auto-detects content type:
        - Table documents → existing table block pipeline
        - General documents → text blocks in reading order with real bboxes
        """
        from ..ocr.fallback import ocr_extract_universal

        width = fitz_page.rect.width
        height = fitz_page.rect.height
        blocks: List[Block] = []
        reading_order = 0

        try:
            ocr_result = ocr_extract_universal(fitz_page, page_idx)

            if ocr_result and isinstance(ocr_result, dict):
                content_type = ocr_result.get("content_type", "table")

                if content_type == "table":
                    # ── Table document: existing logic ──
                    header_text = ocr_result.get("header_text", "").strip()
                    if header_text:
                        blocks.append(Block(
                            block_id=f"blk_{page_idx}_{reading_order}",
                            block_type="text",
                            bbox=(0, 0, width, height * 0.1),
                            reading_order=reading_order,
                            page=page_idx + 1,
                            raw_content=header_text,
                        ))
                        reading_order += 1

                    table_data = ocr_result.get("table", [])
                    if table_data and len(table_data) >= 2:
                        blocks.append(Block(
                            block_id=f"blk_{page_idx}_{reading_order}",
                            block_type="table",
                            bbox=(0, height * 0.1, width, height * 0.9),
                            reading_order=reading_order,
                            page=page_idx + 1,
                            raw_content=table_data,
                        ))
                        reading_order += 1
                    elif table_data:
                        text_lines = [
                            " | ".join(str(c) for c in row if c)
                            for row in table_data if any(c for c in row)
                        ]
                        if text_lines:
                            blocks.append(Block(
                                block_id=f"blk_{page_idx}_{reading_order}",
                                block_type="text",
                                bbox=(0, height * 0.1, width, height * 0.9),
                                reading_order=reading_order,
                                page=page_idx + 1,
                                raw_content="\n".join(text_lines),
                            ))
                            reading_order += 1

                    footer_text = ocr_result.get("footer_text", "").strip()
                    if footer_text:
                        blocks.append(Block(
                            block_id=f"blk_{page_idx}_{reading_order}",
                            block_type="text",
                            bbox=(0, height * 0.9, width, height),
                            reading_order=reading_order,
                            page=page_idx + 1,
                            raw_content=footer_text,
                        ))
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

                    for line in lines:
                        text = line.get("text", "").strip()
                        if not text:
                            continue
                        ox0, oy0, ox1, oy1 = line.get("bbox", (0, 0, 0, 0))
                        bbox = (ox0 * sx, oy0 * sy, ox1 * sx, oy1 * sy)

                        blocks.append(Block(
                            block_id=f"blk_{page_idx}_{reading_order}",
                            block_type="text",
                            bbox=bbox,
                            reading_order=reading_order,
                            page=page_idx + 1,
                            raw_content=text,
                        ))
                        reading_order += 1

                    logger.info(
                        f"[DocMirror] OCR page {page_idx} (general): "
                        f"{len(lines)} text lines extracted"
                    )

        except Exception as e:
            logger.warning(f"[DocMirror] OCR failed on page {page_idx}: {e}")

        return PageLayout(
            page_number=page_idx + 1,
            blocks=tuple(blocks),
            is_scanned=True,
        )

    @staticmethod
    def _group_words_into_lines(
        words: List[dict], tolerance_ratio: float = 0.5
    ) -> List[List[dict]]:
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
        lines: List[List[dict]] = []
        current_line: List[dict] = [items[0][2]]
        current_y = items[0][0]

        for cy, h, w in items[1:]:
            if abs(cy - current_y) <= tolerance:
                current_line.append(w)
            else:
                # Sort line internally by x
                current_line.sort(
                    key=lambda word: word.get("bbox", (0,))[0]
                )
                lines.append(current_line)
                current_line = [w]
                current_y = cy

        if current_line:
            current_line.sort(key=lambda word: word.get("bbox", (0,))[0])
            lines.append(current_line)

        return lines

    # ═══════════════════════════════════════════════════════════════════════════
    # Single-Page Extraction (extracted from extract())
    # ═══════════════════════════════════════════════════════════════════════════

    def _extract_page(
        self, *, page_plum, fitz_page, fitz_doc, page_idx: int,
        layout_al, cleaned_path: Path,
        is_digital: bool = False,
        strategy_params: Optional[Dict[str, Any]] = None,
        page_quality: int = 100,
    ) -> Tuple["PageLayout", List[str], str, float]:
        """Single-page extraction: Zone -> Block conversion.

        Args:
            page_plum: pdfplumber page object.
            fitz_page: PyMuPDF page object.
            fitz_doc: PyMuPDF document object (for image extraction).
            page_idx: Page index (0-based).
            layout_al: Layout analysis result.
            cleaned_path: Pre-processed PDF path.
            strategy_params: PreAnalyzer strategy parameters (optional).
            page_quality: Image quality score (0-100) for this page.

        Returns:
            (PageLayout, ocr_text_parts, extraction_layer, extraction_confidence)
        """
        width, height = FitzEngine.get_page_dimensions(fitz_page)

        # -- Adaptive Quality Router initialization --
        try:
            from .quality_router import AdaptiveQualityRouter
            _router = AdaptiveQualityRouter(strategy_params)
        except Exception as exc:
            logger.debug(f"QualityRouter init: suppressed {exc}")
            _router = None

        # -- Character-level pre-processing --
        # Use deep watermark separation when router recommends it
        _use_deep_watermark = False
        if _router and strategy_params:
            _use_deep_watermark = not strategy_params.get(
                "skip_watermark_filter", False
            ) and page_quality < 85

        if _use_deep_watermark:
            try:
                from ..utils.watermark import separate_watermark_layer
                page_plum = separate_watermark_layer(page_plum)
            except Exception as exc:
                logger.debug(f"deep watermark separation: suppressed {exc}")
                page_plum = filter_watermark_page(page_plum)
        else:
            page_plum = filter_watermark_page(page_plum)
        page_plum = _dedup_overlapping_chars(page_plum)

        # -- Spatial partitioning --
        _seg_t = time.time()
        if self._layout_detector:
            zones = self._model_segmentation(fitz_page, page_plum, page_idx)
        else:
            zones = segment_page_into_zones(page_plum, page_idx)
        _seg_ms = (time.time() - _seg_t) * 1000

        # -- Extract visual features (styles) --
        style_map = self._extract_page_styles(fitz_page)

        # -- Zone -> Block conversion --
        blocks: List[Block] = []
        reading_order = 0
        page_has_table = False
        extraction_layer = "unknown"
        extraction_confidence = 0.0
        ocr_text_parts: List[str] = []
        _formula_ms = 0.0  # timing accumulator: formula recognition
        _table_ms = 0.0    # timing accumulator: table extraction
        semantic_zones: Dict[str, List[str]] = {
            "title_area": [], "metadata_area": [],
            "table_area": [], "text_area": [],
            "footer": [], "pagination": [],
        }

        for zone in zones:
            block_id = f"blk_{page_idx}_{reading_order}"

            if zone.type == "footer":
                block = Block(
                    block_id=block_id, block_type="footer",
                    bbox=zone.bbox, reading_order=reading_order,
                    page=page_idx + 1, raw_content=zone.text,
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
                    block_id=block_id, block_type="title",
                    bbox=zone.bbox, reading_order=reading_order,
                    page=page_idx + 1, raw_content=zone.text,
                    spans=self._build_spans(zone.text, zone.bbox, style_map),
                    heading_level=h_level,
                )
                blocks.append(block)
                semantic_zones["title_area"].append(block_id)
                reading_order += 1
                continue

            if zone.type == "summary":
                pairs: Dict[str, str] = {}
                _extract_summary_entities(zone.chars, pairs)
                if pairs:
                    block = Block(
                        block_id=block_id, block_type="key_value",
                        bbox=zone.bbox, reading_order=reading_order,
                        page=page_idx + 1, raw_content=pairs,
                    )
                    blocks.append(block)
                    semantic_zones["metadata_area"].append(block_id)
                    reading_order += 1
                continue

            if zone.type == "formula":
                _fml_t = time.time()

                # ── 3-tier formula zone gating (skip false-positive OCR) ──
                _skip_formula_ocr = False

                # Gate 1: YOLO confidence threshold
                if zone.confidence < 0.65:
                    _skip_formula_ocr = True
                    logger.debug(
                        f"formula gate: skipped zone (confidence={zone.confidence:.2f} < 0.65)"
                    )

                # Gate 2: Zone area filter (formulas are small)
                if not _skip_formula_ocr:
                    zone_area = (zone.bbox[2] - zone.bbox[0]) * (zone.bbox[3] - zone.bbox[1])
                    page_area = width * height
                    if page_area > 0 and zone_area > page_area * 0.3:
                        _skip_formula_ocr = True
                        logger.debug(
                            f"formula gate: skipped zone (area={zone_area:.0f} > 30% page)"
                        )

                # Gate 3: Character content pre-check (must have math indicators)
                if not _skip_formula_ocr:
                    _MATH_INDICATORS = set("∑∫∂√±≤≥≠∞∈∉∝∀∃αβγδεθλμπσφψω")
                    zone_text = zone.text or ""
                    has_math_chars = bool(set(zone_text) & _MATH_INDICATORS)
                    has_operator_pattern = bool(re.search(r'[=<>≤≥±×÷∑∫^_{}\\]', zone_text))
                    has_superscript = any(
                        c.get("size", 12) < 8 for c in zone.chars
                    ) if zone.chars else False
                    if not (has_math_chars or has_operator_pattern or has_superscript):
                        _skip_formula_ocr = True
                        logger.debug(
                            f"formula gate: skipped zone (no math indicators in '{zone_text[:30]}')"
                        )

                # K1: prefer extracting from character stream (zero latency, always attempted)
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
                _formula_ms += (time.time() - _fml_t) * 1000

                # If formula was extracted, emit as formula block; otherwise fall through to text
                if latex_str:
                    block = Block(
                        block_id=block_id, block_type="formula",
                        bbox=zone.bbox, reading_order=reading_order,
                        page=page_idx + 1,
                        raw_content=latex_str,
                    )
                    blocks.append(block)
                    reading_order += 1
                    continue
                elif _skip_formula_ocr and zone.text:
                    # Gated-out zone with text content → emit as text block instead
                    block = Block(
                        block_id=block_id, block_type="text",
                        bbox=zone.bbox, reading_order=reading_order,
                        page=page_idx + 1,
                        raw_content=zone.text,
                    )
                    blocks.append(block)
                    reading_order += 1
                    continue
                else:
                    continue

            if zone.type == "data_table":
                _tbl_t = time.time()
                page_has_table = True
                zone_tables_extracted = False

                # P3-2: Detect merged cells
                merged_cells = []
                try:
                    from ..table.extraction.engine import detect_merged_cells
                    merged_cells = detect_merged_cells(page_plum, table_zone_bbox=zone.bbox)
                except Exception as exc:
                    logger.debug(f"operation: suppressed {exc}")

                page_tables, extraction_layer, extraction_confidence = extract_tables_layered(
                    page_plum, table_zone_bbox=zone.bbox,
                )
                _table_ms += (time.time() - _tbl_t) * 1000
                for tbl in page_tables:
                    if tbl and len(tbl) >= 1:
                        zone_tables_extracted = True
                        tbl_id = f"blk_{page_idx}_{reading_order}"
                        metadata = {}
                        if merged_cells:
                            metadata["merged_cells"] = merged_cells
                        metadata["extraction_layer"] = extraction_layer
                        metadata["extraction_confidence"] = extraction_confidence
                        block = Block(
                            block_id=tbl_id, block_type="table",
                            bbox=zone.bbox, reading_order=reading_order,
                            page=page_idx + 1, raw_content=tbl,
                        )
                        blocks.append(block)
                        semantic_zones["table_area"].append(tbl_id)
                        reading_order += 1

                if not zone_tables_extracted:
                    fallback_rows = _reconstruct_rows_from_chars(zone.chars)
                    if fallback_rows:
                        fb_id = f"blk_{page_idx}_{reading_order}"
                        block = Block(
                            block_id=fb_id, block_type="table",
                            bbox=zone.bbox, reading_order=reading_order,
                            page=page_idx + 1, raw_content=fallback_rows,
                        )
                        blocks.append(block)
                        semantic_zones["table_area"].append(fb_id)
                        reading_order += 1

                # ── Adaptive: re-extract at high DPI if quality is poor ──
                if (
                    _router
                    and zone_tables_extracted
                    and not is_digital
                    and _router.should_enhance_table(
                        page_tables[0] if page_tables else [],
                        extraction_confidence,
                    )
                ):
                    try:
                        high_dpi = _router._high_dpi
                        logger.info(
                            f"[DocMirror] Quality Router: re-extracting table "
                            f"on page {page_idx} at {high_dpi} DPI "
                            f"(confidence={extraction_confidence:.2f})"
                        )
                        re_result = analyze_scanned_page(
                            fitz_page, page_idx,
                            table_bbox=zone.bbox,
                            target_dpi=high_dpi,
                        )
                        if re_result and re_result.get("table"):
                            re_table = re_result["table"]
                            if len(re_table) >= len(page_tables[0] if page_tables else []):
                                # Replace last table block
                                for i in range(len(blocks) - 1, -1, -1):
                                    if blocks[i].block_type == "table":
                                        blocks[i] = Block(
                                            block_id=blocks[i].block_id,
                                            block_type="table",
                                            bbox=zone.bbox,
                                            reading_order=blocks[i].reading_order,
                                            page=page_idx + 1,
                                            raw_content=re_table,
                                        )
                                        break
                    except Exception as e:
                        logger.debug(
                            f"[DocMirror] Quality Router: re-extraction "
                            f"skipped: {e}"
                        )
                continue

            # unknown or text \u2192 text block
            text_content = None
            try:
                from fitz import Rect
                # Component 3: CPU High-Speed PDF Fetch via PyMuPDF native
                # This bypasses pdfplumber char spacing errors and is much faster
                # when scaling to large document bounding boxes
                text_content = fitz_page.get_textbox(Rect(*zone.bbox)).strip()
            except Exception as exc:
                logger.debug(f"fitz textbox extraction: suppressed {exc}")
                text_content = zone.text.strip() if zone.text else ""

            if not text_content and zone.text:
                text_content = zone.text.strip()

            # P5: DET/REC Decoupling & Advanced Cropping
            # If the zone is still empty (scanned doc) and we have a layout model,
            # crop the ROI to save CPU and run OCR only on the crop with TPS Warping
            if not text_content and layout_al.is_scanned:
                try:
                    import cv2
                    import numpy as np
                    from docmirror.core.ocr.vision.rapidocr_engine import get_ocr_engine
                    ocr_engine = get_ocr_engine()
                    if ocr_engine and ocr_engine._engine:
                        bx0, by0, bx1, by1 = zone.bbox
                        # Expand boundaries by 5px to prevent clipping ascenders/descenders
                        rect = fitz.Rect(max(0, bx0 - 5), max(0, by0 - 5), bx1 + 5, by1 + 5)
                        
                        # High-DPI localized crop
                        pix_patch = fitz_page.get_pixmap(dpi=300, clip=rect)
                        img_patch = np.frombuffer(pix_patch.samples, dtype=np.uint8).reshape(pix_patch.h, pix_patch.w, pix_patch.n)
                        if pix_patch.n == 3:
                            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)
                        elif pix_patch.n == 4:
                            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGBA2BGR)
                        
                        # Local OCR on the tight crop (No NMS needed for small text lines)
                        # The RapidOCR REC network automatically utilizes its SVTR TPS Warping 
                        # natively to straighten out slightly warped document scans.
                        words = ocr_engine.detect_image_words(img_patch, multi_scale=False)
                        if words:
                            # Re-sort OCR words top-to-bottom, left-to-right
                            text_content = " ".join([w[4] for w in sorted(words, key=lambda w: (w[1], w[0]))])
                except Exception as e:
                    logger.debug(f"[DocMirror] Zone OCR fallback/crop failed: {e}")

            if text_content:
                block = Block(
                    block_id=block_id, block_type="text",
                    bbox=zone.bbox, reading_order=reading_order,
                    page=page_idx + 1, raw_content=text_content,
                    spans=self._build_spans(text_content, zone.bbox, style_map),
                )
                blocks.append(block)
                semantic_zones["text_area"].append(block_id)
                reading_order += 1

        # -- Image extraction --
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
                            if (caption_y_range[0] <= by0 <= caption_y_range[1]
                                    and bx0 < img_rect.x1 and bx1 > img_rect.x0):
                                caption = existing_block.raw_content
                                break

                    img_id = f"blk_{page_idx}_{reading_order}"
                    blocks.append(Block(
                        block_id=img_id, block_type="image",
                        bbox=img_bbox, reading_order=reading_order,
                        page=page_idx + 1, raw_content=img_bytes, caption=caption,
                    ))
                    reading_order += 1
                except Exception as exc:
                    logger.debug(f"image extraction: suppressed {exc}")
                    continue
        except Exception as e:
            logger.debug(f"[DocMirror] image extraction skip: {e}")

        # Fallback: layout analysis found table but zone did not detect any
        if not page_has_table and layout_al.has_table and not layout_al.is_scanned:
            page_tables, extraction_layer, extraction_confidence = extract_tables_layered(
                page_plum,
            )
            for tbl in page_tables:
                if tbl and len(tbl) >= 1:
                    tbl_id = f"blk_{page_idx}_{reading_order}"
                    blocks.append(Block(
                        block_id=tbl_id, block_type="table",
                        reading_order=reading_order, page=page_idx + 1,
                        raw_content=tbl,
                    ))
                    semantic_zones["table_area"].append(tbl_id)
                    reading_order += 1

        # OCR Fallback: scanned document without explicit layout zones
        if layout_al.is_scanned and not page_has_table and not zones:
            ocr_result = analyze_scanned_page(fitz_doc[page_idx], page_idx)
            if ocr_result:
                ocr_id = f"blk_{page_idx}_{reading_order}"
                blocks.append(Block(
                    block_id=ocr_id, block_type="table",
                    reading_order=reading_order, page=page_idx + 1,
                    raw_content=ocr_result["table"],
                ))
                semantic_zones["table_area"].append(ocr_id)
                reading_order += 1
                if ocr_result.get("header_text"):
                    ocr_text_parts.append(ocr_result["header_text"])

        page_layout = PageLayout(
            page_number=page_idx + 1,
            width=width, height=height,
            blocks=tuple(blocks),
            semantic_zones=semantic_zones,
            is_scanned=layout_al.is_scanned,
        )
        # ── Per-page timing log ──
        logger.info(
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

    def _model_segmentation(self, fitz_page, page_plum, page_idx: int) -> List:
        """Model-based layout analysis: render page image -> DocLayout-YOLO inference -> Zone list.

        Falls back to rule-based method on failure.
        """
        try:
            import numpy as np

            # Render page as image (configurable DPI, default 200)
            render_dpi = self._model_render_dpi
            pixmap = fitz_page.get_pixmap(dpi=render_dpi)
            img_data = pixmap.samples
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
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
                        if (zone_bbox[0] <= cx <= zone_bbox[2] and
                                zone_bbox[1] <= cy <= zone_bbox[3]):
                            zone_chars.append(char)
                            zone_text += char.get("text", "")
                except Exception as exc:
                    logger.debug(f"operation: suppressed {exc}")

                zones.append(Zone(
                    type=region.category,
                    bbox=zone_bbox,
                    text=zone_text.strip(),
                    chars=zone_chars,
                    confidence=region.confidence,
                ))

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

    def _extract_page_styles(self, fitz_page) -> Dict[str, Style]:
        """Extract visual features of text within the page."""
        style_map: Dict[str, Style] = {}
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
        bbox: Tuple[float, float, float, float],
        style_map: Dict[str, Style],
    ) -> Tuple[TextSpan, ...]:
        """Build TextSpan sequence from text + coordinates + style_map."""
        if not text:
            return ()

        key = text[:20]
        style = style_map.get(key, Style())
        return (TextSpan(text=text, bbox=bbox, style=style),)

    def _infer_heading_level(
        self,
        text: str,
        style_map: Dict[str, Style],
    ) -> Optional[int]:
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

    def _collect_kv_entities(
        self, pages: List[PageLayout]
    ) -> Dict[str, str]:
        """
        Collect entities from extracted key_value blocks.
        
        Note: Full entity extraction (regex/bank name/account name/account number etc.)
        has been moved to middlewares.entity_extractor.EntityExtractor middleware.
        This method only performs simple KV block aggregation to ensure
        BaseResult.metadata has basic entity information.
        """
        entities: Dict[str, str] = {}
        for page in pages:
            for block in page.blocks:
                if block.block_type == "key_value" and isinstance(block.raw_content, dict):
                    entities.update(block.raw_content)
        return entities

    def _merge_cross_page_tables(self, pages: List[PageLayout]) -> List[PageLayout]:
        """Cross-page table merge -- delegated to table_merger module."""
        from ..table.merger import merge_cross_page_tables
        return merge_cross_page_tables(pages)

    def _post_process_tables(self, pages: List[PageLayout]) -> List[PageLayout]:
        """Table post-processing -- header detection + preamble KV extraction + data row cleaning.

        For each table Block, execute the following steps:
          1. Call post_process_table:
             - VOCAB_BY_CATEGORY scans first 10 rows, finds the row with highest vocab_score as header
             - When header_row_idx > 0: call _extract_preamble_kv for summary rows before header
               Results stored as preamble KV (e.g. total amount/count/start date)
             - Call _strip_preamble on data_rows to strip mixed summary rows and duplicate headers
             - Per-row cleanup: filter junk rows, append non-data rows to previous, align column count
          2. Call get_and_clear_preamble_kv() to retrieve KV cache:
             - If non-empty, create key_value Block inserted before the table Block
             - Block ID is "{block_id}_kv", reading_order same as table

        Side effect: if table Block has < 2 rows after processing, retain original Block unchanged.
        """
        new_pages = []
        for page in pages:
            new_blocks = []
            for block in page.blocks:
                if block.block_type == "table" and isinstance(block.raw_content, list):
                    # Fix 9: Skip entirely empty tables
                    if not any(
                        (cell or "").strip()
                        for row in block.raw_content
                        for cell in row
                    ):
                        logger.debug("[DocMirror] skipped empty table")
                        continue
                    try:
                        processed, preamble_kv = post_process_table(block.raw_content)

                        # Table structure fix (line merging/cell cleanup/sticky splitting/column alignment)
                        if processed and len(processed) >= 2:
                            from ..table.table_structure_fix import fix_table_structure
                            processed = fix_table_structure(processed)

                        # Preamble KV obtained directly from return value, no global state
                        if preamble_kv:
                            kv_block = Block(
                                block_id=f"{block.block_id}_kv",
                                block_type="key_value",
                                bbox=block.bbox,
                                reading_order=block.reading_order,
                                page=block.page,
                                raw_content=preamble_kv,
                            )
                            new_blocks.append(kv_block)
                        if processed and len(processed) >= 2:
                            new_block = Block(
                                block_id=block.block_id,
                                block_type="table",
                                bbox=block.bbox,
                                reading_order=block.reading_order,
                                page=block.page,
                                raw_content=processed,
                            )
                            new_blocks.append(new_block)
                        else:
                            new_blocks.append(block)
                    except Exception as e:
                        logger.debug(f"[DocMirror] post_process error: {e}")
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)

            new_page = PageLayout(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                blocks=tuple(new_blocks),
                semantic_zones=page.semantic_zones,
                is_scanned=page.is_scanned,
            )
            new_pages.append(new_page)

        return new_pages
