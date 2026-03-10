"""
核心提取层 (Core Extractor)
============================

本模块是 MultiModal 的「提取引擎」，负责将原始 PDF 文件解析为
不可变的 ``BaseResult`` 数据结构。

=== 提取流程 ===

  Step 1: 预处理 + 预检
          PyMuPDF 快速检查文本层 → 标记电子版/扫描件

  Step 2: 页面迭代 + 版面分析
          对每页调用 segment_page_into_zones 划分语义 Zone

  Step 3: 表格提取
          对每个 data_table Zone 调用 extract_tables_layered (4 层递进)。
          提取结果以 table Block 形式收入 PageLayout。
          扫描件则走 OCR 回退路径 (analyze_scanned_page)。

  Step 4: 跨页合并 (_merge_cross_page_tables)
          检测多页表格并将它们合并为单一 Block。
          续表页的表头匹配策略:
            - 首行是表头 + 与上一页表头匹配 → 直接合并数据行
            - 首行不是表头 (即汇总行开头的续表页) → _strip_preamble 后合并

  Step 5: 表格后处理 (_post_process_tables)
          对每个 table Block 调用 post_process_table:
            - VOCAB_BY_CATEGORY 扫描找到最佳表头行
            - header 前的汇总行 → _extract_preamble_kv 提取为 KV
            - header 后的汇总/重复表头 → _strip_preamble 剥离
            - 数据行清洗 (_is_junk_row / _is_data_row / 单元格对齐)
          提取完成后调用 get_and_clear_preamble_kv():
            - 若有 KV, 创建 key_value Block 插入 table 前面

  Step 6: 组装 BaseResult
          将每页的 Block 列表组装为 frozen BaseResult

=== 输出结构 ===

  每页 blocks 列表 (reading_order 顺序):
    [title]      → 页面标题
    [title]      → 账户信息行
    [key_value]  → preamble 汇总 KV (如有。示例: 汇出总金额/总笔数/开始时间)
    [table]      → 交易明细表 (header + 数据行)
"""

from __future__ import annotations

import logging
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
from ..layout.layout_analysis import (
    analyze_document_layout,
    segment_page_into_zones,
    Zone,
    extract_tables_layered,
    get_last_layer_timings,
    analyze_scanned_page,
    filter_watermark_page,
    _dedup_overlapping_chars,
    _reconstruct_rows_from_chars,
    _extract_summary_entities,
    post_process_table,
    _strip_preamble,
    preprocess_pdf,
    headers_match,
    _is_header_row,
)
from ..exceptions import ExtractionError

logger = logging.getLogger(__name__)


class CoreExtractor:
    """
    核心提取器 — 从 PDF 生成不可变 BaseResult。

    使用方式::

        extractor = CoreExtractor()
        result = await extractor.extract(Path("sample.pdf"))
        # result 是 frozen BaseResult，不可修改

    所有底层函数均来自 MultiModal.core 子模块 (自包含)。
    """

    def __init__(self, seal_detector_fn=None, layout_model_path: Optional[str] = None,
                 max_page_concurrency: int = 1,
                 formula_model_path: Optional[str] = None,
                 model_render_dpi: int = 200):
        """
        Args:
            seal_detector_fn: 可选的印章检测回调函数。
                签名: (fitz_doc) -> Optional[Dict[str, Any]]
                为 None 时跳过印章检测。
            layout_model_path: 可选的 DocLayout-YOLO ONNX 模型路径。
                设置为 "auto" 时从 HuggingFace 自动下载。
                为 None 时使用规则回退。
            max_page_concurrency: 页面级并发度。
                pdfplumber/PyMuPDF 共享文档对象, 当前默认值为 1 (顺序)。
                设为 >1 时使用 ThreadPoolExecutor 并行提取。
            formula_model_path: 可选的公式识别 ONNX 模型路径 (UniMERNet)。
                为 None 时回退到 rapid_latex_ocr → 空字符串。
            model_render_dpi: DocLayout-YOLO 模型推理时的页面渲染 DPI。
                默认 200，较高值提升布局检测精度但增加推理耗时。
        """
        self._seal_detector_fn = seal_detector_fn
        self._layout_detector = None
        self._max_page_concurrency = max_page_concurrency
        self._model_render_dpi = model_render_dpi

        # 公式识别引擎 (策略模式: UniMERNet ONNX > rapid_latex_ocr > empty)
        from ..ocr.formula_engine import FormulaEngine
        self._formula_engine = FormulaEngine(model_path=formula_model_path)

        if layout_model_path:
            try:
                from ..layout.layout_model import LayoutDetector
                # layout_model_path 现在作为模型类型名
                # "auto" → 默认 doclayout_docstructbench
                model_type = "doclayout_docstructbench" if layout_model_path == "auto" else layout_model_path
                self._layout_detector = LayoutDetector(model_type=model_type)
                logger.info("[DocMirror] Layout model enabled (RapidLayout)")
            except Exception as e:
                logger.warning(f"[DocMirror] Layout model init failed, falling back to rules: {e}")

    # 支持的图片格式
    _IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"})

    @staticmethod
    def _image_to_virtual_pdf(image_path: Path) -> "fitz.Document":
        """将图片转换为虚拟单页 PDF, 大图预缩放到 max 2048px。"""
        import fitz

        # 读取图片
        img_doc = fitz.open(str(image_path))
        if len(img_doc) == 0:
            raise ValueError(f"Cannot open image: {image_path}")

        # 大图预缩放 (防止内存爆炸)
        page = img_doc[0]
        w, h = page.rect.width, page.rect.height
        max_dim = 2048
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            logger.info(f"[DocMirror] 图片预缩放: {w:.0f}x{h:.0f} → {w*scale:.0f}x{h*scale:.0f}")
            # 使用 pixmap 缩放
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)
            img_doc.close()
            # 从缩放后的 pixmap 重建
            new_doc = fitz.open()
            new_page = new_doc.new_page(width=pix.width, height=pix.height)
            new_page.insert_image(new_page.rect, pixmap=pix)
            return new_doc

        # 正常尺寸: 直接转 PDF
        pdf_bytes = img_doc.convert_to_pdf()
        img_doc.close()
        return fitz.open("pdf", pdf_bytes)

    async def extract(self, file_path: Path) -> BaseResult:
        """
        主入口: 从 PDF 或图片提取 BaseResult。

        支持格式:
            - PDF 文件 (.pdf)
            - 图片文件 (.jpg, .png, .tiff, .bmp, .webp)

        图片文件将自动转换为虚拟 PDF, 走完整解析管线
        (版面分析 → 表格提取 → 公式识别 → 阅读顺序)。

        Args:
            file_path: PDF 或图片文件路径。

        Returns:
            BaseResult: 不可变的提取结果。
        """
        t0 = time.time()
        file_path = Path(file_path)
        doc_id = str(uuid.uuid4())
        is_image_input = file_path.suffix.lower() in self._IMAGE_SUFFIXES

        logger.info(f"[DocMirror] ▶ extract | file={file_path.name} | image={is_image_input}")
        
        # === [主干解析逻辑 (Heuristics)] ===
        fitz_doc = None
        try:
            # ═══ Step 0: 图片 → 虚拟 PDF ═══
            if is_image_input:
                fitz_doc = self._image_to_virtual_pdf(file_path)
                has_text = False  # 图片没有文字层
                logger.info("[DocMirror] 图片输入 → 虚拟 PDF, 标记为扫描件")
            else:
                # ═══ Step 1: 预处理 + 预检 ═══
                cleaned_path = preprocess_pdf(file_path)
                fitz_doc = FitzEngine.open(cleaned_path)
                has_text = FitzEngine.has_text_layer(fitz_doc)

            if not has_text:
                logger.info(f"[DocMirror] 文本层缺失，标记为扫描件")

            # ═══ Step 1.5: 预分析 (人类认知阶段2) ═══
            from .pre_analyzer import PreAnalyzer
            pre_analysis = PreAnalyzer().analyze(fitz_doc)

            # ═══ Step 2: 版面分析 ═══
            page_layouts_al = analyze_document_layout(fitz_doc)

            # 提取全文
            full_text_parts = []
            for page in fitz_doc:
                full_text_parts.append(page.get_text())
            full_text_raw = "\n\n".join(full_text_parts)
            full_text = unicodedata.normalize("NFKC", full_text_raw)

            # ═══ Step 3: 逐页提取 — Zone → Block ═══
            pages: List[PageLayout] = []
            ocr_text_parts: List[str] = []
            extraction_layer: str = "unknown"  # 最后一次 extract_tables_layered 使用的策略层
            extraction_confidence: float = 0.0  # 最后一次提取的置信度

            if has_text:
                # ── 数字 PDF: pdfplumber + fitz 联合提取 ──
                import pdfplumber
                plumber_path = cleaned_path if not is_image_input else file_path
                with pdfplumber.open(str(plumber_path)) as plumber_doc:
                    for page_idx, layout_al in enumerate(page_layouts_al):
                        page_plum = plumber_doc.pages[page_idx]
                        fitz_page = fitz_doc[page_idx]

                        page_layout, page_ocr_parts, extraction_layer, extraction_confidence = (
                            self._extract_page(
                                page_plum=page_plum,
                                fitz_page=fitz_page,
                                fitz_doc=fitz_doc,
                                page_idx=page_idx,
                                layout_al=layout_al,
                                cleaned_path=cleaned_path if not is_image_input else file_path,
                            )
                        )
                        pages.append(page_layout)
                        ocr_text_parts.extend(page_ocr_parts)
            else:
                # ── 扫描件/图片: OCR 提取 ──
                from ..ocr.fallback import analyze_scanned_page
                for page_idx in range(len(fitz_doc)):
                    fitz_page = fitz_doc[page_idx]
                    page_layout = self._extract_scanned_page(
                        fitz_page=fitz_page,
                        page_idx=page_idx,
                    )
                    pages.append(page_layout)
                    # 收集 OCR 文本
                    for blk in page_layout.blocks:
                        if blk.block_type == "text" and blk.raw_content:
                            ocr_text_parts.append(str(blk.raw_content))

            # OCR 文本合入全文
            if ocr_text_parts:
                full_text = full_text + "\n\n" + "\n\n".join(ocr_text_parts)

            # ═══ Step 4: 跨页合并 ═══
            pages = self._merge_cross_page_tables(pages)

            # ═══ Step 5: 表格后处理 ═══
            pages = self._post_process_tables(pages)

            # ═══ Step 6: 组装 BaseResult ═══
            elapsed = (time.time() - t0) * 1000

            total_blocks = sum(len(p.blocks) for p in pages)
            table_count = sum(
                1 for p in pages for b in p.blocks if b.block_type == "table"
            )

            # 提取实体 (正则 + KV blocks)
            extracted_entities = self._collect_kv_entities(pages)

            # ── 可选: 印章检测 (首页) — 通过依赖注入 ──
            seal_info = None
            if self._seal_detector_fn:
                try:
                    seal_info = self._seal_detector_fn(fitz_doc)
                except Exception as e:
                    logger.debug(f"[DocMirror] Seal detection skip: {e}")
            if seal_info:
                pass  # 并入下方 metadata

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
            }
            if seal_info:
                metadata["seal_info"] = seal_info

            # ── 优化1: 提取质量评估元信息 ──
            if table_count > 0:
                # 找主表 (最大的 table block)
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
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════════════
    # 扫描件 OCR 提取
    # ═══════════════════════════════════════════════════════════════════════════

    def _extract_scanned_page(
        self, *, fitz_page, page_idx: int,
    ) -> "PageLayout":
        """扫描件单页 OCR 提取: fitz page → OCR → Block 结构。

        analyze_scanned_page() 返回:
            {'table': [[cell, ...], ...], 'header_text': str, 'footer_text': str}
        """
        from ..ocr.fallback import analyze_scanned_page

        width = fitz_page.rect.width
        height = fitz_page.rect.height
        blocks: List[Block] = []
        reading_order = 0

        try:
            ocr_result = analyze_scanned_page(fitz_page, page_idx)

            if ocr_result and isinstance(ocr_result, dict):
                # ── Header 文本 ──
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

                # ── 表格数据 ──
                table_data = ocr_result.get("table", [])
                if table_data and len(table_data) >= 2:
                    blocks.append(Block(
                        block_id=f"blk_{page_idx}_{reading_order}",
                        block_type="table",
                        bbox=(0, height * 0.1, width, height * 0.9),
                        reading_order=reading_order,
                        page=page_idx + 1,
                        raw_content=table_data,  # List[List[str]] — 表格行列
                    ))
                    reading_order += 1
                elif table_data:
                    # 不足 2 行, 作为文本处理
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

                # ── Footer 文本 ──
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
                    f"[DocMirror] OCR page {page_idx}: "
                    f"header={bool(header_text)} table={len(table_data)}rows "
                    f"footer={bool(footer_text)}"
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
        """将 OCR words 按 Y 坐标分组为行。

        Args:
            words: OCR 结果中的 word 列表, 每个 word 有 bbox 和 text。
            tolerance_ratio: 行间距容差 (相对于平均字高)。

        Returns:
            按行分组的 word 列表。
        """
        if not words:
            return []

        # 计算每个 word 的中心 y
        items = []
        for w in words:
            bbox = w.get("bbox", (0, 0, 0, 0))
            if len(bbox) >= 4:
                cy = (bbox[1] + bbox[3]) / 2
                h = bbox[3] - bbox[1]
                items.append((cy, h, w))

        if not items:
            return []

        # 按 y 排序
        items.sort(key=lambda x: x[0])

        # 估计平均字高
        avg_h = sum(h for _, h, _ in items) / len(items) if items else 10
        tolerance = avg_h * tolerance_ratio

        # 分行
        lines: List[List[dict]] = []
        current_line: List[dict] = [items[0][2]]
        current_y = items[0][0]

        for cy, h, w in items[1:]:
            if abs(cy - current_y) <= tolerance:
                current_line.append(w)
            else:
                # 行内按 x 排序
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
    # 单页提取 (从 extract() 提取)
    # ═══════════════════════════════════════════════════════════════════════════

    def _extract_page(
        self, *, page_plum, fitz_page, fitz_doc, page_idx: int,
        layout_al, cleaned_path: Path,
    ) -> Tuple["PageLayout", List[str], str, float]:
        """单页提取: Zone → Block 转换。

        Args:
            page_plum: pdfplumber 页面对象。
            fitz_page: PyMuPDF 页面对象。
            fitz_doc: PyMuPDF 文档对象 (用于图片提取)。
            page_idx: 页面索引 (0-based)。
            layout_al: 版面分析结果。
            cleaned_path: 预处理后的 PDF 路径。

        Returns:
            (PageLayout, ocr_text_parts, extraction_layer, extraction_confidence)
        """
        width, height = FitzEngine.get_page_dimensions(fitz_page)

        # ── 字符级预处理 ──
        page_plum = filter_watermark_page(page_plum)
        page_plum = _dedup_overlapping_chars(page_plum)

        # ── 空间分区 ──
        if self._layout_detector:
            zones = self._model_segmentation(fitz_page, page_plum, page_idx)
        else:
            zones = segment_page_into_zones(page_plum, page_idx)

        # ── 提取视觉特征 (Style) ──
        style_map = self._extract_page_styles(fitz_page)

        # ── Zone → Block 转换 ──
        blocks: List[Block] = []
        reading_order = 0
        page_has_table = False
        extraction_layer = "unknown"
        extraction_confidence = 0.0
        ocr_text_parts: List[str] = []
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
                # K1: 优先从字符流提取 (数字 PDF 100% 精度, 零延迟)
                latex_str = None
                try:
                    from ..ocr.formula_chars import extract_formula_from_chars
                    if zone.chars:
                        latex_str = extract_formula_from_chars(zone.chars, zone.bbox)
                except Exception:
                    pass

                # K1 fallback: OCR 裁图识别
                if not latex_str:
                    formula_img = self._crop_zone_image(fitz_page, zone.bbox)
                    latex_str = self._recognize_formula(formula_img)

                block = Block(
                    block_id=block_id, block_type="formula",
                    bbox=zone.bbox, reading_order=reading_order,
                    page=page_idx + 1,
                    raw_content=latex_str or "",
                )
                blocks.append(block)
                reading_order += 1
                continue

            if zone.type == "data_table":
                page_has_table = True
                zone_tables_extracted = False

                # P3-2: 检测合并单元格
                merged_cells = []
                try:
                    from ..table.extraction.engine import detect_merged_cells
                    merged_cells = detect_merged_cells(page_plum, table_zone_bbox=zone.bbox)
                except Exception:
                    pass

                page_tables, extraction_layer, extraction_confidence = extract_tables_layered(
                    page_plum, table_zone_bbox=zone.bbox,
                )
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
                continue

            # unknown → text
            if zone.text:
                block = Block(
                    block_id=block_id, block_type="text",
                    bbox=zone.bbox, reading_order=reading_order,
                    page=page_idx + 1, raw_content=zone.text,
                    spans=self._build_spans(zone.text, zone.bbox, style_map),
                )
                blocks.append(block)
                semantic_zones["text_area"].append(block_id)
                reading_order += 1

        # ── 图像提取 ──
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
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"[DocMirror] image extraction skip: {e}")

        # Fallback: 版面分析有表格但 zone 没检测到
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

        # OCR 回退: 扫描件
        if layout_al.is_scanned and not page_has_table:
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
        return page_layout, ocr_text_parts, extraction_layer, extraction_confidence

    # ═══════════════════════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════════════════════

    # _run_seal_detection_if_enabled 已移除 — 印章检测通过 __init__(seal_detector_fn=...) 依赖注入

    def _model_segmentation(self, fitz_page, page_plum, page_idx: int) -> List:
        """模型级版面分析: 渲染页面图片 → DocLayout-YOLO 推理 → Zone 列表。

        失败时自动回退到规则方法。
        """
        try:
            import numpy as np

            # 渲染页面为图片 (可配置 DPI, 默认 200)
            render_dpi = self._model_render_dpi
            pixmap = fitz_page.get_pixmap(dpi=render_dpi)
            img_data = pixmap.samples
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            # RGBA → RGB
            if pixmap.n == 4:
                img = img[:, :, :3]

            # 模型推理
            regions = self._layout_detector.detect(img, confidence_threshold=0.4)

            if not regions:
                logger.debug(f"[DocMirror] model detected 0 regions on page {page_idx}, falling back")
                return segment_page_into_zones(page_plum, page_idx)

            # 坐标转换: 像素空间 → PDF 点空间 (72 DPI)
            scale = 72.0 / render_dpi
            zones = []
            for region in regions:
                rx0, ry0, rx1, ry1 = region.bbox
                zone_bbox = (rx0 * scale, ry0 * scale, rx1 * scale, ry1 * scale)

                # 获取区域内的文字
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
                except Exception:
                    pass

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
        """裁切页面中指定 bbox 区域的图片。"""
        try:
            import fitz as pymupdf
            rect = pymupdf.Rect(*bbox)
            clip = fitz_page.get_pixmap(clip=rect, dpi=300)
            return clip.tobytes("png")
        except Exception:
            return b""

    def _recognize_formula(self, image_bytes: bytes) -> str:
        """公式图片 → LaTeX (委托给 FormulaEngine)。

        FormulaEngine 内部按策略选择后端:
            UniMERNet ONNX > rapid_latex_ocr > 空字符串
        """
        return self._formula_engine.recognize(image_bytes)

    def _extract_page_styles(self, fitz_page) -> Dict[str, Style]:
        """提取页面内文本的视觉特征。"""
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
        """从文本 + 坐标 + style_map 构建 TextSpan 序列。"""
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
        """根据字体大小和加粗信息推断标题层级。

        策略:
            - 收集 style_map 中所有字体大小
            - 计算文档正文的中位数字体大小作为基线
            - 依据相对大小 + 加粗判定 h1/h2/h3:
              * font_size >= baseline * 1.6 且 bold → h1
              * font_size >= baseline * 1.2 且 bold → h2
              * bold only → h3
              * 无 bold 但字体显著偏大 → h2

        Returns:
            1, 2, 3 或 None（无法判定时）
        """
        if not text:
            return None

        key = text[:20]
        style = style_map.get(key)
        if not style:
            return None

        # 收集页面内所有字体大小作为上下文
        all_sizes = [s.font_size for s in style_map.values() if s.font_size > 0]
        if not all_sizes:
            return 3 if style.is_bold else None

        all_sizes.sort()
        # 中位数作为正文基线
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
            return 2  # 大字体但不加粗 → h2
        else:
            return None

    def _collect_kv_entities(
        self, pages: List[PageLayout]
    ) -> Dict[str, str]:
        """
        从已提取的 key_value blocks 收集实体。
        
        注: 完整的实体提取（正则/银行名/户名/账号等）已移至
        middlewares.entity_extractor.EntityExtractor 中间件。
        此方法仅做 KV block 的简单聚合，确保 BaseResult.metadata
        中有基础实体信息。
        """
        entities: Dict[str, str] = {}
        for page in pages:
            for block in page.blocks:
                if block.block_type == "key_value" and isinstance(block.raw_content, dict):
                    entities.update(block.raw_content)
        return entities

    def _merge_cross_page_tables(self, pages: List[PageLayout]) -> List[PageLayout]:
        """跨页表格合并 — 委托给 table_merger 模块。"""
        from ..table.merger import merge_cross_page_tables
        return merge_cross_page_tables(pages)

    def _post_process_tables(self, pages: List[PageLayout]) -> List[PageLayout]:
        """表格后处理 — 表头检测 + preamble KV 提取 + 数据行清洗。

        对每个 table Block 执行以下步骤:
          1. 调用 post_process_table:
             - VOCAB_BY_CATEGORY 扫描前10行, 找 vocab_score 最高的行作为表头
             - header_row_idx > 0 时: 对 header 之前的汇总行调用 _extract_preamble_kv
               将结果写入 _preamble_kv_store (如 汇出总金额/总笔数/开始时间)
             - 对 data_rows 调用 _strip_preamble, 剥离混入的汇总行和重复表头行
             - 逐行清洗: 过滤 junk 行, 非数据行追加到前一行, 对齐列数
          2. 调用 get_and_clear_preamble_kv() 取走 KV 缓存:
             - 若非空, 创建 key_value Block 并插入到 table Block 前面
             - Block ID 为 "{block_id}_kv", reading_order 与 table 相同

        副作用: 若 table Block 处理后行数 < 2, 保留原始 Block 不替换。
        """
        new_pages = []
        for page in pages:
            new_blocks = []
            for block in page.blocks:
                if block.block_type == "table" and isinstance(block.raw_content, list):
                    # Fix 9: 跳过全空表格
                    if not any(
                        (cell or "").strip()
                        for row in block.raw_content
                        for cell in row
                    ):
                        logger.debug("[DocMirror] skipped empty table")
                        continue
                    try:
                        processed, preamble_kv = post_process_table(block.raw_content)

                        # 表格结构修复 (行合并/单元格清理/粘连拆分/列对齐)
                        if processed and len(processed) >= 2:
                            from ..table.table_structure_fix import fix_table_structure
                            processed = fix_table_structure(processed)

                        # Preamble KV 直接从返回值获取, 无全局状态
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
