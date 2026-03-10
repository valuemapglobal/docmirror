"""
预分析器 (PreAnalyzer)
========================

对应人类认知阶段 2: "拿到文档先快速翻一下"。

在版面分析之前，用最低成本 (~10ms) 获得文档的"第一印象"：

    1. 文本层检查:   电子件 vs 扫描件
    2. 页数统计:     规模判断
    3. 第一页内容类型: 表格主导 vs 文本主导 vs 混合
    4. 质量评估:     文字清晰度 / 版面规则性
    5. 复杂度评估:   决定投入多少计算资源
    6. 策略推荐:     fast / standard / deep

输出 ``PreAnalysisResult`` (frozen) 存入 BaseResult.metadata，
供 Orchestrator 和中间件动态调参。
"""

from __future__ import annotations

import logging
import dataclasses
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PreAnalysisResult:
    """
    文档预分析结果 (不可变)。

    由 PreAnalyzer 在提取前生成，为后续所有阶段提供决策依据。
    """
    # 基础属性
    has_text_layer: bool = True
    num_pages: int = 0

    # 内容类型
    content_type: str = "unknown"
    # "table_dominant"  — 表格占主导 (银行流水、发票)
    # "text_dominant"   — 纯文本 (合同、报告)
    # "mixed"           — 混合版面
    # "scanned"         — 扫描件 (无/极少文本层)

    # 质量评估 (0.0-1.0)
    quality_score: float = 1.0
    # 1.0 = 清晰电子件
    # 0.7 = 轻微模糊/有水印
    # 0.3 = 低质扫描件

    # 复杂度
    complexity_level: str = "medium"
    # "simple"  — 单表格、少于 10 页
    # "medium"  — 多内容块、10-50 页
    # "complex" — 复杂混合版面、>50 页、多表格

    # 策略推荐
    recommended_strategy: str = "standard"
    # "fast"     — 跳过非必要中间件，简化提取
    # "standard" — 标准管线
    # "deep"     — 启用高 DPI OCR、更多验证维度

    # 首页预览
    first_page_preview: str = ""

    # 版面统计
    estimated_table_pages: int = 0
    estimated_scanned_pages: int = 0

    # 版面一致性 (同构文档标志)
    layout_homogeneous: bool = False
    # True  = 采样页的版面结构高度一致 (如银行流水每页都是同一表格)
    # False = 各页版面差异较大 (如混合报告)

    # 检测到的文档语言
    detected_language: str = "unknown"
    # 示例: "zh", "en", "ja", "ko", "mixed", "unknown"

    # 策略参数 (供 Orchestrator 直接使用, 替代硬编码 if/else)
    strategy_params: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict — 存入 BaseResult.metadata。"""
        return dataclasses.asdict(self)


class PreAnalyzer:
    """
    预分析引擎。

    设计原则:
        - 极低成本: 仅读 1-3 页，~10ms 完成
        - 不修改文档: 只读操作
        - 全面覆盖: 5 个维度的"第一印象"

    使用方式::

        pre = PreAnalyzer()
        result = pre.analyze(fitz_doc)
        # result.recommended_strategy → "fast" / "standard" / "deep"
    """

    def analyze(self, fitz_doc) -> PreAnalysisResult:
        """
        分析文档，生成 PreAnalysisResult。

        Args:
            fitz_doc: 已打开的 PyMuPDF 文档对象。
        """
        num_pages = len(fitz_doc)
        if num_pages == 0:
            return PreAnalysisResult(num_pages=0, content_type="unknown")

        # ── Step 1: 文本层检查 ──
        has_text, text_density = self._check_text_layer(fitz_doc)

        # ── Step 2: 第一页分析 ──
        first_page = fitz_doc[0]
        first_page_stats = self._analyze_page(first_page, 0)
        first_page_preview = first_page.get_text()[:200].strip()

        # ── Step 3: 快速全文档扫描 (仅统计, 不深入) ──
        table_pages = 0
        scanned_pages = 0
        page_fingerprints: List[tuple] = []  # 版面一致性用

        # 最多扫描前 10 页 + 最后 2 页 (典型银行流水)
        sample_pages = list(range(min(10, num_pages)))
        if num_pages > 10:
            sample_pages.extend([num_pages - 2, num_pages - 1])

        for idx in sample_pages:
            if idx >= num_pages:
                continue
            stats = self._analyze_page(fitz_doc[idx], idx)
            if stats["has_table"]:
                table_pages += 1
            if stats["is_scanned"]:
                scanned_pages += 1
            # 收集结构指纹 (版面一致性检测)
            page_fingerprints.append(
                (stats["x_column_count"], stats["has_table"], stats["text_blocks"])
            )

        # ── Step 3.5: 版面一致性判定 ──
        layout_homogeneous = self._check_layout_homogeneity(page_fingerprints)

        # ── Step 4: 内容类型判定 ──
        content_type = self._determine_content_type(
            has_text, text_density, first_page_stats,
            table_pages, scanned_pages, len(sample_pages),
        )

        # ── Step 5: 质量评估 ──
        quality_score = self._assess_quality(
            has_text, text_density, first_page_stats, scanned_pages,
        )

        # ── Step 6: 复杂度评估 ──
        complexity_level = self._assess_complexity(
            num_pages, content_type, first_page_stats,
        )

        # ── Step 7: 策略推荐 ──
        strategy = self._recommend_strategy(
            complexity_level, quality_score, content_type,
        )

        # ── Step 8: 策略参数化 ──
        strategy_params = self._build_strategy_params(
            strategy, content_type, layout_homogeneous,
            has_text, quality_score,
        )

        # ── Step 9: 语言检测 ──
        detected_language = self._detect_language(first_page_preview)

        result = PreAnalysisResult(
            has_text_layer=has_text,
            num_pages=num_pages,
            content_type=content_type,
            quality_score=round(quality_score, 2),
            complexity_level=complexity_level,
            recommended_strategy=strategy,
            first_page_preview=first_page_preview,
            estimated_table_pages=table_pages,
            estimated_scanned_pages=scanned_pages,
            layout_homogeneous=layout_homogeneous,
            detected_language=detected_language,
            strategy_params=strategy_params,
        )

        logger.info(
            f"[DocMirror] PreAnalyzer ▶ "
            f"pages={num_pages} | type={content_type} | "
            f"quality={quality_score:.2f} | complexity={complexity_level} | "
            f"strategy={strategy} | homogeneous={layout_homogeneous} | "
            f"lang={detected_language}"
        )

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # 内部分析方法
    # ═══════════════════════════════════════════════════════════════════════════

    def _check_text_layer(self, fitz_doc) -> Tuple[bool, float]:
        """
        检查文本层存在性和密度。

        Returns:
            (has_text, density): density = 平均每页字符数 / 1000
        """
        total_chars = 0
        check_pages = min(3, len(fitz_doc))

        for i in range(check_pages):
            text = fitz_doc[i].get_text()
            total_chars += len(text.strip())

        avg_chars = total_chars / check_pages if check_pages else 0
        has_text = avg_chars > 30  # 少于 30 字符 → 基本无文本
        density = avg_chars / 1000

        return has_text, density

    def _analyze_page(self, page, page_idx: int) -> Dict[str, Any]:
        """
        单页快速分析 (~5ms)。

        提取:
            - 文本块数、图片块数
            - 是否有表格 (find_tables)
            - 是否为扫描件 (文字极少 + 大图)
            - 文本区域数、图像覆盖率
        """
        rect = page.rect
        page_area = rect.width * rect.height
        text_dict = page.get_text("dict", flags=0)
        text_blocks = [b for b in text_dict.get("blocks", []) if b.get("type") == 0]
        image_blocks = [b for b in text_dict.get("blocks", []) if b.get("type") == 1]

        # 字符总数
        total_chars = sum(
            len(span.get("text", ""))
            for b in text_blocks
            for line in b.get("lines", [])
            for span in line.get("spans", [])
        )

        # 图像覆盖率
        image_area = 0
        for b in image_blocks:
            bx = b["bbox"]
            image_area += max(0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
        image_coverage = image_area / page_area if page_area else 0

        # 表格检测
        has_table = False
        table_count = 0
        try:
            tables = page.find_tables()
            table_count = len(tables.tables)
            has_table = table_count > 0
        except Exception:
            pass

        # 扫描件判断
        is_scanned = (total_chars < 50 and image_coverage > 0.4)

        # 文本密度 (列分布, 用于判断是否是表格)
        x_positions = set()
        for b in text_blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        x_positions.add(round(span["bbox"][0] / 20) * 20)

        return {
            "text_blocks": len(text_blocks),
            "image_blocks": len(image_blocks),
            "total_chars": total_chars,
            "image_coverage": image_coverage,
            "has_table": has_table,
            "table_count": table_count,
            "is_scanned": is_scanned,
            "x_column_count": len(x_positions),  # 列分布广 → 表格特征
        }

    def _determine_content_type(
        self,
        has_text: bool,
        text_density: float,
        first_page: Dict,
        table_pages: int,
        scanned_pages: int,
        sample_size: int,
    ) -> str:
        """判定内容类型。"""
        if not has_text or (scanned_pages > sample_size * 0.5):
            return "scanned"

        table_ratio = table_pages / sample_size if sample_size else 0

        if table_ratio >= 0.6:
            return "table_dominant"
        elif table_ratio <= 0.2 and first_page["x_column_count"] <= 3:
            return "text_dominant"
        else:
            return "mixed"

    def _assess_quality(
        self,
        has_text: bool,
        text_density: float,
        first_page: Dict,
        scanned_pages: int,
    ) -> float:
        """质量评估 (0-1)。"""
        if not has_text:
            return 0.3  # 扫描件基线

        score = 1.0

        # 文本密度低 → 质量下降
        if text_density < 0.1:
            score -= 0.3
        elif text_density < 0.5:
            score -= 0.1

        # 图像覆盖率高但有文本 → 可能有水印
        if first_page["image_coverage"] > 0.3 and has_text:
            score -= 0.1

        # 有扫描页 → 质量下降
        if scanned_pages > 0:
            score -= 0.2

        return max(0.1, min(1.0, score))

    def _assess_complexity(
        self,
        num_pages: int,
        content_type: str,
        first_page: Dict,
    ) -> str:
        """复杂度评估。"""
        # 页数维度
        if num_pages > 50:
            return "complex"

        # 混合版面
        if content_type == "mixed":
            return "complex" if num_pages > 20 else "medium"

        # 扫描件总是更复杂
        if content_type == "scanned":
            return "complex" if num_pages > 10 else "medium"

        # 简单的单表格
        if content_type == "table_dominant" and num_pages <= 10:
            return "simple"

        if content_type == "text_dominant" and num_pages <= 5:
            return "simple"

        return "medium"

    def _recommend_strategy(
        self,
        complexity: str,
        quality: float,
        content_type: str,
    ) -> str:
        """策略推荐。"""
        if complexity == "simple" and quality >= 0.8:
            return "fast"

        if complexity == "complex" or quality < 0.5:
            return "deep"

        return "standard"

    # ═══════════════════════════════════════════════════════════════════════════
    # 版面一致性 + 策略参数化
    # ═══════════════════════════════════════════════════════════════════════════

    def _check_layout_homogeneity(
        self, fingerprints: List[tuple],
    ) -> bool:
        """判定采样页间的版面结构是否一致。

        一致性定义: ≥80% 的采样页具有相同的 (x_column_count, has_table) 特征。
        样本 < 3 页时无法判定，返回 False。
        """
        if len(fingerprints) < 3:
            return False

        # 只看 (x_column_count, has_table) 两个关键维度
        patterns = [(fp[0], fp[1]) for fp in fingerprints]
        most_common, count = Counter(patterns).most_common(1)[0]
        ratio = count / len(patterns)
        return ratio >= 0.8

    def _build_strategy_params(
        self,
        strategy: str,
        content_type: str,
        layout_homogeneous: bool,
        has_text: bool,
        quality_score: float,
    ) -> dict:
        """根据预分析结果生成具体的策略参数字典。

        这些参数供 Orchestrator 和中间件直接消费,
        避免下游对 recommended_strategy 字符串做硬编码 if/else。
        """
        params: Dict[str, Any] = {}

        # 水印过滤: 高质量纯文本文档可以跳过
        params["skip_watermark_filter"] = (
            content_type == "text_dominant" and quality_score >= 0.9
        )

        # Docling: standard/deep 且有文本层时考虑启用
        params["enable_docling"] = (
            strategy in ("standard", "deep") and has_text
        )

        # 表格提取方法优先级
        if content_type == "table_dominant":
            params["table_method_priority"] = [
                "lines", "hline_columns", "rect_columns", "text",
            ]
        else:
            params["table_method_priority"] = ["lines", "text"]

        # 首页结构复用: 同构文档 + 表格主导
        params["reuse_first_page_structure"] = (
            layout_homogeneous and content_type == "table_dominant"
        )

        # LLM 启用建议
        params["enable_llm"] = (strategy == "deep")

        return params

    def _detect_language(self, text_sample: str) -> str:
        """检测文档语言。

        使用 fast-langdetect (如果安装) 或基于字符统计的回退方法。

        Args:
            text_sample: 文本样本 (通常为首页前 200 字符)。

        Returns:
            语言代码: "zh", "en", "ja", "ko", "mixed", 或 "unknown"
        """
        if not text_sample or len(text_sample.strip()) < 10:
            return "unknown"

        # 尝试使用 fast-langdetect
        try:
            from fast_langdetect import detect  # type: ignore
            result = detect(text_sample)
            if isinstance(result, dict):
                return result.get("lang", "unknown")
            elif isinstance(result, str):
                return result
        except ImportError:
            pass  # fast-langdetect 未安装，使用回退方法
        except Exception:
            pass

        # 回退: 基于字符统计的简易检测
        cjk_count = 0
        latin_count = 0
        for ch in text_sample:
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
                    or 0x20000 <= cp <= 0x2A6DF):
                cjk_count += 1
            elif (0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A):
                latin_count += 1

        total = cjk_count + latin_count
        if total == 0:
            return "unknown"

        cjk_ratio = cjk_count / total
        if cjk_ratio > 0.7:
            return "zh"  # 简化: CJK 字符为主视为中文
        elif cjk_ratio < 0.1:
            return "en"
        else:
            return "mixed"
