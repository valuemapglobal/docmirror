"""
布局检测模型封装 (Layout Detection Model)
==========================================

封装 RapidLayout (ONNX) 模型，用于模型级布局检测。
当模型不可用时，自动回退到规则方法。

来源: https://github.com/RapidAI/RapidLayout
内置模型: DocLayout-YOLO, PP-Layout-CDLA, PP-DocLayoutV3 等 12 种

使用方式::

    detector = LayoutDetector()           # 默认 DOCLAYOUT_DOCSTRUCTBENCH
    detector = LayoutDetector("cdla")     # 中文文档版面
    regions = detector.detect(page_image)

需要:
    - rapid-layout (pip install rapid-layout)
    - onnxruntime (已有, RapidOCR 依赖)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from rapid_layout import RapidLayout, RapidLayoutInput, ModelType
    HAS_RAPID_LAYOUT = True
except ImportError:
    HAS_RAPID_LAYOUT = False

# RapidLayout class_name → MultiModal Zone 类型
_CATEGORY_MAP = {
    # DocLayout-YOLO (DOCLAYOUT_DOCSTRUCTBENCH) 10 类
    "title": "title",
    "plain text": "text",
    "abandon": "abandon",
    "figure": "data_table",           # figure 同样作为独立区域
    "figure_caption": "text",
    "table": "data_table",
    "table_caption": "text",
    "table_footnote": "text",
    "isolate_formula": "formula",
    "formula_caption": "text",
    # PP-Layout-CDLA 额外类别
    "text": "text",
    "header": "title",
    "footer": "footer",
    "reference": "text",
    "equation": "formula",
}

# 模型类型别名映射 (字符串 → ModelType)
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
    """模型检测到的区域。"""
    category: str           # 映射后的 MultiModal Zone 类型
    bbox: Tuple[float, float, float, float]   # (x0, y0, x1, y1)
    confidence: float       # 置信度 0-1
    raw_category_id: str    # 原始模型类别名


class LayoutDetector:
    """RapidLayout 布局检测器。

    懒加载模型，首次 detect() 时初始化。
    支持 12 种内置模型，通过 model_type 选择。
    """

    def __init__(self, model_type: str = "doclayout_docstructbench"):
        """
        Args:
            model_type: 模型类型名称，支持简写别名。
                常用值: "doclayout", "cdla", "layoutv3", "paper", "general6"
        """
        self._model_type_str = model_type
        self._engine = None
        self._available = HAS_RAPID_LAYOUT

    def _ensure_engine(self) -> bool:
        """懒加载引擎。"""
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
        """检测页面布局区域。

        Args:
            page_image: 页面图像 (numpy ndarray, HxWx3 RGB/BGR, 或路径)。
            confidence_threshold: 置信度阈值。

        Returns:
            DetectedRegion 列表，按 y 坐标排序。
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

            if category == "abandon":
                continue

            bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

            # 过滤面积过小的检测结果
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < 100:
                continue

            regions.append(DetectedRegion(
                category=category,
                bbox=bbox,
                confidence=float(score),
                raw_category_id=cls_name,
            ))

        # 按 y 坐标排序 (阅读顺序)
        regions.sort(key=lambda r: r.bbox[1])

        logger.debug(f"[LayoutDetector] Detected {len(regions)} regions (threshold={confidence_threshold})")
        return regions

    @property
    def is_available(self) -> bool:
        """模型是否可用。"""
        return self._available
