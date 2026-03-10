import logging
import re
from typing import List, Tuple, Any, Optional
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
    HAS_RAPIDOCR = True
except ImportError:
    HAS_RAPIDOCR = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OCR 后处理: 将 RapidOCR 的 "行级" 产出拆分为 "词级" 产出
# ─────────────────────────────────────────────────────────────────────────────
# RapidOCR 输出的是一整行的文本块 (如 "1720240224"), 而非独立的词。
# 为了与 PyMuPDF 的 get_text("words") 对齐, 需要将合并文本拆分为
# 独立的词块, 每个词块具有近似的 bounding box。
#
# 拆分规则 (按优先级):
#   1. 序号+日期合并体: "1720240224" → "17", "20240224"
#   2. 中文/数字边界:   "消费-实物商品" → 保留原样 (对列归属无影响)
#   3. 纯数字金额:       "10,665.66" → 保留原样

# 识别 "序号+YYYYMMDD" 的合并体
_SEQ_DATE_RE = re.compile(r'^(\d{1,4})(20\d{6})$')


def _split_ocr_block(
    x0: float, y0: float, x1: float, y1: float, text: str
) -> List[Tuple[float, float, float, float, str]]:
    """
    尝试将合并的 OCR 文本块拆分为多个子词块。
    对无法/不必拆分的文本, 直接返回原块。
    
    Returns:
        List of (x0, y0, x1, y1, sub_text)
    """
    text = text.strip()
    if not text:
        return []

    total_width = x1 - x0
    
    # Rule 1: 序号+日期 (e.g. "1720240224" → "17" + "20240224")
    m = _SEQ_DATE_RE.match(text)
    if m:
        seq_part, date_part = m.group(1), m.group(2)
        ratio = len(seq_part) / len(text)
        split_x = x0 + total_width * ratio
        return [
            (x0, y0, split_x, y1, seq_part),
            (split_x, y0, x1, y1, date_part),
        ]

    # Default: keep as-is
    return [(x0, y0, x1, y1, text)]


class RapidOCREngine:
    """
    Singleton wrapper for RapidOCR ONNX Runtime engine.
    Extracts text and bounding boxes from images and normalizes the format
    to match PyMuPDF's `get_text("words")` tuple structure.
    
    关键设计:
      1. 调用 RapidOCR 获取行级文本+polygon 坐标。
      2. 通过 _split_ocr_block 后处理, 将合并文本拆分为词级单元。
      3. 输出格式与 PyMuPDF 完全一致: (x0, y0, x1, y1, text, block, line, word)。
    """
    _instance = None
    _engine = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not HAS_RAPIDOCR:
            logger.warning("rapidocr_onnxruntime is not installed. OCR will fail.")
            return

        if self._engine is None:
            # 自动检测 GPU 可用性
            use_cuda = False
            try:
                import onnxruntime
                providers = onnxruntime.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    use_cuda = True
                    logger.info("GPU (CUDA) detected, enabling GPU acceleration for OCR")
            except Exception:
                pass

            logger.info(f"Initializing RapidOCR ONNX model (GPU={use_cuda})...")
            if use_cuda:
                self._engine = RapidOCR(use_cuda=True)
            else:
                self._engine = RapidOCR()
            logger.info("RapidOCR model loaded.")

    def detect_image_words(
        self, img: np.ndarray
    ) -> List[Tuple[float, float, float, float, str, int, int, int]]:
        """
        Detect words in an image and return them in PyMuPDF 'words' format:
        (x0, y0, x1, y1, text, block_no, line_no, word_no)
        
        Args:
            img: OpenCV/numpy image array (BGR format from cv2)
            
        Returns:
            List of tuples matching PyMuPDF get_text('words') output format.
        """
        if not self._engine:
            raise RuntimeError(
                "RapidOCR engine not available. "
                "Please install rapidocr_onnxruntime."
            )

        # rapidocr returns: (result, elapse)
        # result is a list of [box, text, confidence]
        # box is [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        result, _ = self._engine(img)

        words = []
        if not result:
            return words

        word_idx = 0
        for i, res in enumerate(result):
            box, text, confidence = res
            
            # Extract axis-aligned bounding rect from 4-point polygon
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            bx0, bx1 = min(x_coords), max(x_coords)
            by0, by1 = min(y_coords), max(y_coords)
            
            # Split merged OCR blocks into individual word units
            sub_words = _split_ocr_block(bx0, by0, bx1, by1, text)
            
            for sx0, sy0, sx1, sy1, sub_text in sub_words:
                words.append((
                    float(sx0), float(sy0), float(sx1), float(sy1),
                    str(sub_text), 0, i, word_idx
                ))
                word_idx += 1

        return words


# Singleton instance for easy import
ocr_engine = RapidOCREngine()

def get_ocr_engine() -> RapidOCREngine:
    return ocr_engine
