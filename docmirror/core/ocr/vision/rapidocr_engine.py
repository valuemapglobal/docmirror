from __future__ import annotations
import logging
import re
from typing import List, Tuple, Optional
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
    HAS_RAPIDOCR = True
except ImportError:
    HAS_RAPIDOCR = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OCR Post-processing: split RapidOCR "line-level" output into "word-level" units
# ─────────────────────────────────────────────────────────────────────────────
# RapidOCR outputs whole-line text blocks (e.g. "1720240224"), not
# individual words.  To align with PyMuPDF's get_text("words") format,
# merged text must be split into independent word units, each with an
# approximate bounding box.
#
# Split rules (by priority):
#   1. Sequence-number + date compound: "1720240224" → "17", "20240224"
#   2. CJK / digit boundary:           kept as-is (no effect on column assignment)
#   3. Pure numeric amount:             kept as-is ("10,665.66")

# Detect "sequence_number + YYYYMMDD" compounds
_SEQ_DATE_RE = re.compile(r'^(\d{1,4})(20\d{6})$')


def _split_ocr_block(
    x0: float, y0: float, x1: float, y1: float, text: str
) -> List[Tuple[float, float, float, float, str]]:
    """
    Attempt to split merged OCR text blocks into individual sub-word units.
    Text that cannot or need not be split is returned as-is.
    
    Returns:
        List of (x0, y0, x1, y1, sub_text)
    """
    text = text.strip()
    if not text:
        return []

    total_width = x1 - x0
    
    # Rule 1: sequence number + date (e.g. "1720240224" → "17" + "20240224")
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
    
    Key design:
      1. Call RapidOCR to obtain line-level text + polygon coordinates.
      2. Post-process via ``_split_ocr_block`` to split merged text into
         word-level units.
      3. Output format matches PyMuPDF exactly:
         ``(x0, y0, x1, y1, text, block_no, line_no, word_no)``.
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
            # Auto-detect optimal thread count
            import os
            try:
                cpu_count = os.cpu_count() or 4
                intra_threads = max(1, cpu_count - 1) # Use all but 1 core for ONNX matrix mult
            except Exception as exc:
                logger.debug(f"cpu_count detection: suppressed {exc}")
                intra_threads = 4
                
            use_cuda = False
            try:
                import onnxruntime
                providers = onnxruntime.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    use_cuda = True
                    logger.info("GPU (CUDA) detected, enabling GPU acceleration for OCR")
                elif "CoreMLExecutionProvider" in providers:
                    logger.info("Mac CoreML detected. Continuing with CPUExecutionProvider + Multi-threading (ONNX CoreML often underperforms on simple CV models).")
            except Exception as exc:
                logger.debug(f"operation: suppressed {exc}")

            logger.info(f"Initializing RapidOCR ONNX model (GPU={use_cuda}, Threads={intra_threads})...")
            
            # Phase 6 Part 1: Hardware-Aware Engine Initialization
            # The underlying ONNX Runtime models (DET, REC, CLS) run completely sequentially in pure CPU loops.
            # `intra_op_num_threads` is the most powerful flag for ONNX Runtime CPU scaling, parallelizing internal tensor algebra.
            # `inter_op_num_threads` is less useful for these specific linear CNN models.
            tuning_kwargs = {
                "intra_op_num_threads": intra_threads,
                "inter_op_num_threads": 2, 
                "rec_batch_num": 32, # Batch size for text recognition (default 6)
                "cls_batch_num": 32, # Batch size for text classifier
            }
            
            if use_cuda:
                self._engine = RapidOCR(use_cuda=True, **tuning_kwargs)
            else:
                self._engine = RapidOCR(**tuning_kwargs)
            logger.info("RapidOCR model loaded with Extreme CPU Tuning.")

    def _detect_only(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Runs only the Detection (DET) model on an image and returns un-scaled bounding boxes."""
        raw_h, raw_w = img.shape[:2]
        op_record = {}
        processed_img, ratio_h, ratio_w = self._engine.preprocess(img)
        op_record["preprocess"] = {"ratio_h": ratio_h, "ratio_w": ratio_w}
        
        processed_img, op_record = self._engine.maybe_add_letterbox(processed_img, op_record)
        dt_boxes, _ = self._engine.auto_text_det(processed_img)
        
        if dt_boxes is not None:
            dt_boxes = self._engine._get_origin_points(dt_boxes, op_record, raw_h, raw_w)
        return dt_boxes

    def _nms_boxes(self, boxes: List[np.ndarray], iou_threshold: float = 0.5) -> List[np.ndarray]:
        """
        Non-Maximum Suppression (NMS) for oriented bounding boxes.
        Merges multi-scale bounding boxes by keeping the larger bounding boxes
        that encapsulate smaller fragments, while preserving small independent boxes.
        """
        if not boxes:
            return []
            
        import cv2
        # Convert to standard OpenCV bounding rects: [x, y, w, h] for NMS calculation
        # Store original polygon mapped to its bounding rect
        rects = []
        scores = []
        box_polygon_map = []
        
        for idx, box in enumerate(boxes):
            if len(box) != 4:
                continue
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            x, y = int(min(x_coords)), int(min(y_coords))
            w, h = int(max(x_coords) - x), int(max(y_coords) - y)
            
            # Area as confidence score - favors larger encapsulating boxes
            area = float(w * h)
            if area > 10:  # Ignore microscopic noise
                rects.append([x, y, w, h])
                scores.append(area)
                box_polygon_map.append(box)
                
        if not rects:
            return []

        # cv2.dnn.NMSBoxes expects floats for scores
        indices = cv2.dnn.NMSBoxes(rects, scores, score_threshold=0.0, nms_threshold=iou_threshold)
        
        fused_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                fused_boxes.append(box_polygon_map[i])
                
        # Sort top-to-bottom, left-to-right
        fused_boxes.sort(key=lambda b: (min([p[1] for p in b]), min([p[0] for p in b])))
        # Ensure correct type format for RapidOCR downstream (list of np.ndarray)
        return [np.array(b, dtype=np.float32) for b in fused_boxes]

    def detect_multiscale_words(
        self, img: np.ndarray, scales: List[float] = [1.0, 2.0]
    ) -> List[Tuple[float, float, float, float, str, int, int, int]]:
        """
        Multi-Scale OCR with NMS (Non-Maximum Suppression).
        1. Runs the DET model multiple times at different scales.
        2. Merges and deduplicates bounding boxes via IoU NMS.
        3. Runs the computationally heavy REC model ONLY ONCE on the fused boxes.
        """
        if not self._engine:
            raise RuntimeError("RapidOCR engine not available.")

        import cv2
        all_scaled_boxes = []
        
        # 1. Multi-Pass Detection
        for scale in scales:
            if scale == 1.0:
                scaled_img = img
            else:
                scaled_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
                
            dt_boxes = self._detect_only(scaled_img)
            
            if dt_boxes is not None:
                # Map coordinates back to scale=1.0
                if scale != 1.0:
                    dt_boxes = dt_boxes / scale
                all_scaled_boxes.extend(dt_boxes)
                
        if not all_scaled_boxes:
            return []
            
        # 2. IoU NMS Merging
        fused_boxes = self._nms_boxes(all_scaled_boxes, iou_threshold=0.3)
        if not fused_boxes:
            return []
            
        # 3. Optimized REC pass (Run only once)
        # RapidOCR requires a specific format: list of numpy arrays
        # Then we create crop images based on these boxes
        fused_boxes_np = np.array(fused_boxes)
        img_crops = self._engine.get_crop_img_list(img, fused_boxes_np)
        
        rec_res, _ = self._engine.text_rec(img_crops, return_word_box=False)
        
        # Construct final output
        words = []
        if not rec_res:
            return words
            
        word_idx = 0
        for i, (box, rec_data) in enumerate(zip(fused_boxes, rec_res)):
            if not rec_data or not rec_data[0]:
                continue
                
            text, conf = rec_data
            
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            bx0, bx1 = min(x_coords), max(x_coords)
            by0, by1 = min(y_coords), max(y_coords)
            
            sub_words = _split_ocr_block(bx0, by0, bx1, by1, text)
            
            for sx0, sy0, sx1, sy1, sub_text in sub_words:
                words.append((
                    float(sx0), float(sy0), float(sx1), float(sy1),
                    str(sub_text), 0, i, word_idx, float(conf)
                ))
                word_idx += 1
                
        return words

    def detect_image_words(
        self, img: np.ndarray, multi_scale: bool = False
    ) -> List[Tuple[float, float, float, float, str, int, int, int]]:
        """
        Detect words in an image and return them in PyMuPDF 'words' format.
        Wraps detect_multiscale_words for unified access.
        """
        scales = [1.0, 2.0] if multi_scale else [1.0]
        return self.detect_multiscale_words(img, scales=scales)

    def force_recognize_regions(
        self, img: np.ndarray, regions: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[float, float, float, float, str, float]]:
        """
        Bypass DET (Detection) and force REC (Recognition) on specific regions.
        Useful for rescuing text blocks that are too degraded for the DET model
        but can still be read by the REC model.

        Args:
            img: OpenCV/numpy image array (BGR format)
            regions: List of (x0, y0, x1, y1) bounding boxes to force-recognize

        Returns:
            List of (x0, y0, x1, y1, text, confidence) for successfully recognized regions.
        """
        if not self._engine or not hasattr(self._engine, 'text_rec'):
            return []

        # Phase 6: Dynamic Tensor Batching for REC
        crop_list = []
        valid_regions = []
        for x0, y0, x1, y1 in regions:
            # Crop region
            crop_img = img[int(y0):int(y1), int(x0):int(x1)]
            if crop_img.size == 0 or crop_img.shape[0] < 5 or crop_img.shape[1] < 5:
                continue
            crop_list.append(crop_img)
            valid_regions.append((x0, y0, x1, y1))
            
        if not crop_list:
            return []

        results = []
        try:
            # Process all valid crops into a single batched ONNX REC inference
            rec_res, _ = self._engine.text_rec(crop_list)
            for (x0, y0, x1, y1), rec_data in zip(valid_regions, rec_res):
                if rec_data and rec_data[0]:
                    text, conf = rec_data
                    if text.strip() and conf >= 0.5:
                        results.append((float(x0), float(y0), float(x1), float(y1), text, float(conf)))
        except Exception as exc:
            logger.debug(f"operation: suppressed {exc}")
            
        return results


# Singleton instance for easy import
ocr_engine = RapidOCREngine()

def get_ocr_engine() -> RapidOCREngine:
    return ocr_engine
