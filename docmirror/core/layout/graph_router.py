"""
图基语义路由 (Graph-based Semantic Router)
=============================================

DeepSeek-OCR 2 的核心理念之一是 Visual Causal Flow (VCF)，即放弃原本死板的由上而下扫描，
转而基于各视觉块的关联建立“拓扑排序流”。
本模块为 MultiModal 专门设计的轻量级 Graph Router，用于接管传统 y-band 的硬切分：
1. 构建二维空间连通图 (Spatial Graph Construction)
2. 对离群节点打压 (Sidebar Penalization)
3. 拓扑排序输出因果阅读流 (Causal Reading Sequence)
"""

import math
from typing import List, Tuple, Dict, Set, Any, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# 为了规避循环引用，引入我们在 layout_analysis 中定义的 Zone 的结构
# 仅用于类型提示，如果是鸭子类型 bbox 和 type 也可以直接运算

class GraphRouter:
    def __init__(self, page_width: float, page_height: float):
        self.page_width = page_width
        self.page_height = page_height

    def _get_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x0, y0, x1, y1 = bbox
        return (x0 + x1) / 2, (y0 + y1) / 2

    def _is_sidebar(self, bbox: Tuple[float, float, float, float]) -> bool:
        """启发式判断一个块是否属于远离主轴的侧边栏或极边缘的页眉脚。"""
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        cx, cy = self._get_center(bbox)
        
        # 狭长的垂直元素
        if w < self.page_width * 0.15 and (cx < self.page_width * 0.15 or cx > self.page_width * 0.85):
            return True
        return False

    def _detect_columns(self, zones: List[Any]) -> List[int]:
        """检测页面中的栏结构 (单栏/双栏/三栏)。

        通过对 zone 中心 x 坐标聚类来判断栏数。
        返回每个 zone 对应的栏号 (0-based, 从左到右)。

        对单栏文档 (如银行流水) 所有 zone 返回栏号 0 — 无影响。
        """
        if not zones:
            return []

        # 收集非侧边栏的 zone 中心 x 坐标
        cx_list = []
        for z in zones:
            x0, y0, x1, y1 = z.bbox
            w = x1 - x0
            # 跳过宽度超过页面 60% 的块 (跨栏标题等)
            if w > self.page_width * 0.6:
                cx_list.append(None)
            else:
                cx_list.append((x0 + x1) / 2)

        # 过滤有效的中心点
        valid_cx = [cx for cx in cx_list if cx is not None]
        if len(valid_cx) < 2:
            return [0] * len(zones)

        # 简单聚类: 按 x 坐标排序后寻找显著间隔
        sorted_cx = sorted(valid_cx)
        gaps = []
        for i in range(1, len(sorted_cx)):
            gap = sorted_cx[i] - sorted_cx[i-1]
            if gap > self.page_width * 0.15:  # 间隔超过页宽 15% 视为栏分隔
                gaps.append((sorted_cx[i-1] + sorted_cx[i]) / 2)

        if not gaps:
            return [0] * len(zones)

        # 限制最多 3 栏
        gaps = gaps[:2]

        # 为每个 zone 分配栏号
        columns = []
        for cx in cx_list:
            if cx is None:
                # 跨栏块: 分配到第一栏 (会被最先处理)
                columns.append(-1)
            else:
                col = 0
                for g in gaps:
                    if cx > g:
                        col += 1
                columns.append(col)

        num_cols = len(gaps) + 1
        if num_cols > 1:
            logger.debug(f"[GraphRouter] Detected {num_cols}-column layout")

        return columns

    def build_flow(self, zones: List[Any], reading_order_model=None,
                   enable_column_detection: bool = True) -> List[Any]:
        """
        基于图论对 zones 重新进行语义优先级和空间关系的拓扑排序。
        不再单纯依赖 top/bottom 的 y-band 拦截。

        Args:
            zones: Zone 列表。
            reading_order_model: 可选的 layoutreader 模型路径或 "auto"。
            enable_column_detection: 是否启用显式栏检测。默认 True。
                对单栏文档无影响，对多栏文档可大幅提升阅读顺序准确性。
        """
        if not zones:
            return []
            
        n = len(zones)
        if n == 1:
            return zones

        # ── 模型分支: layoutreader ──
        if reading_order_model:
            model_result = self._model_reading_order(zones, reading_order_model)
            if model_result is not None:
                return model_result

        # 1. 建立邻接图
        # 边的方向代表 "A -> 应该先于 -> B 阅读"
        adj: Dict[int, Set[int]] = defaultdict(set)
        in_degree: Dict[int, int] = defaultdict(int)
        
        # 预计算属性
        is_sidebar = [self._is_sidebar(z.bbox) for z in zones]

        # 栏检测 (对单栏文档所有 zone 返回 0 — 无影响)
        columns = self._detect_columns(zones) if enable_column_detection else [0] * n
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                z_i = zones[i]
                z_j = zones[j]
                x0_i, y0_i, x1_i, y1_i = z_i.bbox
                x0_j, y0_j, x1_j, y1_j = z_j.bbox
                
                cy_i = (y0_i + y1_i) / 2
                cy_j = (y0_j + y1_j) / 2
                
                # Causal Constraints (因果有向边构建)

                # Rule 0: 跨栏标题先于栏内内容
                if columns[i] == -1 and columns[j] >= 0:
                    if y1_i < y0_j + 15:
                        adj[i].add(j)
                        continue
                
                # Rule A: 主干先于侧边栏如果属于同一水平面 (Penalty logic)
                if is_sidebar[j] and not is_sidebar[i]:
                    # 若在同一大段高度内，主干必须优先于 Sidebar
                    if abs(cy_i - cy_j) < self.page_height * 0.2:
                        adj[i].add(j)
                        continue
                
                # Rule B: 显著的上方优先于下方
                if y1_i < y0_j + 15:  # i 的底部还在 j 顶部的明显上方
                    adj[i].add(j)
                    continue
                    
                # Rule C: 水平分栏的情况 (左侧优先于右侧)
                # 当它们的高度显著重叠时
                y_overlap = max(0, min(y1_i, y1_j) - max(y0_i, y0_j))
                h_i, h_j = y1_i - y0_i, y1_j - y0_j
                if y_overlap > min(h_i, h_j) * 0.4:  # 重叠度达到 40% 视为同一栏
                    if x1_i < x0_j + 15:  # i 在 j 左边
                        adj[i].add(j)
                        
        # 计算入度
        for i in range(n):
            in_degree[i] = 0 # 初始化所有节点
        for u in adj:
            for v in adj[u]:
                in_degree[v] += 1

        # 2. 拓扑排序 (Kahn's Algorithm)
        # 用优先队列（堆）来做拓扑排序的决胜局
        # 权重: 栏号 > type 语义 > y 轴高度
        import heapq
        
        _ZONE_ORDER = {
            "title": 0, 
            "summary": 1, 
            "data_table": 2,
            "formula": 2,
            "unknown": 3, 
            "footer": 4
        }
        
        # Queue item: (column, type_weight, y_position, index)
        queue = []
        for i in range(n):
            if in_degree[i] == 0:
                col = max(0, columns[i])  # -1 (跨栏) → 0 (最先)
                qw = _ZONE_ORDER.get(zones[i].type, 3)
                heapq.heappush(queue, (col, qw, zones[i].bbox[1], i))
                
        sorted_indices = []
        
        while queue:
            _, _, _, u = heapq.heappop(queue)
            sorted_indices.append(u)
            
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    col = max(0, columns[v])
                    qw = _ZONE_ORDER.get(zones[v].type, 3)
                    heapq.heappush(queue, (col, qw, zones[v].bbox[1], v))
        
        # 兜底：如果存在环 (Cycle)，降级为原始 Y 轴+语义双键排序
        if len(sorted_indices) != n:
            logger.debug("[v2] Graph Router detected cycle, falling back to static sort.")
            return sorted(zones, key=lambda z: (_ZONE_ORDER.get(z.type, 3), z.bbox[1]))
            
        logger.debug(f"[v2] Graph Router applied successfully. Visual Causal Flow established.")
        return [zones[i] for i in sorted_indices]

    def _model_reading_order(self, zones: List[Any], model_path: str) -> Optional[List[Any]]:
        """使用 layoutreader 模型预测阅读顺序。

        Args:
            zones: Zone 列表。
            model_path: HuggingFace 模型路径或 "auto"。

        Returns:
            排序后的 Zone 列表，或 None (失败时回退到图方法)。
        """
        try:
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer
            import torch
        except ImportError:
            logger.debug("[GraphRouter] transformers/torch not available, using graph fallback")
            return None

        try:
            repo_id = "hantian/layoutreader" if model_path == "auto" else model_path

            if not hasattr(self, '_layoutreader_model'):
                self._layoutreader_model = LayoutLMv3ForTokenClassification.from_pretrained(repo_id)
                self._layoutreader_tokenizer = LayoutLMv3Tokenizer.from_pretrained(repo_id)
                self._layoutreader_model.eval()
                logger.info(f"[GraphRouter] Loaded layoutreader from {repo_id}")

            # 准备 bbox 输入 (归一化到 0-1000)
            bboxes = []
            for z in zones:
                x0, y0, x1, y1 = z.bbox
                norm_bbox = [
                    max(0, int(x0 / self.page_width * 1000)),
                    max(0, int(y0 / self.page_height * 1000)),
                    min(1000, int(x1 / self.page_width * 1000)),
                    min(1000, int(y1 / self.page_height * 1000)),
                ]
                bboxes.append(norm_bbox)

            # 简化输入: 每个 zone 一个 token
            words = [f"zone{i}" for i in range(len(zones))]

            encoding = self._layoutreader_tokenizer(
                words,
                boxes=bboxes,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            with torch.no_grad():
                outputs = self._layoutreader_model(**encoding)

            # 预测的结果是每个 token 的阅读顺序标签
            logits = outputs.logits
            predictions = logits.argmax(-1).squeeze().tolist()

            if isinstance(predictions, int):
                predictions = [predictions]

            # 去掉 [CLS] 和 [SEP] 标记的预测
            # 实际 token 对应 predictions[1:-1]
            zone_orders = predictions[1:len(zones)+1]

            # 按照预测的阅读顺序排序
            indexed_zones = list(enumerate(zones))
            indexed_zones.sort(key=lambda x: zone_orders[x[0]] if x[0] < len(zone_orders) else 999)

            sorted_zones = [z for _, z in indexed_zones]
            logger.info(f"[GraphRouter] Model reading order applied: {len(sorted_zones)} zones")
            return sorted_zones

        except Exception as e:
            logger.warning(f"[GraphRouter] model reading order failed: {e}, using graph fallback")
            return None
