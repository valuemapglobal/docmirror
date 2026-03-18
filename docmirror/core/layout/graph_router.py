# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph-Based Semantic Router
============================

Inspired by DeepSeek-OCR 2's Visual Causal Flow (VCF) concept, this module
replaces rigid top-to-bottom scanning with a topology-aware reading-order
algorithm.  It constructs a spatial graph of visual blocks and determines
their causal reading sequence through topological sorting.

Three-stage pipeline:
    1. **Spatial Graph Construction** — build a 2-D connectivity graph where
       a directed edge i → j means "block *i* should be read before block *j*".
    2. **Sidebar Penalisation** — suppress outlier nodes (narrow edge strips).
    3. **Causal Reading Sequence** — Kahn-style topological sort with a
       priority heap that breaks ties by column → semantic type → y-position.

The router also supports an optional **model branch**: when a
``reading_order_model`` path is provided (or ``"auto"``), LayoutLMv3
(``hantian/layoutreader``) is used to predict token-level reading order.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SyntacticBridger:
    """Evaluates semantic text continuity (N-gram/character-level syntax)
    between two text blocks to improve reading-order flow."""

    @staticmethod
    def _extract_text(zone: Any) -> str:
        """Safely extract text from a layout Zone object."""
        if hasattr(zone, "text") and zone.text:
            return zone.text
        if hasattr(zone, "chars") and zone.chars:
            return "".join(
                c.get("text", "") for c in sorted(zone.chars, key=lambda c: (c.get("top", 0), c.get("x0", 0)))
            )
        return ""

    @staticmethod
    def bridging_score(zone_a: Any, zone_b: Any) -> float:
        """
        Calculates a semantic continuity score between the end of zone_a and start of zone_b.
        Score > 0 implies likely continuity (same sentence or list item).
        Score < 0 implies likely break (new paragraph, new thought).
        """
        text_a = SyntacticBridger._extract_text(zone_a).strip()
        text_b = SyntacticBridger._extract_text(zone_b).strip()

        if not text_a or not text_b:
            return 0.0

        # Get last few characters of A and first few of B
        tail_a = text_a[-5:]
        head_b = text_b[:5]

        score = 0.0

        # 1. Punctuation continuity
        if tail_a[-1] in ".。!?！？":
            # Sentence ended
            score -= 1.0
        elif tail_a[-1] in ",，、:：;；":
            # Sentence continues
            score += 1.5
        else:
            # Ends with alphanumeric/hanzi, might be broken word or no-punctuation boundary
            score += 0.5

        # 2. Case continuity (applicable to English)
        if tail_a[-1].islower() and head_b[0].islower():
            score += 1.0
        elif head_b[0].isupper() and tail_a[-1] not in ".。!?！？":
            score -= 0.5

        # 3. List/Enum continuity
        # e.g., A ends with ":" and B starts with "1." or "•"
        if len(head_b) > 1 and head_b[0] in "•-123456789" and head_b[1] in ".、 ":
            if tail_a[-1] in ":：":
                score += 2.0
            else:
                score -= 0.5

        return score


# To avoid circular imports, we reference the Zone structure from layout_analysis.
# Only used for type hints; duck typing with bbox and type also works.


class GraphRouter:
    def __init__(self, page_width: float, page_height: float):
        self.page_width = page_width
        self.page_height = page_height

    def _get_center(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x0, y0, x1, y1 = bbox
        return (x0 + x1) / 2, (y0 + y1) / 2

    def _is_sidebar(self, bbox: tuple[float, float, float, float]) -> bool:
        """Heuristically determine if a block is a sidebar or extreme header/footer."""
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        cx, cy = self._get_center(bbox)

        # Narrow vertical elements hugging the page edges
        if w < self.page_width * 0.15 and (cx < self.page_width * 0.15 or cx > self.page_width * 0.85):
            return True
        return False

    def _detect_columns(self, zones: list[Any]) -> list[int]:
        """Detect column structure (single / double / triple column).

        Determines the number of columns by clustering zone center
        x-coordinates.  Returns one column index per zone (0-based,
        left to right).

        For single-column documents all zones return column 0 — the
        column detection has no effect on sorting.
        """
        if not zones:
            return []

        # Collect centre x-coordinates; skip blocks wider than 60 % of
        # the page (cross-column titles, banners, etc.)
        cx_list = []
        for z in zones:
            x0, y0, x1, y1 = z.bbox
            w = x1 - x0
            if w > self.page_width * 0.6:
                cx_list.append(None)
            else:
                cx_list.append((x0 + x1) / 2)

        valid_cx = [cx for cx in cx_list if cx is not None]
        if len(valid_cx) < 2:
            return [0] * len(zones)

        # Simple gap-based clustering: sort by x and identify significant gaps
        sorted_cx = sorted(valid_cx)
        gaps = []
        for i in range(1, len(sorted_cx)):
            gap = sorted_cx[i] - sorted_cx[i - 1]
            # A gap exceeding 15 % of page width is treated as a column separator
            if gap > self.page_width * 0.15:
                gaps.append((sorted_cx[i - 1] + sorted_cx[i]) / 2)

        if not gaps:
            return [0] * len(zones)

        # Limit to at most 3 columns
        gaps = gaps[:2]

        # Assign a column number to each zone
        columns = []
        for cx in cx_list:
            if cx is None:
                # Cross-column block → assign to column -1 (processed first)
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

    def build_flow(self, zones: list[Any], reading_order_model=None, enable_column_detection: bool = True) -> list[Any]:
        """Produce a reading-order sorted list of zones using graph-based
        topological sort with semantic priorities and spatial relations.

        Primary path: Delaunay triangulation (O(n log n)) via spatial_graph.
        Fallback: O(n²) brute-force adjacency when SciPy is unavailable.

        Args:
            zones: List of zone objects (must have ``.bbox`` and ``.type``).
            reading_order_model: Optional path to a LayoutLMv3 reading-order
                model, or ``"auto"`` for the default ``hantian/layoutreader``.
            enable_column_detection: Whether to enable explicit column
                detection.  Defaults to ``True``.
        """
        if not zones:
            return []

        n = len(zones)
        if n == 1:
            return zones

        # ── Model branch: LayoutLMv3 layoutreader ──
        if reading_order_model:
            model_result = self._model_reading_order(zones, reading_order_model)
            if model_result is not None:
                return model_result

        # ── Primary: Delaunay spatial graph (O(n log n)) ──
        try:
            from .spatial_graph import (
                build_delaunay_adjacency,
                compute_reading_order,
                detect_columns_geometric,
            )

            adj = build_delaunay_adjacency(zones, self.page_width, self.page_height)
            columns = detect_columns_geometric(zones, self.page_width) if enable_column_detection else [0] * n
            sorted_indices = compute_reading_order(
                zones,
                self.page_width,
                self.page_height,
                adj=adj,
                columns=columns,
            )
            logger.debug("Graph Router applied via Delaunay spatial graph. Visual Causal Flow established.")
            return [zones[i] for i in sorted_indices]

        except Exception as exc:
            logger.debug(f"[GraphRouter] Delaunay path failed ({exc}), using O(n²) fallback")

        # -------------------------------------------------------------------
        # Fallback: O(n²) brute-force adjacency (original logic)
        # -------------------------------------------------------------------
        adj: dict[int, set[int]] = defaultdict(set)
        in_degree: dict[int, int] = defaultdict(int)

        # Pre-compute attributes
        is_sidebar = [self._is_sidebar(z.bbox) for z in zones]

        # Column detection (all zones get column 0 for single-column docs)
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

                # ── Causal constraints (directed edge construction) ──

                # Rule 0: Cross-column titles precede column content
                if columns[i] == -1 and columns[j] >= 0:
                    if y1_i < y0_j + 15:
                        adj[i].add(j)
                        continue

                # Evaluate semantic continuity
                semantic_score = SyntacticBridger.bridging_score(z_i, z_j)

                # Rule A: Main content precedes sidebar at same vertical band
                if is_sidebar[j] and not is_sidebar[i]:
                    if abs(cy_i - cy_j) < self.page_height * 0.2:
                        # Only add if it doesn't heavily break semantics
                        if semantic_score >= -1.0:
                            adj[i].add(j)
                        continue

                # Rule B: Block clearly above another takes precedence
                if y1_i < y0_j + 15:  # bottom of i is still above top of j
                    adj[i].add(j)
                    continue

                # Rule C: Horizontal case — left precedes right when heights significantly overlap
                y_overlap = max(0, min(y1_i, y1_j) - max(y0_i, y0_j))
                h_i, h_j = y1_i - y0_i, y1_j - y0_j
                if y_overlap > min(h_i, h_j) * 0.4:  # 40 % overlap → same horizontal row
                    if x1_i < x0_j + 15:  # i is to the left of j
                        if semantic_score >= 0.0:
                            adj[i].add(j)
                        elif x1_i < x0_j - 50:
                            adj[i].add(j)

        # Compute in-degree for each node
        for i in range(n):
            in_degree[i] = 0
        for u in adj:
            for v in adj[u]:
                in_degree[v] += 1

        # -------------------------------------------------------------------
        # Topological sort (Kahn's algorithm) with a priority heap
        # -------------------------------------------------------------------
        import heapq

        _ZONE_ORDER = {"title": 0, "summary": 1, "data_table": 2, "formula": 2, "unknown": 3, "footer": 4}

        queue = []
        for i in range(n):
            if in_degree[i] == 0:
                col = max(0, columns[i])
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

        # Safety net: if cycles exist, fall back to static y + semantic sort
        if len(sorted_indices) != n:
            logger.debug("Graph Router detected cycle, falling back to static sort.")
            return sorted(zones, key=lambda z: (_ZONE_ORDER.get(z.type, 3), z.bbox[1]))

        logger.info(f"[GraphRouter] Topological sort applied successfully: {n} zones. Visual Causal Flow established.")
        return [zones[i] for i in sorted_indices]

    def _model_reading_order(self, zones: list[Any], model_path: str) -> list[Any] | None:
        """Use a LayoutLMv3-based model to predict reading order.

        Args:
            zones: List of zone objects.
            model_path: HuggingFace model path or ``"auto"`` (defaults to
                ``hantian/layoutreader``).

        Returns:
            Sorted zone list, or ``None`` on failure (triggers fallback
            to the graph-based method).
        """
        try:
            import torch
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer
        except ImportError:
            logger.debug("[GraphRouter] transformers/torch not available, using graph fallback")
            return None

        try:
            repo_id = "hantian/layoutreader" if model_path == "auto" else model_path

            if not hasattr(self, "_layoutreader_model"):
                self._layoutreader_model = LayoutLMv3ForTokenClassification.from_pretrained(repo_id)
                self._layoutreader_tokenizer = LayoutLMv3Tokenizer.from_pretrained(repo_id)
                self._layoutreader_model.eval()
                logger.info(f"[GraphRouter] Loaded layoutreader from {repo_id}")

            # Prepare bbox input (normalised to 0–1000 coordinate space)
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

            # Simplified input: one token per zone
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

            # Predicted result: reading-order label for each token
            logits = outputs.logits
            predictions = logits.argmax(-1).squeeze().tolist()

            if isinstance(predictions, int):
                predictions = [predictions]

            # Strip [CLS] and [SEP] token predictions;
            # actual zone tokens correspond to predictions[1 : len(zones)+1]
            zone_orders = predictions[1 : len(zones) + 1]

            # Sort zones by their predicted reading order
            indexed_zones = list(enumerate(zones))
            indexed_zones.sort(key=lambda x: zone_orders[x[0]] if x[0] < len(zone_orders) else 999)

            sorted_zones = [z for _, z in indexed_zones]
            logger.info(f"[GraphRouter] Model reading order applied: {len(sorted_zones)} zones")
            return sorted_zones

        except Exception as e:
            logger.warning(f"[GraphRouter] model reading order failed: {e}, using graph fallback")
            return None
