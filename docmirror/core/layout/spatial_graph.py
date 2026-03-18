# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Spatial Proximity Graph Engine
===============================

Computational geometry module for layout analysis, replacing O(n²) brute-force
adjacency construction with Delaunay triangulation and 1D signal projection.

Three core algorithms:

1. **Delaunay Adjacency** — O(n log n) spatial neighbor graph via SciPy Delaunay.
2. **Geometric Column Detection** — 1D X-axis projection with peak finding.
3. **Topological Reading Order** — Kahn's algorithm over the pruned DAG.

All algorithms are CPU-only, use no GPU or VLM, and require only SciPy
(already a project dependency via pre_analyzer.py).
"""

from __future__ import annotations

import heapq
import logging
from collections import defaultdict
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


def build_delaunay_adjacency(
    zones: list[Any],
    page_width: float,
    page_height: float,
) -> dict[int, set[int]]:
    """Build a spatial adjacency graph using Delaunay triangulation.

    Each zone is represented by its bounding-box centre point.  The Delaunay
    triangulation connects every pair of points that share a triangle edge,
    which naturally captures "visual neighbour" relationships.

    For n < 3 or collinear points, falls back to a simple sequential chain.

    Args:
        zones: Zone objects with ``.bbox`` attribute ``(x0, y0, x1, y1)``.
        page_width: Page width in points.
        page_height: Page height in points.

    Returns:
        Undirected adjacency dict ``{zone_idx: set(neighbor_indices)}``.
    """
    n = len(zones)
    adj: dict[int, set[int]] = defaultdict(set)

    if n <= 1:
        return adj

    # For n == 2, connect them directly
    if n == 2:
        adj[0].add(1)
        adj[1].add(0)
        return adj

    # Extract centre points
    points = []
    for z in zones:
        x0, y0, x1, y1 = z.bbox
        points.append(((x0 + x1) / 2, (y0 + y1) / 2))

    try:
        import numpy as np
        from scipy.spatial import Delaunay

        pts = np.array(points)

        # Check for collinearity (all points on a line)
        # QHull will raise an error if points are degenerate
        tri = Delaunay(pts, qhull_options="QJ")  # QJ = joggled input for robustness

        # Extract adjacency from triangle simplices
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    adj[a].add(b)
                    adj[b].add(a)

    except Exception as exc:
        # Fallback: sequential chain (connect i → i+1)
        logger.debug(f"Delaunay fallback to chain: {exc}")
        sorted_indices = sorted(range(n), key=lambda i: (points[i][1], points[i][0]))
        for k in range(len(sorted_indices) - 1):
            a, b = sorted_indices[k], sorted_indices[k + 1]
            adj[a].add(b)
            adj[b].add(a)

    return adj


def detect_columns_geometric(
    zones: list[Any],
    page_width: float,
) -> list[int]:
    """Detect column structure using 1D X-axis projection and peak finding.

    Clusters zone centre X-coordinates into columns by finding natural
    gaps in the projection.  Cross-column blocks (width > 60% page) are
    assigned column -1 (processed first in reading order).

    Args:
        zones: Zone list with ``.bbox``.
        page_width: Page width in points.

    Returns:
        Column index per zone (0-based, left-to-right); -1 = cross-column.
    """
    n = len(zones)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Collect centre X for narrow zones only
    cx_list = []
    for z in zones:
        x0, y0, x1, y1 = z.bbox
        w = x1 - x0
        if w > page_width * 0.6:
            cx_list.append(None)  # cross-column
        else:
            cx_list.append((x0 + x1) / 2)

    valid_cx = [cx for cx in cx_list if cx is not None]
    if len(valid_cx) < 2:
        return [0 if cx is not None else -1 for cx in cx_list]

    # Sort and find significant gaps (> 15% page width)
    sorted_cx = sorted(valid_cx)
    gap_threshold = page_width * 0.15
    gaps = []
    for i in range(1, len(sorted_cx)):
        gap = sorted_cx[i] - sorted_cx[i - 1]
        if gap > gap_threshold:
            gaps.append((sorted_cx[i - 1] + sorted_cx[i]) / 2)

    if not gaps:
        return [0 if cx is not None else -1 for cx in cx_list]

    # Limit to at most 2 gaps (3 columns)
    gaps = gaps[:2]

    # Assign column numbers
    columns = []
    for cx in cx_list:
        if cx is None:
            columns.append(-1)
        else:
            col = 0
            for g in gaps:
                if cx > g:
                    col += 1
            columns.append(col)

    num_cols = len(gaps) + 1
    if num_cols > 1:
        logger.debug(f"[SpatialGraph] Detected {num_cols}-column layout")

    return columns


def compute_reading_order(
    zones: list[Any],
    page_width: float,
    page_height: float,
    *,
    adj: dict[int, set[int]] | None = None,
    columns: list[int] | None = None,
    syntactic_bridger: Any = None,
) -> list[int]:
    """Compute reading order via topological sort over a directed acyclic graph.

    Converts the undirected Delaunay adjacency graph into a DAG by applying
    directional constraints:
      - Above → below  (primary)
      - Left column → right column  (multi-column)
      - Cross-column blocks precede column content

    Then applies Kahn's algorithm with a priority heap to break ties:
      ``(column, semantic_weight, y_position, index)``

    Args:
        zones: Zone list.
        page_width: Page width.
        page_height: Page height.
        adj: Pre-computed undirected adjacency (built if None).
        columns: Pre-computed column assignment (detected if None).
        syntactic_bridger: Optional SyntacticBridger instance for
            semantic continuity scoring.

    Returns:
        List of zone indices in reading order.
    """
    n = len(zones)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Build adjacency and columns if not provided
    if adj is None:
        adj = build_delaunay_adjacency(zones, page_width, page_height)
    if columns is None:
        columns = detect_columns_geometric(zones, page_width)

    # ── Convert undirected → directed (DAG) ──
    dag: dict[int, set[int]] = defaultdict(set)

    # Pre-compute zone attributes
    _ZONE_ORDER = {
        "title": 0,
        "summary": 1,
        "data_table": 2,
        "formula": 2,
        "unknown": 3,
        "footer": 4,
    }

    def _is_sidebar(bbox):
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        cx = (x0 + x1) / 2
        return w < page_width * 0.15 and (cx < page_width * 0.15 or cx > page_width * 0.85)

    is_sidebar = [_is_sidebar(z.bbox) for z in zones]

    for i in range(n):
        for j in adj.get(i, set()):
            if i == j:
                continue

            x0_i, y0_i, x1_i, y1_i = zones[i].bbox
            x0_j, y0_j, x1_j, y1_j = zones[j].bbox
            cy_i = (y0_i + y1_i) / 2
            cy_j = (y0_j + y1_j) / 2

            # Rule 0: Cross-column titles precede column content
            if columns[i] == -1 and columns[j] >= 0:
                if y1_i < y0_j + 15:
                    dag[i].add(j)
                    continue

            # Rule A: Main content precedes sidebar at same vertical band
            if is_sidebar[j] and not is_sidebar[i]:
                if abs(cy_i - cy_j) < page_height * 0.2:
                    dag[i].add(j)
                    continue

            # Rule B: Block clearly above → precedes
            if y1_i < y0_j + 15:
                dag[i].add(j)
                continue

            # Rule C: Horizontal — left precedes right when overlapping vertically
            y_overlap = max(0, min(y1_i, y1_j) - max(y0_i, y0_j))
            h_i, h_j = y1_i - y0_i, y1_j - y0_j
            if h_i > 0 and h_j > 0 and y_overlap > min(h_i, h_j) * 0.4:
                if x1_i < x0_j + 15:
                    dag[i].add(j)

    # ── Kahn's algorithm with priority heap ──
    in_degree = [0] * n
    for u in dag:
        for v in dag[u]:
            in_degree[v] += 1

    queue = []
    for i in range(n):
        if in_degree[i] == 0:
            col = max(0, columns[i])
            qw = _ZONE_ORDER.get(getattr(zones[i], "type", "unknown"), 3)
            heapq.heappush(queue, (col, qw, zones[i].bbox[1], i))

    sorted_indices = []
    while queue:
        _, _, _, u = heapq.heappop(queue)
        sorted_indices.append(u)
        for v in dag[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                col = max(0, columns[v])
                qw = _ZONE_ORDER.get(getattr(zones[v], "type", "unknown"), 3)
                heapq.heappush(queue, (col, qw, zones[v].bbox[1], v))

    # Cycle safety: if not all nodes visited, fall back to static sort
    if len(sorted_indices) != n:
        logger.debug("[SpatialGraph] Cycle detected, falling back to static sort")
        return sorted(
            range(n),
            key=lambda i: (
                _ZONE_ORDER.get(getattr(zones[i], "type", "unknown"), 3),
                zones[i].bbox[1],
            ),
        )

    return sorted_indices
