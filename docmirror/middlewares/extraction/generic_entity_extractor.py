"""
GenericEntityExtractor — 跨格式通用实体提取中间件
====================================================

从 BaseResult 的 key_value Block 中提取实体。
不依赖任何格式专有逻辑，适用于所有文件格式。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..base import BaseMiddleware
from ...models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)


class GenericEntityExtractor(BaseMiddleware):
    """通用实体提取 — 从 KV blocks 提取实体到 enhanced_data。"""

    def process(self, result: EnhancedResult) -> EnhancedResult:
        if result.base_result is None:
            return result

        entities = result.base_result.entities
        if not entities:
            return result

        existing = result.enhanced_data.get("extracted_entities", {})
        existing.update(entities)
        result.enhanced_data["extracted_entities"] = existing

        result.record_mutation(
            self.name, "doc", "entities", {},
            {k: str(v)[:50] for k, v in entities.items()},
            reason=f"Extracted {len(entities)} entities from KV blocks",
        )
        return result
