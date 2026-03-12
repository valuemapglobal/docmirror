"""
GenericEntityExtractor \u2014 Universal Entity Extraction Middleware
==============================================================

Extracts entities iteratively cleanly across any document format securely.
"""
from __future__ import annotations


import logging

from ..base import BaseMiddleware
from ...models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)


class GenericEntityExtractor(BaseMiddleware):
    """Generic entities Extraction \u2014 Harvests target explicitly safely."""

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
