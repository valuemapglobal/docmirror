# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
GenericEntityExtractor — Universal Entity Extraction Middleware
==============================================================

Extracts entities from KV pairs across any document format.
Writes to ParseResult.entities.domain_specific.
"""

from __future__ import annotations

import logging

from ...models.entities.parse_result import ParseResult
from ..base import BaseMiddleware

logger = logging.getLogger(__name__)


class GenericEntityExtractor(BaseMiddleware):
    """Generic entity extraction — harvests KV entities from all pages."""

    def process(self, result: ParseResult) -> ParseResult:
        entities = result.kv_entities
        if not entities:
            return result

        # Merge into domain_specific
        existing = result.entities.domain_specific.get("extracted_entities", {})
        existing.update(entities)
        result.entities.domain_specific["extracted_entities"] = existing

        result.record_mutation(
            self.name,
            "doc",
            "entities",
            {},
            {k: str(v)[:50] for k, v in entities.items()},
            reason=f"Extracted {len(entities)} entities from KV blocks",
        )
        return result
