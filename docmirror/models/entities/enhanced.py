# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
EnhancedResult \u2014 The Augmented Final Extract
============================================

This serves as the finalized output from the MiddlewarePipeline, aggregating:
    1. Immutable reference to the original BaseResult raw extraction.
    2. Augmented structured data injections.
    3. Document scene/classification detections.
    4. Comprehensive mutation transformation histories.

Also supplies `to_parser_output()` bridging back to legacy `ParserOutput`
ensuring absolute backwards compatibility with existing `ParserDispatcher`
and `PerceptionResult` configurations smoothly.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Literal, Optional

from ..tracking.mutation import Mutation
from .domain import BaseResult

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EnhancedResult:
    """
    Enhanced structure \u2014 the culmination object of the MiddlewarePipeline.

    Design principles:
        - base_result retains read-only purity, never mutated.
        - enhanced_data populated progressively across individual middlewares.
        - mutations logs an audit trail of all transformation actions.
        - status reflects Pipeline execution health dynamically.
    """

    document_id: str = ""
    base_result: BaseResult | None = None
    enhanced_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    scene: str = "unknown"
    institution: str | None = None  # L2 institution id, e.g., ccb / citic
    mutations: list[Mutation] = dataclasses.field(default_factory=list)
    status: Literal["success", "partial", "failed"] = "success"
    processing_time: float = 0.0
    errors: list[str] = dataclasses.field(default_factory=list)

    # \u2500\u2500 Middleware Helper Methods \u2500\u2500

    def add_mutation(self, mutation: Mutation) -> None:
        """Appends a single audited transformation boundary record."""
        self.mutations.append(mutation)

    def record_mutation(
        self,
        middleware_name: str,
        target_block_id: str,
        field_changed: str,
        old_value: Any,
        new_value: Any,
        confidence: float = 1.0,
        reason: str = "",
    ) -> None:
        """Create and attach a Mutation object directly."""
        self.mutations.append(
            Mutation.create(
                middleware_name=middleware_name,
                target_block_id=target_block_id,
                field_changed=field_changed,
                old_value=old_value,
                new_value=new_value,
                confidence=confidence,
                reason=reason,
            )
        )

    def add_error(self, error: str) -> None:
        """Records error signals and downgrades operational status."""
        self.errors.append(error)
        if self.status == "success":
            self.status = "partial"

    # \u2500\u2500 Access Convenience Properties \u2500\u2500

    @property
    def validation_result(self) -> dict[str, Any] | None:
        """Fetch analytical validation results."""
        return self.enhanced_data.get("validation")

    @property
    def mutation_count(self) -> int:
        return len(self.mutations)

    @property
    def mutation_summary(self) -> dict[str, int]:
        """Summarize active mutations metrics bounded per middleware origin."""
        summary: dict[str, int] = {}
        for m in self.mutations:
            summary[m.middleware_name] = summary.get(m.middleware_name, 0) + 1
        return summary

    @classmethod
    def from_base_result(cls, base_result: BaseResult) -> EnhancedResult:
        """Instantiate EnhancedResult derived natively from BaseResult."""
        return cls(
            document_id=base_result.document_id,
            base_result=base_result,
        )
