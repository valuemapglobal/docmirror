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

from .domain import BaseResult
from ..tracking.mutation import Mutation

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
    base_result: Optional[BaseResult] = None
    enhanced_data: Dict[str, Any] = dataclasses.field(default_factory=dict)
    scene: str = "unknown"
    institution: Optional[str] = None  # L2 institution id, e.g., ccb / citic
    mutations: List[Mutation] = dataclasses.field(default_factory=list)
    status: Literal["success", "partial", "failed"] = "success"
    processing_time: float = 0.0
    errors: List[str] = dataclasses.field(default_factory=list)

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
    def validation_result(self) -> Optional[Dict[str, Any]]:
        """Fetch analytical validation results."""
        return self.enhanced_data.get("validation")

    @property
    def mutation_count(self) -> int:
        return len(self.mutations)

    @property
    def mutation_summary(self) -> Dict[str, int]:
        """Summarize active mutations metrics bounded per middleware origin."""
        summary: Dict[str, int] = {}
        for m in self.mutations:
            summary[m.middleware_name] = summary.get(m.middleware_name, 0) + 1
        return summary

    # \u2500\u2500 v1 Compatibility Bridge \u2500\u2500

    def to_parser_output(self):
        """
        [DEPRECATED] Bridges functionally to legacy v1 ParserOutput.

        Favor `PerceptionResultBuilder.build()` exclusively instead.
        This function is retained only for explicit backward compatibility.
        """
        import warnings
        warnings.warn(
            "to_parser_output() is deprecated, use "
            "PerceptionResultBuilder.build() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        ParserOutput = None
        ParserStatus = None
        try:
            from docmirror.framework.base import (
                ParserOutput as _PO,
                ParserStatus as _PS,
            )
            ParserOutput = _PO
            ParserStatus = _PS
        except ImportError:
            pass

        if self.base_result is None:
            if ParserOutput and ParserStatus:
                return ParserOutput(
                    status=ParserStatus.FAILURE,
                    error="No base result available",
                )
            return {"status": "failure", "error": "No base result available"}

        status_map = {
            "success": ParserStatus.SUCCESS if ParserStatus else "success",
            "partial": (
                ParserStatus.PARTIAL_SUCCESS if ParserStatus else "partial"
            ),
            "failed": ParserStatus.FAILURE if ParserStatus else "failed",
        }

        # Structure mapping safely rationally from base logic structurally
        doc_structure = []
        for block in self.base_result.all_blocks:
            entry: Dict[str, Any] = {
                "type": block.block_type,
                "page": block.page,
            }
            if block.block_type == "table":
                table_data = block.raw_content
                if isinstance(table_data, list) and table_data:
                    entry["headers"] = table_data[0] if table_data else []
                    entry["rows"] = (
                        table_data[1:] if len(table_data) > 1 else []
                    )
                    entry["data"] = table_data
            elif block.block_type == "key_value":
                pairs = {}
                if isinstance(block.raw_content, dict):
                    pairs = block.raw_content
                entry["pairs"] = pairs
            elif block.block_type in ("text", "title", "footer"):
                content = ""
                if isinstance(block.raw_content, str):
                    content = block.raw_content
                entry["text"] = content
            doc_structure.append(entry)



        metadata = dict(self.base_result.metadata)
        metadata.update({
            "parser": "DocMirror",
            "scene": self.scene,
            "enhance_mode": self.enhanced_data.get("enhance_mode", "unknown"),
            "page_count": self.base_result.page_count,
            "mutation_count": self.mutation_count,
            "mutation_summary": self.mutation_summary,
            "processing_time_ms": round(self.processing_time, 1),
            "errors": self.errors,
        })

        entities = self.base_result.entities
        inst_val = self.institution or self.enhanced_data.get("institution")
        if not inst_val and isinstance(entities, dict):
            inst_val = entities.get(
                "bank_name", entities.get("Bank name", "")
            )
        metadata["institution"] = inst_val

        acc_name = entities.get(
            "Account name", entities.get("Account name", "")
        )
        acc_num = entities.get(
            "Account number", entities.get("Card number", "")
        )
        q_period = entities.get("Query period", entities.get("Period", ""))
        currency = entities.get("Currency", "CNY")
        metadata["identity"] = {
            "document_type": self.scene,
            "page_count": self.base_result.page_count,
            "properties": {
                "institution": inst_val,
                "account_holder": acc_name,
                "account_number": acc_num,
                "query_period": q_period,
                "currency": currency,
            },
        }

        validation = self.validation_result
        if validation:
            metadata["l2_score"] = validation.get("total_score")
            metadata["l2_passed"] = validation.get("passed")

        if self.base_result.metadata.get("seal_info"):
            metadata["trust"] = metadata.get("trust") or {}
            metadata["trust"]["seal_info"] = self.base_result.metadata[
                "seal_info"
            ]

        # \u2500\u2500 Output Delivery Bounds \u2500\u2500
        active_status = "success"
        if isinstance(self.status, str):
            active_status = self.status
        conf = 1.0 if active_status == "success" else (
            0.5 if active_status == "partial" else 0.0
        )

        if ParserOutput and ParserStatus:
            mapped_status = ParserStatus.FAILURE
            if hasattr(self, 'status') and isinstance(self.status, str):
                if self.status in status_map:
                    mapped_status = status_map[self.status]
            return ParserOutput(
                status=mapped_status,
                confidence=conf,
                structured_text=self.base_result.full_text,
                key_entities=entities,
                document_structure=doc_structure,
                metadata=metadata,
            )

        return {
            "status": active_status,
            "confidence": conf,
            "structured_text": self.base_result.full_text,
            "key_entities": entities,
            "document_structure": doc_structure,
            "metadata": metadata,
        }

    @classmethod
    def from_base_result(cls, base_result: BaseResult) -> EnhancedResult:
        """Instantiate EnhancedResult derived natively from BaseResult."""
        return cls(
            document_id=base_result.document_id,
            base_result=base_result,
        )
