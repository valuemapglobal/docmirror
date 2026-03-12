"""
Mutation (Data Lineage Tracker)
========================================

Every operation performed by a Middleware on the Data is recorded via a
Mutation, implementing 100% operation traceability to meet audit requirements.

Usage::

    mutation = Mutation.create(
        middleware_name="EntityExtractor",
        target_block_id="blk_a1",
        field_changed="account_holder",
        old_value="",
        new_value="John Doe",
        confidence=0.95,
    )
"""
from __future__ import annotations



import dataclasses
from datetime import datetime, timezone
from typing import Any


@dataclasses.dataclass
class Mutation:
    """
    Single data transformation record.

    Attributes:
        middleware_name: The name of the Middleware executing the change.
        target_block_id: The ID of the transformed Block.
        field_changed:   The name of the transformed field.
        old_value:       The value before the transformation.
        new_value:       The value after the transformation.
        confidence:      The confidence of the transformation (0.0 ~ 1.0).
        timestamp:       The time the transformation occurred.
        reason:          The reason for the transformation (for debugging).
    """
    middleware_name: str
    target_block_id: str
    field_changed: str
    old_value: Any
    new_value: Any
    confidence: float = 1.0
    timestamp: datetime = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    reason: str = ""

    @classmethod
    def create(
        cls,
        middleware_name: str,
        target_block_id: str,
        field_changed: str,
        old_value: Any,
        new_value: Any,
        confidence: float = 1.0,
        reason: str = "",
    ) -> Mutation:
        """Factory Method \u2014 Automatically populates the timestamp."""
        return cls(
            middleware_name=middleware_name,
            target_block_id=target_block_id,
            field_changed=field_changed,
            old_value=old_value,
            new_value=new_value,
            confidence=confidence,
            reason=reason,
        )

    def to_dict(self) -> dict:
        """Serialization to dict \u2014 Used for logging and persistence."""
        return {
            "middleware": self.middleware_name,
            "block_id": self.target_block_id,
            "field": self.field_changed,
            "old": str(self.old_value)[:200],
            "new": str(self.new_value)[:200],
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
        }
