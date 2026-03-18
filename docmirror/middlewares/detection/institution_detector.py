# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
L2 Institution Identification Middleware
=========================================

When scene=bank_statement, identifies the specific banking institution.
Reads from ParseResult.full_text, writes to ParseResult.entities.organization.
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional

from ...models.entities.parse_result import ParseResult
from ..base import BaseMiddleware

logger = logging.getLogger(__name__)


def _load_registry() -> dict[str, dict[str, Any]]:
    """Load the institution yaml configuration registry."""
    try:
        import yaml

        registry_path = Path(__file__).parent.parent.parent / "configs" / "institution_registry.yaml"
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("institutions", {})
    except Exception as e:
        logger.debug(f"[InstitutionDetector] Failed to load registry: {e}")
    return {}


def _extract_header_area(full_text: str, max_chars: int = 5000) -> str:
    """Isolate the title/header area before transaction column keywords."""
    column_keywords = [
        "Transaction date",
        "凭证Type",
        "交易时间",
        "序号",
        "交易明细",
        "记账Date",
        "交易Type",
    ]
    cut_pos = len(full_text)
    for kw in column_keywords:
        idx = full_text.find(kw)
        if 15 < idx < cut_pos:
            cut_pos = idx
    return full_text[: min(cut_pos, max_chars)]


def detect_institution(full_text: str, registry: dict[str, dict[str, Any]]) -> str | None:
    """Execute L2 Institution Identification."""
    if not full_text or not registry:
        return None
    normalized = unicodedata.normalize("NFKC", full_text)
    header_text = _extract_header_area(normalized)

    # Pass 1: Characteristic Fingerprints
    for inst_id, info in registry.items():
        keywords = info.get("identification_keywords") or []
        if keywords and all(kw in normalized for kw in keywords):
            return inst_id

    # Pass 1.5: Generic regional bank regex
    import re

    _BANK_CONTEXT_KEYWORDS = ("流水", "对账单", "交易明细", "银行", "账号", "账户")
    _has_bank_context = any(kw in header_text for kw in _BANK_CONTEXT_KEYWORDS)
    if _has_bank_context:
        regional_patterns = [
            (r"([\u4e00-\u9fff]{2,6})(农商银行|农村商业银行|农商行)", "rural"),
            (r"([\u4e00-\u9fff]{2,6})(城商银行|城市商业银行)", "city"),
            (r"([\u4e00-\u9fff]{2,6})(村镇银行)", "village"),
        ]
        for pattern, prefix in regional_patterns:
            m = re.search(pattern, header_text)
            if m:
                region = m.group(1)
                return f"{prefix}_{region}"

    # Pass 2: Complete Institutional Names
    sorted_banks = sorted(
        registry.items(),
        key=lambda kv: len(kv[1].get("name", "")),
        reverse=True,
    )
    for inst_id, info in sorted_banks:
        name = info.get("name", "")
        if name and name in header_text:
            return inst_id

    # Pass 3: Alias Matching
    for inst_id, info in registry.items():
        for alias in info.get("aliases", []):
            if alias and alias in header_text:
                return inst_id

    return None


class InstitutionDetector(BaseMiddleware):
    """L2 Institution Identification Middleware — writes to result.entities.organization."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._registry: dict[str, dict[str, Any]] | None = None

    def _get_registry(self) -> dict[str, dict[str, Any]]:
        if self._registry is None:
            self._registry = _load_registry()
        return self._registry

    def process(self, result: ParseResult) -> ParseResult:
        if result.entities.document_type != "bank_statement":
            result.entities.domain_specific["institution"] = None
            return result

        full_text = result.full_text or ""
        registry = self._get_registry()
        institution = detect_institution(full_text, registry)

        result.entities.domain_specific["institution"] = institution
        if institution:
            # Set organization to the institution name from registry
            inst_info = registry.get(institution, {})
            inst_name = inst_info.get("name", institution)
            if not result.entities.organization:
                result.entities.organization = inst_name
            result.record_mutation(
                middleware_name=self.name,
                target_block_id="document",
                field_changed="institution",
                old_value=None,
                new_value=institution,
                confidence=0.9,
                reason="registry match",
            )
            logger.info(f"[InstitutionDetector] -> institution={institution}")
        else:
            logger.debug("[InstitutionDetector] -> no institution matched")
        return result
