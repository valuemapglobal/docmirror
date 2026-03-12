"""
L2 Institution Identification Middleware (Institution Detector)
=============================================================

When scene=bank_statement, identifies the specific concrete banking institution 
(institution id) utilizing OCR text boundaries from the first page/headline texts.

Strategy: 
  - Pass 1: Strict identification_keywords fingerprint match.
  - Pass 2: Fallback length-sorted full institutional name string match.
  - Pass 3: Alias matching capabilities (e.g. abbreviations).
Configuration mapped from `configs/institution_registry.yaml`.
"""
from __future__ import annotations


import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional

from ..base import BaseMiddleware
from ...models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)


def _load_registry() -> Dict[str, Dict[str, Any]]:
    """Load the institution yaml configuration registry."""
    try:
        import yaml
        registry_path = Path(__file__).parent.parent.parent / "configs" / "institution_registry.yaml"
        if registry_path.exists():
            with open(registry_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("institutions", {})
    except Exception as e:
        logger.debug(f"[InstitutionDetector] Failed to load registry: {e}")
    return {}


def _extract_header_area(full_text: str, max_chars: int = 5000) -> str:
    """
    Isolate the "Title/Header Area": the textual payload occurring prior to the 
    first detection of common transaction column table keywords.
    Crucial for avoiding false-positive matches reading counterparty bank names 
    deep inside transaction history rows.
    """
    column_keywords = [
        "Transaction date", "凭证Type", "交易时间", "序号",
        "交易明细", "记账Date", "交易Type",
    ]
    cut_pos = len(full_text)
    for kw in column_keywords:
        idx = full_text.find(kw)
        if 15 < idx < cut_pos:
            cut_pos = idx
    return full_text[: min(cut_pos, max_chars)]


def detect_institution(full_text: str, registry: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    Execute L2 Institution Identification: Returns matched institution_id or None.
    Evaluates primarily via unique `identification_keywords`, falling back 
    to full-name overlap, sorted descending by length to prevent partial short-matches.
    """
    if not full_text or not registry:
        return None
    normalized = unicodedata.normalize("NFKC", full_text)
    header_text = _extract_header_area(normalized)

    # Pass 1: Characteristic Fingerprints (identification_keywords)
    for inst_id, info in registry.items():
        keywords = info.get("identification_keywords") or []
        if keywords and all(kw in normalized for kw in keywords):
            return inst_id

    # Pass 2: Complete Institutional Names (Sorted descending by length)
    sorted_banks = sorted(
        registry.items(),
        key=lambda kv: len(kv[1].get("name", "")),
        reverse=True,
    )
    for inst_id, info in sorted_banks:
        name = info.get("name", "")
        if name and name in header_text:
            return inst_id

    # Pass 3: Alias Matching (Alternative names/acronyms, e.g., "工行" \u2192 icbc)
    for inst_id, info in registry.items():
        for alias in info.get("aliases", []):
            if alias and alias in header_text:
                return inst_id

    return None


class InstitutionDetector(BaseMiddleware):
    """
    L2 Institution Identification Middleware.
    
    Contextually restricts execution to `bank_statement` scenes.
    Writes matched outputs into `result.enhanced_data["institution"]`, feeding 
    downstream logic like entity extraction and identity resolution.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._registry: Optional[Dict[str, Dict[str, Any]]] = None

    def _get_registry(self) -> Dict[str, Dict[str, Any]]:
        if self._registry is None:
            self._registry = _load_registry()
        return self._registry

    def process(self, result: EnhancedResult) -> EnhancedResult:
        if result.base_result is None:
            return result
        if result.scene != "bank_statement":
            result.enhanced_data["institution"] = None
            return result

        full_text = result.base_result.full_text or ""
        registry = self._get_registry()
        institution = detect_institution(full_text, registry)

        result.enhanced_data["institution"] = institution
        result.institution = institution
        if institution:
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
