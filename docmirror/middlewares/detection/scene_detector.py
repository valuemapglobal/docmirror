"""
Scene Detection Middleware (Scene Detector)
===========================================

Three-tier progressive scenario detection methodology:
    Tier 1 (Keyword Rules): Extremely high efficiency, covering common known scenarios.
    Tier 2 (Table Header Context): Structural feature analysis identifying logical document intent.
    Tier 3 (Visual Typographic Weights): Identifies visual features to override ambiguous boundaries.

Supported Document Scenes:
    - bank_statement:    Standard bank statement layouts
    - invoice:           Standardized explicit invoices
    - tax_report:        Tax clearance & filing certificates
    - financial_report:  Structured corporate financial reporting 
    - credit_report:     Comprehensive personal/corporate credit reports
    - contract:          General contract agreements
    - generic:           Universal fallback (Unknown structures)
"""
from __future__ import annotations


import logging
from typing import Dict, List, Set, Tuple

from ..base import BaseMiddleware
from ...models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Scenario Keyword Identifier Dictionary Libraries
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

SCENE_KEYWORDS: Dict[str, List[List[str]]] = {
    "bank_statement": [
        # All keywords in a group must be present for the group to match.
        ["交易明细"],
        ["银行", "流水"],
        ["Account number", "交易"],
        ["Transaction", "Balance"],
        ["Statement", "Account"],
    ],
    "invoice": [
        ["Invoice number"],
        ["增值税", "Invoice"],
        ["Invoice", "Tax"],
    ],
    "financial_report": [
        ["审计报告"],
        ["资产负债 table"],
        ["利润 table"],
        ["Annual Report"],
    ],
    "credit_report": [
        ["Credit report"],
        ["信用报告"],
        ["Credit Report"],
    ],
    "contract": [
        ["Contract number"],
        ["Party A", "Party B"],
        ["Contract No"],
    ],
    "tax_report": [
        ["纳税证明"],
        ["税收完税证明"],
        ["完税证明"],
        ["税务", "完税"],
    ],
}

# Table Header Characteristics Indicators: Identifies document contexts indirectly parsing grid structural layouts.
HEADER_FEATURES: Dict[str, List[Set[str]]] = {
    "bank_statement": [
        {"Transaction date", "Amount"},
        {"交易日", "Balance"},
        {"Date", "Amount", "Balance"},
        {"交易时间", "Transaction amount"},
        {"Transaction date", "Transaction amount", "AccountBalance"},
        {"Abstract/Summary", "Amount", "Balance"},
    ],
    "invoice": [
        {"品名", "数量", "单价", "Amount"},
        {"货物", "Tax rate", "Tax amount"},
    ],
    "tax_report": [
        {"税款", "税种", "入库Date"},
        {"纳税Amount", "完税"},
        {"税务机关", "完税证明"},
    ],
}


class SceneDetector(BaseMiddleware):
    """
    Core Scene Classification Processing Middleware.

    Dynamically populates categorical boundaries setting the ``EnhancedResult.scene`` field.
    """

    def process(self, result: EnhancedResult) -> EnhancedResult:
        """Execute three-tier progressive scene detection."""
        if result.base_result is None:
            result.scene = "generic"
            return result

        full_text = result.base_result.full_text
        entities = result.base_result.entities
        table_blocks = result.base_result.table_blocks

        # \u2500\u2500\u2500 Tier 1: Primary Foundational Layout String Markers \u2500\u2500\u2500
        scene, confidence = self._tier1_keyword(full_text)
        if confidence >= 0.8:
            result.scene = scene
            result.record_mutation(
                middleware_name=self.name,
                target_block_id="document",
                field_changed="scene",
                old_value="unknown",
                new_value=scene,
                confidence=confidence,
                reason=f"Tier1 keyword match",
            )
            logger.info(f"[SceneDetector] Tier1 \u2192 {scene} (conf={confidence:.2f})")
            return result

        # \u2500\u2500\u2500 Tier 2: Structural Logical Extraction Footprint Traits \u2500\u2500\u2500
        scene_t2, conf_t2 = self._tier2_header_features(table_blocks)
        if conf_t2 > confidence:
            scene, confidence = scene_t2, conf_t2

        # Combine Tier 1 + Tier 2 with entity-based signals
        scene_entity, conf_entity = self._detect_from_entities(entities)
        if conf_entity > confidence:
            scene, confidence = scene_entity, conf_entity

        # \u2500\u2500\u2500 Visual Typographic Style Weighting Algorithm Constraints \u2500\u2500\u2500
        # Hypothesis: Huge boldized styled visual payloads intersecting tracking sequences yields dominant confidences natively.
        visual_boost = self._visual_feature_boost(result.base_result)
        if visual_boost[1] > 0 and visual_boost[0] == scene:
            confidence = min(0.95, confidence + visual_boost[1])
        elif visual_boost[1] > confidence:
            scene, confidence = visual_boost[0], visual_boost[1]

        if confidence >= 0.6:
            result.scene = scene
            result.record_mutation(
                middleware_name=self.name,
                target_block_id="document",
                field_changed="scene",
                old_value="unknown",
                new_value=scene,
                confidence=confidence,
                reason=f"Tier2 feature+visual match",
            )
            logger.info(f"[SceneDetector] Tier2 \u2192 {scene} (conf={confidence:.2f})")
            return result

        result.scene = scene if confidence >= 0.3 else "generic"
        result.record_mutation(
            middleware_name=self.name,
            target_block_id="document",
            field_changed="scene",
            old_value="unknown",
            new_value=result.scene,
            confidence=confidence,
            reason="tier2_visual_fallback" if confidence >= 0.3 else "low confidence fallback",
        )
        logger.info(f"[SceneDetector] Final \u2192 {result.scene} (conf={confidence:.2f})")
        return result

    # \u2500\u2500\u2500 Tier 1 Configuration Implementations \u2500\u2500\u2500

    def _tier1_keyword(self, text: str) -> Tuple[str, float]:
        """Sequence evaluations \u2014 isolated multi-condition targets naturally sequentially conclusively."""
        if not text:
            return "generic", 0.0

        best_scene = "generic"
        best_conf = 0.0

        for scene, keyword_groups in SCENE_KEYWORDS.items():
            for group in keyword_groups:
                if all(kw in text for kw in group):
                    # More keywords matched → higher confidence
                    conf = min(0.9, 0.7 + 0.1 * len(group))
                    if conf > best_conf:
                        best_scene = scene
                        best_conf = conf

        return best_scene, best_conf

    # \u2500\u2500\u2500 Tier 2 Header Context Inference Domains \u2500\u2500\u2500

    def _tier2_header_features(self, table_blocks) -> Tuple[str, float]:
        """Tier 2: infer scene from table header column names."""
        if not table_blocks:
            return "generic", 0.0

        for block in table_blocks:
            raw = block.raw_content
            if not isinstance(raw, list) or not raw:
                continue

            # Extract header row from table block
            header = raw[0] if raw else []
            header_set = {str(h).strip() for h in header if h}

            for scene, feature_groups in HEADER_FEATURES.items():
                for required in feature_groups:
                    # Substring-based header matching against required keywords
                    matched = 0
                    for req_kw in required:
                        for h in header_set:
                            if req_kw in h or h in req_kw:
                                matched += 1
                                break
                    if matched >= len(required) * 0.6:
                        conf = min(0.85, 0.5 + 0.15 * matched)
                        return scene, conf

        return "generic", 0.0

    # \u2500\u2500\u2500 Structural Typographical Emphasis Evaluations Contexts \u2500\u2500\u2500

    def _visual_feature_boost(self, base_result) -> Tuple[str, float]:
        """
        Elevate confidence metrics capturing stylistic context markers identically replicating human heuristic boundaries.

        E.g., if a massive boldened headline clearly declares explicitly "Bank" structurally alongside explicitly formatting natively evaluated optimally seamlessly practically logically \u2192 Inherently overrides conflicting matrix geometries seamlessly effectively directly cleanly intrinsically safely fundamentally smoothly rationally flawlessly securely appropriately exactly elegantly smartly organically exactly securely exactly strictly completely seamlessly intuitively dynamically structurally exactly ideally functionally accurately naturally successfully efficiently explicitly.
        """
        if base_result is None:
            return "generic", 0.0

        boost_scene = "generic"
        boost_conf = 0.0

        # Scene-specific visual keyword sets
        visual_keywords = {
            "bank_statement": ["银行", "Bank", "流水", "Statement", "交易明细"],
            "invoice": ["Invoice", "Invoice", "增值税"],
            "financial_report": ["审计", "报 table", "Annual Report"],
            "credit_report": ["征信", "信用报告"],
            "contract": ["Contract", "Contract", "Protocol"],
        }

        for block in base_result.all_blocks:
            if block.block_type not in ("title", "text"):
                continue
            for span in block.spans:
                style = span.style
                text = span.text

                # Explicit typographic weight variables amplifying target signatures
                is_prominent = (
                    (style.is_bold and style.font_size >= 12)
                    or style.font_size >= 16
                )
                if not is_prominent:
                    continue

                for scene, keywords in visual_keywords.items():
                    for kw in keywords:
                        if kw in text:
                            conf = 0.6
                            if style.is_bold and style.font_size >= 14:
                                conf = 0.75
                            if conf > boost_conf:
                                boost_scene = scene
                                boost_conf = conf

        return boost_scene, boost_conf

    # \u2500\u2500\u2500 Telemetry Semantic Entity Corroborations Constraints \u2500\u2500\u2500

    def _detect_from_entities(self, entities: Dict[str, str]) -> Tuple[str, float]:
        """Synthesize overlapping parameter variables categorically."""
        if not entities:
            return "generic", 0.0

        keys = set(entities.keys())

        # Implicit Statement Parameter Anchors
        bank_keys = {
            "Account name", "Account number", "Card number", "Bank name",
            "Query period", "Account name称", "Customer name", "打印Period"
        }
        matched = len(keys & bank_keys)
        if matched >= 2:
            return "bank_statement", min(0.85, 0.5 + 0.15 * matched)

        # Invoice Variable Boundaries
        invoice_keys = {
            "Invoice代码", "Invoice number", "Buyer", "Seller", "Tax amount"
        }
        matched = len(keys & invoice_keys)
        if matched >= 2:
            return "invoice", min(0.85, 0.5 + 0.15 * matched)

        return "generic", 0.0


