# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Scene Detection Middleware (Scene Detector)
===========================================

Three-tier progressive scenario detection:
    Tier 1 (Keyword Rules): High efficiency, covering common known scenarios.
    Tier 2 (Table Header Context): Structural feature analysis.
    Tier 3 (Visual Typographic Weights): Visual features for ambiguous cases.

Supported Scenes:
    bank_statement, invoice, tax_report, financial_report,
    credit_report, contract, generic (fallback).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

from ...models.entities.parse_result import ParseResult
from ..base import BaseMiddleware

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario Keyword Dictionary
# ═══════════════════════════════════════════════════════════════════════════════

SCENE_KEYWORDS: dict[str, list[list[str]]] = {
    "bank_statement": [
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

HEADER_FEATURES: dict[str, list[set[str]]] = {
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
    """Core Scene Classification Middleware — writes to result.entities.document_type."""

    def process(self, result: ParseResult) -> ParseResult:
        """Execute three-tier progressive scene detection."""
        full_text = result.full_text
        entities = result.kv_entities
        table_blocks = result.all_tables()

        # ─── Tier 1: Keyword Markers ───
        scene, confidence = self._tier1_keyword(full_text)
        if confidence >= 0.8:
            result.entities.document_type = scene
            result.record_mutation(
                middleware_name=self.name,
                target_block_id="document",
                field_changed="scene",
                old_value="unknown",
                new_value=scene,
                confidence=confidence,
                reason="Tier1 keyword match",
            )
            logger.info(f"[SceneDetector] Tier1 → {scene} (conf={confidence:.2f})")
            return result

        # ─── Tier 2: Structural Feature Analysis ───
        scene_t2, conf_t2 = self._tier2_header_features(table_blocks)
        if conf_t2 > confidence:
            scene, confidence = scene_t2, conf_t2

        scene_entity, conf_entity = self._detect_from_entities(entities)
        if conf_entity > confidence:
            scene, confidence = scene_entity, conf_entity

        # ─── Visual Typographic Boost ───
        visual_boost = self._visual_feature_boost(result)
        if visual_boost[1] > 0 and visual_boost[0] == scene:
            confidence = min(0.95, confidence + visual_boost[1])
        elif visual_boost[1] > confidence:
            scene, confidence = visual_boost[0], visual_boost[1]

        if confidence >= 0.6:
            result.entities.document_type = scene
            result.record_mutation(
                middleware_name=self.name,
                target_block_id="document",
                field_changed="scene",
                old_value="unknown",
                new_value=scene,
                confidence=confidence,
                reason="Tier2 feature+visual match",
            )
            logger.info(f"[SceneDetector] Tier2 → {scene} (conf={confidence:.2f})")
            return result

        final_scene = scene if confidence >= 0.3 else "generic"
        result.entities.document_type = final_scene
        result.record_mutation(
            middleware_name=self.name,
            target_block_id="document",
            field_changed="scene",
            old_value="unknown",
            new_value=final_scene,
            confidence=confidence,
            reason="tier2_visual_fallback" if confidence >= 0.3 else "low confidence fallback",
        )
        logger.info(f"[SceneDetector] Final → {final_scene} (conf={confidence:.2f})")
        return result

    # ─── Tier 1 ───

    def _tier1_keyword(self, text: str) -> tuple[str, float]:
        if not text:
            return "generic", 0.0
        best_scene = "generic"
        best_conf = 0.0
        for scene, keyword_groups in SCENE_KEYWORDS.items():
            for group in keyword_groups:
                if all(kw in text for kw in group):
                    conf = min(0.9, 0.7 + 0.1 * len(group))
                    if conf > best_conf:
                        best_scene = scene
                        best_conf = conf
        return best_scene, best_conf

    # ─── Tier 2 ───

    def _tier2_header_features(self, table_blocks) -> tuple[str, float]:
        """Tier 2: infer scene from table header column names."""
        if not table_blocks:
            return "generic", 0.0

        for table in table_blocks:
            if not table.headers:
                continue
            header_set = {str(h).strip() for h in table.headers if h}
            for scene, feature_groups in HEADER_FEATURES.items():
                for required in feature_groups:
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

    # ─── Visual Feature Boost ───

    def _visual_feature_boost(self, result: ParseResult) -> tuple[str, float]:
        """Elevate confidence from prominent text (large/bold headings)."""
        boost_scene = "generic"
        boost_conf = 0.0

        visual_keywords = {
            "bank_statement": ["银行", "Bank", "流水", "Statement", "交易明细"],
            "invoice": ["Invoice", "Invoice", "增值税"],
            "financial_report": ["审计", "报 table", "Annual Report"],
            "credit_report": ["征信", "信用报告"],
            "contract": ["Contract", "Contract", "Protocol"],
        }

        # Check text blocks for title/heading content
        for page in result.pages:
            for text_block in page.texts:
                if text_block.level.value in ("title", "h1", "h2"):
                    content = text_block.content
                    for scene, keywords in visual_keywords.items():
                        for kw in keywords:
                            if kw in content:
                                conf = 0.65
                                if text_block.level.value in ("title", "h1"):
                                    conf = 0.75
                                if conf > boost_conf:
                                    boost_scene = scene
                                    boost_conf = conf

        return boost_scene, boost_conf

    # ─── Entity Corroboration ───

    def _detect_from_entities(self, entities: dict[str, str]) -> tuple[str, float]:
        if not entities:
            return "generic", 0.0
        keys = set(entities.keys())

        bank_keys = {
            "Account name",
            "Account number",
            "Card number",
            "Bank name",
            "Query period",
            "Account name称",
            "Customer name",
            "打印Period",
        }
        matched = len(keys & bank_keys)
        if matched >= 2:
            return "bank_statement", min(0.85, 0.5 + 0.15 * matched)

        invoice_keys = {"Invoice代码", "Invoice number", "Buyer", "Seller", "Tax amount"}
        matched = len(keys & invoice_keys)
        if matched >= 2:
            return "invoice", min(0.85, 0.5 + 0.15 * matched)

        return "generic", 0.0
