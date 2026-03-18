# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Entity Extraction Middleware
============================

Identifies key financial entities from KV pairs, table blocks,
and full text regex patterns. Writes directly to ParseResult.entities.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

from ...models.entities.parse_result import ParseResult
from ..base import BaseMiddleware

logger = logging.getLogger(__name__)

# Field Label Filters
_FIELD_LABELS = re.compile(
    r"客户号|Account number|账单|Currency|标志|Type|Date|Amount|"
    r"Balance|合计|页数|凭证|编号|条件|证件|Abstract/Summary|"
    r"流水|交易|期末|期初|汇总"
)


class EntityExtractor(BaseMiddleware):
    """Recognizes key business entities from document content."""

    def process(self, result: ParseResult) -> ParseResult:
        """Execute structural entity extraction."""
        full_text = result.full_text or ""
        entities: dict[str, str] = {}

        # 1. Collect entities from KV pairs
        for page in result.pages:
            for kv in page.key_values:
                if kv.key:
                    entities[kv.key] = kv.value

        # 2. Regex Fallback from first page text
        first_page_text = full_text[:500] if full_text else ""

        self._extract_bank_name(entities, first_page_text)
        self._extract_account_holder(entities, first_page_text, result.pages)
        self._extract_account_number(entities, first_page_text, result.pages)
        self._extract_period(entities, first_page_text)
        self._extract_print_date(entities, first_page_text)
        self._extract_currency(entities, first_page_text)

        # 3. Normalize locale-specific keys
        from docmirror.configs.domain_registry import normalize_entity_keys

        entities = normalize_entity_keys(entities)

        # Write to ParseResult.entities
        # subject_name: try multiple keys (企业名称, 户名, 客户名称, Account name)
        for name_key in ("Account name", "企业名称", "客户名称", "户名", "Account name称", "Card holder"):
            if entities.get(name_key):
                result.entities.subject_name = entities[name_key]
                break
        if entities.get("bank_name"):
            result.entities.organization = entities["bank_name"]
        if entities.get("Account number"):
            result.entities.subject_id = entities["Account number"]
        if entities.get("Query period"):
            period = entities["Query period"]
            result.entities.domain_specific["query_period"] = period
            # Split into period_start / period_end
            self._split_period(result, period)
        if entities.get("Print date"):
            result.entities.document_date = entities["Print date"]
        if entities.get("Currency"):
            result.entities.domain_specific["currency"] = entities["Currency"]

        # Store all extracted entities in domain_specific for backward compat
        result.entities.domain_specific["extracted_entities"] = entities

        logger.info(f"[EntityExtractor] keys: {list(entities.keys())}")
        return result

    @staticmethod
    def _split_period(result: ParseResult, period: str) -> None:
        """Split period string into period_start and period_end."""
        import re

        # Pattern: 2025年01月01日-2025年03月31日
        m = re.match(
            r"(\d{4}年\d{1,2}月\d{1,2}日?)\s*[-~至到–]\s*(\d{4}年\d{1,2}月\d{1,2}日?)",
            period,
        )
        if m:
            result.entities.period_start = m.group(1)
            result.entities.period_end = m.group(2)
            return
        # Pattern: 2025/01/01-2025/03/31 or 2025-01-01~2025-03-31
        m = re.match(
            r"(\d{4}[/-]\d{2}[/-]\d{2})\s*[-~至到–]\s*(\d{4}[/-]\d{2}[/-]\d{2})",
            period,
        )
        if m:
            result.entities.period_start = m.group(1)
            result.entities.period_end = m.group(2)

    def _extract_bank_name(self, entities: dict, text: str) -> None:
        if "bank_name" in entities or "Bank name" in entities:
            return
        bank_patterns = [
            r"(中国[建工农交]设?银行)",
            r"(招商银行|兴业银行|浦发银行|民生银行|中信银行|光大银行|"
            r"华夏银行|平安银行)",
            r"(中[国]?银行)",
            r"([\u4e00-\u9fa5]{2,8}银行)",
            r"([\u4e00-\u9fa5]{2,10}银行)(?:电子|\s*对账|流水|交易)",
        ]
        for pat in bank_patterns:
            m = re.search(pat, text)
            if m:
                entities["bank_name"] = m.group(1)
                break

    def _extract_account_holder(self, entities: dict, text: str, pages) -> None:
        if any(k in entities for k in ("Account name", "Customer name")):
            return
        for pat in [
            r"(?:本方)?Account name[：:]\\s*(.+?)(?:\\n|$)",
            r"(?:Account name称|Customer name|Account holder|"
            r"Card holder)[：:]\\s*(.+?)(?:\\n|$)",
            r"(?:Account name称|Customer name)\\n(.+?)(?:\\n|$)",
            r"戶名[：:]?\\s*(.+?)(?:\\n|$)",
            r"Account\\s*Name[：:]?\\s*(.+?)(?:\\n|$)",
            r"(?:账户名称|Account name称)\\s*Account\\s*Name\\s*([\\u4e00-\\u9fa5].+?)(?:[\\n账客]|$)",
            r"客户名称\\s*Customer\\s*Name\\s*([\\u4e00-\\u9fa5].+?)(?:客户号|Customer\\s*Number|[\\n]|$)",
        ]:
            m = re.search(pat, text)
            if m:
                val = m.group(1).strip()
                if (
                    val
                    and 2 <= len(val) <= 30
                    and not val.isdigit()
                    and not re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", val)
                    and not _FIELD_LABELS.match(val)
                ):
                    entities["Account name"] = val
                    return

        # Table-based fallback
        _name_keywords = ["Account name", "Customer name", "Account name称", "Account holder"]
        for page in pages:
            for table in page.tables:
                if not table.headers:
                    continue
                for i, h in enumerate(table.headers):
                    if h and any(kw in str(h) for kw in _name_keywords):
                        if table.rows and i < len(table.rows[0].cells):
                            val = table.rows[0].cells[i].text.strip()
                            if (
                                val
                                and 2 <= len(val) <= 30
                                and not val.isdigit()
                                and not re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", val)
                            ):
                                entities["Account name"] = val
                                return

    def _extract_account_number(self, entities: dict, text: str, pages) -> None:
        if "Account number" in entities:
            return
        for pat in [
            r"账\s*号[：:]\s*(\d{10,25})",
            r"卡\s*号[：:]\s*(\d{10,25})",
            r"Account\s*(?:No\.?|Number)[：:]?\s*(\d{10,25})",
            r"账戶[：:]?\s*(\d{10,25})",
            r"客户号\s*Customer\s*Number\s*(\d{5,25})",
            r"账号\s*Account\s*Number\s*(\d{5,25})",
        ]:
            m = re.search(pat, text)
            if m:
                entities["Account number"] = m.group(1).strip()
                return

        # Table-based fallback
        for page in pages:
            for table in page.tables:
                if not table.headers:
                    continue
                for i, h in enumerate(table.headers):
                    if h and "Account number" in str(h):
                        if table.rows and i < len(table.rows[0].cells):
                            val = table.rows[0].cells[i].text.strip()
                            if val and len(val) >= 10:
                                entities["Account number"] = val
                                return

    def _extract_period(self, entities: dict, text: str) -> None:
        if "Query period" in entities or "Period" in entities:
            return
        for pat in [
            r"(?:Query|交易|账单)Period[：:]\s*(.+?)(?:\n|$)",
            r"(\d{4}年\d{1,2}月\d{1,2}日?\s*[-至到]\s*\d{4}年\d{1,2}月\d{1,2}日?)",
            r"(?:账单统计日期|查询期间|交易日期)\s*(?:Start\s*Time.*?End\s*Time|Period)\s*(\d{4}/\d{2}/\d{2}\s*[-–~]\s*\d{4}/\d{2}/\d{2})",
            r"(\d{4}/\d{2}/\d{2}\s*[-–~]\s*\d{4}/\d{2}/\d{2})",
        ]:
            m = re.search(pat, text)
            if m:
                entities["Query period"] = m.group(1).strip()
                return

    def _extract_print_date(self, entities: dict, text: str) -> None:
        if "Print date" in entities:
            return
        m = re.search(r"Print date[：:]\s*(\d{4}年\d{1,2}月\d{1,2}日)", text)
        if m:
            entities["Print date"] = m.group(1).strip()

    def _extract_currency(self, entities: dict, text: str) -> None:
        if "Currency" in entities:
            return
        if "人民币" in text:
            entities["Currency"] = "CNY"
            return
        m = re.search(r"(?:币种|Currency)\s*((?:人民币|CNY|USD|EUR|GBP|JPY|HKD))", text)
        if m:
            val = m.group(1).strip()
            currency_map = {"人民币": "CNY"}
            entities["Currency"] = currency_map.get(val, val)
