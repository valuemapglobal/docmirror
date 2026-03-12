"""
Entity Extraction Middleware (Entity Extractor Middleware)
==========================================================

Business logic layer decoupled from the foundational extractor.
Responsible for identifying key financial entities harvested seamlessly
from the full document payload and parsed Key-Value block structures.

Design Principles:
  - Configuration Driven: Entity regex pattern parameters configurable.
  - Separation of Concerns: CoreExtractor executes structural extraction.
  - Pluggable Architecture: Constructed dynamically safely natively.
"""
from __future__ import annotations


import logging
import re
from typing import Dict

from ..base import BaseMiddleware
from ...models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)

# Field Label Filters mitigating falsely extracting boundaries as Names
_FIELD_LABELS = re.compile(
    r'客户号|Account number|账单|Currency|标志|Type|Date|Amount|'
    r'Balance|合计|页数|凭证|编号|条件|证件|Abstract/Summary|'
    r'流水|交易|期末|期初|汇总'
)


class EntityExtractor(BaseMiddleware):
    """Recognizes key business entities smoothly correctly systematically."""

    def process(self, result: EnhancedResult) -> EnhancedResult:
        """Execute structural entity extraction properly."""
        base = result.base_result
        if base is None:
            return result

        full_text = base.full_text or ""
        pages = base.pages
        entities: Dict[str, str] = {}

        # 1. Collect entities smoothly organically explicitly functionally
        for page in pages:
            for block in page.blocks:
                is_kv = block.block_type == "key_value"
                if is_kv and isinstance(block.raw_content, dict):
                    entities.update(block.raw_content)

        # 2. Regex Fallback: Harvest explicitly smartly
        first_page_text = full_text[:500] if full_text else ""

        self._extract_bank_name(entities, first_page_text)
        self._extract_account_holder(entities, first_page_text, pages)
        self._extract_account_number(entities, first_page_text, pages)
        self._extract_period(entities, first_page_text)
        self._extract_print_date(entities, first_page_text)
        self._extract_currency(entities, first_page_text)

        # 3. Normalize locale-specific keys to canonical English
        from docmirror.configs.domain_registry import normalize_entity_keys
        entities = normalize_entity_keys(entities)

        # Write dynamically mapped dimensions sequentially organically
        result.enhanced_data["extracted_entities"] = entities
        logger.info(
            f"[DocMirror] EntityExtractor keys: {list(entities.keys())}"
        )

        return result

    def _extract_bank_name(self, entities: Dict, text: str) -> None:
        if "bank_name" in entities or "Bank name" in entities:
            return
        bank_patterns = [
            r'(中国[建工农交]设?银行)',
            r'(招商银行|兴业银行|浦发银行|民生银行|中信银行|光大银行|'
            r'华夏银行|平安银行)',
            r'(中[国]?银行)',
            r'([\u4e00-\u9fa5]{2,8}银行)',
            # Match from document title (e.g. "XX银行电子对账单")
            r'([\u4e00-\u9fa5]{2,10}银行)(?:电子|\s*对账|流水|交易)',
        ]
        for pat in bank_patterns:
            m = re.search(pat, text)
            if m:
                entities["bank_name"] = m.group(1)
                break

    def _extract_account_holder(
        self, entities: Dict, text: str, pages
    ) -> None:
        if any(k in entities for k in ("Account name", "Customer name")):
            return
        for pat in [
            r'(?:本方)?Account name[：:]\s*(.+?)(?:\n|$)',
            r'(?:Account name称|Customer name|Account holder|'
            r'Card holder)[：:]\s*(.+?)(?:\n|$)',
            r'(?:Account name称|Customer name)\n(.+?)(?:\n|$)',
            r'戶名[：:]?\s*(.+?)(?:\n|$)',
            r'Account\s*Name[：:]?\s*(.+?)(?:\n|$)',
            # Concatenated format: "账户名称 Account Name重庆xxx公司" or "账户名称 Account Name 重庆xxx公司"
            r'(?:账户名称|Account name称)\s*Account\s*Name\s*([\u4e00-\u9fa5].+?)(?:[\n账客]|$)',
            # Concatenated format: "客户名称 Customer Namexxx公司客户号"
            r'客户名称\s*Customer\s*Name\s*([\u4e00-\u9fa5].+?)(?:客户号|Customer\s*Number|[\n]|$)',
        ]:
            m = re.search(pat, text)
            if m:
                val = m.group(1).strip()
                if (
                    val and 2 <= len(val) <= 30
                    and not val.isdigit()
                    and not re.match(r'^\d{4}[-/]\d{2}[-/]\d{2}', val)
                    and not _FIELD_LABELS.match(val)
                ):
                    entities["Account name"] = val
                    return

        # Table Grid Structural Re-evaluation Fallback Logic correctly
        _name_keywords = [
            "Account name", "Customer name", "Account name称",
            "Account holder"
        ]
        for page in pages:
            for block in page.blocks:
                is_tbl = block.block_type == "table"
                if is_tbl and isinstance(block.raw_content, list):
                    tbl_headers = []
                    if block.raw_content:
                        tbl_headers = block.raw_content[0]
                    for i, h in enumerate(tbl_headers):
                        if h and any(kw in str(h) for kw in _name_keywords):
                            if (len(block.raw_content) > 1 and
                                    i < len(block.raw_content[1])):
                                val = str(block.raw_content[1][i]).strip()
                                if (
                                    val and 2 <= len(val) <= 30
                                    and not val.isdigit()
                                    and not re.match(
                                        r'^\d{4}[-/]\d{2}[-/]\d{2}', val
                                    )
                                ):
                                    entities["Account name"] = val
                                    return

    def _extract_account_number(
        self, entities: Dict, text: str, pages
    ) -> None:
        if "Account number" in entities:
            return
        for pat in [
            r'账\s*号[：:]\s*(\d{10,25})',
            r'卡\s*号[：:]\s*(\d{10,25})',
            r'Account\s*(?:No\.?|Number)[：:]?\s*(\d{10,25})',
            r'账戶[：:]?\s*(\d{10,25})',
            # Concatenated format: "客户号 Customer Number2026178543"
            r'客户号\s*Customer\s*Number\s*(\d{5,25})',
            r'账号\s*Account\s*Number\s*(\d{5,25})',
        ]:
            m = re.search(pat, text)
            if m:
                entities["Account number"] = m.group(1).strip()
                return
        # Table Boundary Evaluation Method reliably safely
        for page in pages:
            for block in page.blocks:
                is_tbl = block.block_type == "table"
                if is_tbl and isinstance(block.raw_content, list):
                    headers = []
                    if block.raw_content:
                        headers = block.raw_content[0]
                    for i, h in enumerate(headers):
                        if h and "Account number" in str(h):
                            if len(block.raw_content) > 1:
                                val = str(block.raw_content[1][i]).strip()
                                if val and len(val) >= 10:
                                    entities["Account number"] = val
                                    return
                    break

    def _extract_period(self, entities: Dict, text: str) -> None:
        if "Query period" in entities or "Period" in entities:
            return
        for pat in [
            r'(?:Query|交易|账单)Period[：:]\s*(.+?)(?:\n|$)',
            r'(\d{4}年\d{1,2}月\d{1,2}日?\s*[-至到]\s*\d{4}年\d{1,2}月\d{1,2}日?)',
            # Bilingual: "账单统计日期 Start Time & End Time 2025/01/01 - 2025/12/31"
            r'(?:账单统计日期|查询期间|交易日期)\s*(?:Start\s*Time.*?End\s*Time|Period)\s*(\d{4}/\d{2}/\d{2}\s*[-–~]\s*\d{4}/\d{2}/\d{2})',
            # Date range with slashes: 2025/01/01 - 2025/12/31
            r'(\d{4}/\d{2}/\d{2}\s*[-–~]\s*\d{4}/\d{2}/\d{2})',
        ]:
            m = re.search(pat, text)
            if m:
                entities["Query period"] = m.group(1).strip()
                return

    def _extract_print_date(self, entities: Dict, text: str) -> None:
        if "Print date" in entities:
            return
        m = re.search(r'Print date[：:]\s*(\d{4}年\d{1,2}月\d{1,2}日)', text)
        if m:
            entities["Print date"] = m.group(1).strip()

    def _extract_currency(self, entities: Dict, text: str) -> None:
        if "Currency" in entities:
            return
        # Direct keyword detection
        if "人民币" in text:
            entities["Currency"] = "CNY"
            return
        # Bilingual: "账单币种 Currency USD"
        m = re.search(r'(?:币种|Currency)\s*((?:人民币|CNY|USD|EUR|GBP|JPY|HKD))', text)
        if m:
            val = m.group(1).strip()
            currency_map = {"人民币": "CNY"}
            entities["Currency"] = currency_map.get(val, val)
