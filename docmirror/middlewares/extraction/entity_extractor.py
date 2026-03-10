"""
实体提取中间件 (Entity Extractor Middleware)
=============================================

从 extractor.py 的 _extract_entities_from_text() 抽离的业务逻辑层。
负责从文档全文和已提取的 KV blocks 中识别银行名、户名、账号、期间等关键实体。

设计原则:
    - 配置驱动: 实体正则模式未来可通过 hints.yaml 扩展
    - 职责分离: CoreExtractor 只做物理提取, 实体识别属于业务增强
    - 可插拔: 作为标准中间件, 可在任何 enhance_mode 下自由装卸
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..base import BaseMiddleware
from ...models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)

# 字段标签过滤 (避免把字段名误识别为户名)
_FIELD_LABELS = re.compile(
    r'客户号|账号|账单|币种|标志|类型|日期|金额|余额|合计|页数|凭证|编号|条件|证件|摘要|流水|交易|期末|期初|汇总'
)


class EntityExtractor(BaseMiddleware):
    """
    实体提取中间件 — 识别文档中的关键业务实体。

    从文档全文和已提取的 key_value blocks 中识别:
      - 银行名 (bank_name)
      - 户名
      - 账号
      - 查询期间
      - 打印日期
      - 币种
    """

    def process(self, result: EnhancedResult) -> EnhancedResult:
        """执行实体提取。"""
        base = result.base_result
        full_text = base.full_text or ""
        pages = base.pages

        entities: Dict[str, str] = {}

        # 1. 从已提取的 key_value blocks 收集
        for page in pages:
            for block in page.blocks:
                if block.block_type == "key_value" and isinstance(block.raw_content, dict):
                    entities.update(block.raw_content)

        # 2. 正则兜底: 从首页文本提取
        first_page_text = full_text[:500] if full_text else ""

        self._extract_bank_name(entities, first_page_text)
        self._extract_account_holder(entities, first_page_text, pages)
        self._extract_account_number(entities, first_page_text, pages)
        self._extract_period(entities, first_page_text)
        self._extract_print_date(entities, first_page_text)
        self._extract_currency(entities, first_page_text)

        # 写入 enhanced_data
        result.enhanced_data["extracted_entities"] = entities
        logger.info(f"[DocMirror] EntityExtractor: {list(entities.keys())}")

        return result

    def _extract_bank_name(self, entities: Dict, text: str) -> None:
        if "bank_name" in entities or "开户行" in entities:
            return
        bank_patterns = [
            r'(中国[建工农交]设?银行)',
            r'(招商银行|兴业银行|浦发银行|民生银行|中信银行|光大银行|华夏银行|平安银行)',
            r'(中[国]?银行)',
            r'([\u4e00-\u9fa5]{2,8}银行)',
        ]
        for pat in bank_patterns:
            m = re.search(pat, text)
            if m:
                entities["bank_name"] = m.group(1)
                break

    def _extract_account_holder(self, entities: Dict, text: str, pages) -> None:
        if any(k in entities for k in ("户名", "本方户名", "客户姓名", "客户名称")):
            return
        for pat in [
            r'(?:本方)?户名[：:]\s*(.+?)(?:\n|$)',
            r'(?:账户名称|客户名称|客户姓名|开户人|持卡人)[：:]\s*(.+?)(?:\n|$)',
            r'(?:账户名称|客户名称|客户姓名)\n(.+?)(?:\n|$)',
            r'戶名[：:]?\s*(.+?)(?:\n|$)',
            r'Account\s*Name[：:]?\s*(.+?)(?:\n|$)',
        ]:
            m = re.search(pat, text)
            if m:
                val = m.group(1).strip()
                if (val and 2 <= len(val) <= 30
                        and not val.isdigit()
                        and not re.match(r'^\d{4}[-/]\d{2}[-/]\d{2}', val)
                        and not _FIELD_LABELS.match(val)):
                    entities["户名"] = val
                    return

        # 表格回溯
        _name_keywords = ["户名", "本方户名", "客户名称", "客户姓名", "账户名称", "开户人"]
        for page in pages:
            for block in page.blocks:
                if block.block_type == "table" and isinstance(block.raw_content, list):
                    tbl_headers = block.raw_content[0] if block.raw_content else []
                    for i, h in enumerate(tbl_headers):
                        if h and any(kw in str(h) for kw in _name_keywords):
                            if len(block.raw_content) > 1 and i < len(block.raw_content[1]):
                                val = str(block.raw_content[1][i]).strip()
                                if (val and 2 <= len(val) <= 30
                                        and not val.isdigit()
                                        and not re.match(r'^\d{4}[-/]\d{2}[-/]\d{2}', val)):
                                    entities["户名"] = val
                                    return

    def _extract_account_number(self, entities: Dict, text: str, pages) -> None:
        if "账号" in entities:
            return
        for pat in [
            r'账\s*号[：:]\s*(\d{10,25})',
            r'卡\s*号[：:]\s*(\d{10,25})',
            r'Account\s*(?:No\.?|Number)[：:]?\s*(\d{10,25})',
            r'账戶[：:]?\s*(\d{10,25})',
        ]:
            m = re.search(pat, text)
            if m:
                entities["账号"] = m.group(1).strip()
                return
        # 表格回溯
        for page in pages:
            for block in page.blocks:
                if block.block_type == "table" and isinstance(block.raw_content, list):
                    headers = block.raw_content[0] if block.raw_content else []
                    for i, h in enumerate(headers):
                        if h and "账号" in str(h):
                            if len(block.raw_content) > 1:
                                val = str(block.raw_content[1][i]).strip()
                                if val and len(val) >= 10:
                                    entities["账号"] = val
                                    return
                    break

    def _extract_period(self, entities: Dict, text: str) -> None:
        if "查询期间" in entities or "期间" in entities:
            return
        for pat in [
            r'(?:查询|交易|账单)期间[：:]\s*(.+?)(?:\n|$)',
            r'(\d{4}年\d{1,2}月\d{1,2}日?\s*[-至到]\s*\d{4}年\d{1,2}月\d{1,2}日?)',
        ]:
            m = re.search(pat, text)
            if m:
                entities["查询期间"] = m.group(1).strip()
                return

    def _extract_print_date(self, entities: Dict, text: str) -> None:
        if "打印日期" in entities:
            return
        m = re.search(r'打印日期[：:]\s*(\d{4}年\d{1,2}月\d{1,2}日)', text)
        if m:
            entities["打印日期"] = m.group(1).strip()

    def _extract_currency(self, entities: Dict, text: str) -> None:
        if "币种" in entities:
            return
        if "人民币" in text:
            entities["币种"] = "CNY"
