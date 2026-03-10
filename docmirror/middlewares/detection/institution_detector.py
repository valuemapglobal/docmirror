"""
L2 机构识别中间件 (Institution Detector)
=========================================

在 scene=bank_statement 时，从全文/首页文本识别具体银行 (institution id)。
策略：先 identification_keywords 精确匹配，再按银行全称（按名称长度降序）匹配。
配置来自 configs/institution_registry.yaml。
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import BaseMiddleware
from ...models.enhanced import EnhancedResult

logger = logging.getLogger(__name__)


def _load_registry() -> Dict[str, Dict[str, Any]]:
    """加载机构注册表。"""
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
    截取「表头区域」：在常见交易列关键字之前的内容，用于机构识别。
    避免在交易流水条目中误匹配对手方银行名。
    """
    column_keywords = [
        "交易日期", "凭证类型", "交易时间", "序号",
        "交易明细", "记账日期", "交易类型",
    ]
    cut_pos = len(full_text)
    for kw in column_keywords:
        idx = full_text.find(kw)
        if 15 < idx < cut_pos:
            cut_pos = idx
    return full_text[: min(cut_pos, max_chars)]


def detect_institution(full_text: str, registry: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    L2 机构识别：返回 institution_id 或 None。
    先按 identification_keywords 匹配，再按银行全称（按名称长度降序）匹配。
    """
    if not full_text or not registry:
        return None
    normalized = unicodedata.normalize("NFKC", full_text)
    header_text = _extract_header_area(normalized)

    # Pass 1: 特征指纹
    for inst_id, info in registry.items():
        keywords = info.get("identification_keywords") or []
        if keywords and all(kw in normalized for kw in keywords):
            return inst_id

    # Pass 2: 名称全称（按长度降序）
    sorted_banks = sorted(
        registry.items(),
        key=lambda kv: len(kv[1].get("name", "")),
        reverse=True,
    )
    for inst_id, info in sorted_banks:
        name = info.get("name", "")
        if name and name in header_text:
            return inst_id

    # Pass 3: 别名匹配 (缩写/俗称, 如 "工行"→icbc)
    for inst_id, info in registry.items():
        for alias in info.get("aliases", []):
            if alias and alias in header_text:
                return inst_id

    return None


class InstitutionDetector(BaseMiddleware):
    """
    L2 机构识别：在 bank_statement 场景下输出 institution。
    将结果写入 result.enhanced_data["institution"]，供 ColumnMapper 等使用。
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
            logger.info(f"[InstitutionDetector] → institution={institution}")
        else:
            logger.debug("[InstitutionDetector] → no institution matched")
        return result
