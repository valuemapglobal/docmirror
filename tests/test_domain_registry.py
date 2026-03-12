"""
Unit tests for domain_registry — key normalization and identity resolution.
"""

import pytest
from docmirror.configs.domain_registry import (
    KEY_SYNONYMS,
    normalize_entity_keys,
    resolve_identity,
)


class TestKeySynonymsLoading:
    """Verify YAML-based key synonyms are loaded correctly."""

    def test_synonyms_loaded(self):
        """KEY_SYNONYMS should be populated from key_synonyms.yaml."""
        assert len(KEY_SYNONYMS) > 0

    def test_chinese_bank_keys_present(self):
        """Core Chinese bank statement keys should be in the synonym table."""
        assert KEY_SYNONYMS["账号"] == "Account number"
        assert KEY_SYNONYMS["账户名"] == "Account name"
        assert KEY_SYNONYMS["币种"] == "Currency"
        assert KEY_SYNONYMS["交易时间"] == "Query period"

    def test_japanese_keys_present(self):
        """Japanese keys should also be loaded."""
        assert KEY_SYNONYMS["口座番号"] == "Account number"
        assert KEY_SYNONYMS["口座名義"] == "Account name"


class TestNormalizeEntityKeys:
    """Verify the normalize_entity_keys function."""

    def test_chinese_keys_normalized(self):
        """Chinese keys should be translated to canonical English."""
        raw = {"账号": "651204680300015", "账户名": "重庆恒腾科技有限公司", "币种": "人民币"}
        result = normalize_entity_keys(raw)
        assert result["Account number"] == "651204680300015"
        assert result["Account name"] == "重庆恒腾科技有限公司"
        assert result["Currency"] == "人民币"

    def test_english_keys_pass_through(self):
        """English keys not in the synonym table should pass through."""
        raw = {"Account number": "123456", "Custom field": "value"}
        result = normalize_entity_keys(raw)
        assert result["Account number"] == "123456"
        assert result["Custom field"] == "value"

    def test_no_overwrite_existing_canonical(self):
        """If a canonical key already exists, synonym should not overwrite."""
        raw = {"Account number": "ORIGINAL_VALUE", "账号": "SYNONYM_VALUE"}
        result = normalize_entity_keys(raw)
        assert result["Account number"] == "ORIGINAL_VALUE"

    def test_unknown_keys_preserved(self):
        """Keys not in the synonym table should be preserved as-is."""
        raw = {"unknown_key": "some_value", "另一个字段": "另一个值"}
        result = normalize_entity_keys(raw)
        assert result["unknown_key"] == "some_value"
        assert result["另一个字段"] == "另一个值"

    def test_empty_dict_returns_empty(self):
        """Empty input should return empty output."""
        assert normalize_entity_keys({}) == {}

    def test_original_dict_not_mutated(self):
        """The original dict should not be modified."""
        raw = {"账号": "12345"}
        original_copy = dict(raw)
        normalize_entity_keys(raw)
        assert raw == original_copy


class TestResolveIdentityWithNormalization:
    """Verify resolve_identity works with multilingual entity keys."""

    def test_chinese_bank_statement_identity(self):
        """Chinese-key entities should produce populated identity fields."""
        entities = {
            "账号": "651204680300015",
            "账户名": "重庆恒腾科技有限公司",
            "币种": "人民币",
            "交易时间": "2025-07-01 至 2025-12-31",
        }
        identity = resolve_identity("bank_statement", entities)
        assert identity["document_type"] == "bank_statement"
        assert identity["account_number"] == "651204680300015"
        assert identity["account_holder"] == "重庆恒腾科技有限公司"
        assert identity["currency"] == "人民币"
        assert identity["query_period"] == "2025-07-01 至 2025-12-31"

    def test_english_bank_statement_identity(self):
        """English-key entities should still work as before (no regression)."""
        entities = {
            "Account number": "1234567890",
            "Account name": "John Doe",
            "Currency": "USD",
            "bank_name": "HSBC",
        }
        identity = resolve_identity("bank_statement", entities)
        assert identity["account_number"] == "1234567890"
        assert identity["account_holder"] == "John Doe"
        assert identity["currency"] == "USD"
        assert identity["institution"] == "HSBC"

    def test_mixed_chinese_english_entities(self):
        """Mixed-language entities should resolve correctly."""
        entities = {
            "Account number": "999888777",
            "账户名": "混合测试公司",
            "Currency": "CNY",
        }
        identity = resolve_identity("bank_statement", entities)
        assert identity["account_number"] == "999888777"
        assert identity["account_holder"] == "混合测试公司"
        assert identity["currency"] == "CNY"

    def test_unknown_domain_uses_wildcard(self):
        """Unknown domains should fall back to wildcard identity fields."""
        entities = {"Title": "My Document", "Date": "2025-01-01"}
        identity = resolve_identity("unknown_type", entities)
        assert identity["document_type"] == "unknown_type"
        assert identity["title"] == "My Document"
        assert identity["date"] == "2025-01-01"

    def test_empty_entities_returns_empty_identity(self):
        """Empty entities should produce identity with empty-string values."""
        identity = resolve_identity("bank_statement", {})
        assert identity["document_type"] == "bank_statement"
        assert identity["institution"] == ""
        assert identity["account_holder"] == ""
        assert identity["account_number"] == ""
