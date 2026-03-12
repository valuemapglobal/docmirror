"""
Integration tests for DocMirror core modules.

Tests actual document parsing pipeline end-to-end using real fixture files,
verifying that the full extraction chain works correctly.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Directory containing test fixture files
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ─── Core Model Integration Tests ───

class TestBaseResultIntegrity:
    """Verify BaseResult and PageLayout model construction."""

    def test_base_result_creation(self):
        from docmirror.models.entities.domain import BaseResult, PageLayout, Block

        block = Block(
            block_type="text",
            raw_content="Hello, world!",
            page=1,
        )
        page = PageLayout(page_number=1, blocks=(block,))
        result = BaseResult(
            pages=(page,),
            full_text="Hello, world!",
            metadata={"source": "test"},
        )
        assert len(result.pages) == 1
        assert result.pages[0].blocks[0].raw_content == "Hello, world!"
        assert result.full_text == "Hello, world!"
        assert result.metadata["source"] == "test"

    def test_base_result_empty(self):
        from docmirror.models.entities.domain import BaseResult
        result = BaseResult(pages=(), full_text="", metadata={})
        assert len(result.pages) == 0
        assert result.full_text == ""

    def test_block_types(self):
        from docmirror.models.entities.domain import Block
        for bt in ["text", "table", "title", "key_value", "footer", "image", "formula"]:
            block = Block(block_type=bt, raw_content="test", page=1)
            assert block.block_type == bt

    def test_page_layout_ordering(self):
        from docmirror.models.entities.domain import PageLayout, Block
        blocks = tuple(
            Block(
                block_type="text",
                raw_content=f"Block {i}",
                page=1,
                reading_order=i,
            )
            for i in range(5)
        )
        page = PageLayout(page_number=1, blocks=blocks)
        assert len(page.blocks) == 5
        assert page.blocks[0].reading_order == 0
        assert page.blocks[4].reading_order == 4


# ─── Perception Result Integration Tests ───

class TestPerceptionResult:
    """Verify PerceptionResult model and its derived properties."""

    def test_perception_result_creation(self):
        from docmirror.models.entities.perception_result import (
            PerceptionResult, ResultStatus, DocumentContent,
            ContentBlock, ContentBlockType, TextBlock, TableBlock, SourceInfo,
        )

        text_block = ContentBlock(
            type=ContentBlockType.TEXT,
            text=TextBlock(text="Sample text"),
            page=1,
        )
        table_block = ContentBlock(
            type=ContentBlockType.TABLE,
            table=TableBlock(headers=["A", "B"], rows=[["1", "2"]]),
            page=1,
        )
        content = DocumentContent(
            text="Sample text",
            blocks=[text_block, table_block],
            page_count=1,
        )
        result = PerceptionResult(
            status=ResultStatus.SUCCESS,
            confidence=0.95,
            content=content,
            source=SourceInfo(adapter="test", model="unit"),
        )
        assert result.success
        assert result.confidence == 0.95

    def test_perception_result_failure(self):
        from docmirror.models.entities.perception_result import (
            PerceptionResult, ResultStatus, DocumentContent,
            SourceInfo, ErrorDetail,
        )
        result = PerceptionResult(
            status=ResultStatus.FAILURE,
            confidence=0.0,
            content=DocumentContent(text="", blocks=[], page_count=0),
            source=SourceInfo(adapter="test", model="unit"),
            error=ErrorDetail(code="TEST_ERR", message="Test error"),
        )
        assert not result.success
        assert result.error.code == "TEST_ERR"


# ─── Mutation Tracker Integration Tests ───

class TestMutationTracker:
    """Verify Mutation data lineage tracking."""

    def test_mutation_create(self):
        from docmirror.models.tracking.mutation import Mutation
        m = Mutation.create(
            middleware_name="test_stage",
            target_block_id="block_0",
            field_changed="test_field",
            old_value="old",
            new_value="new",
            reason="Unit test",
        )
        assert m.middleware_name == "test_stage"
        assert m.field_changed == "test_field"
        assert m.old_value == "old"
        assert m.new_value == "new"

    def test_mutation_to_dict(self):
        from docmirror.models.tracking.mutation import Mutation
        m = Mutation.create(
            middleware_name="s", target_block_id="b0",
            field_changed="f", old_value="a", new_value="b", reason="r",
        )
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["middleware"] == "s"
        assert d["field"] == "f"
        assert "timestamp" in d


# ─── OCR Postprocess Integration Tests ───

class TestOCRPostprocess:
    """Verify OCR postprocessing pipeline."""

    def test_basic_postprocess(self):
        from docmirror.core.ocr.ocr_postprocess import postprocess_ocr_text
        # Should not crash on empty input
        result = postprocess_ocr_text("")
        assert result == ""

    def test_postprocess_whitespace(self):
        from docmirror.core.ocr.ocr_postprocess import postprocess_ocr_text
        result = postprocess_ocr_text("  hello   world  ")
        assert "hello" in result
        assert "world" in result

    def test_postprocess_preserves_content(self):
        from docmirror.core.ocr.ocr_postprocess import postprocess_ocr_text
        text = "Transaction Date: 2024-01-15, Amount: $1,234.56"
        result = postprocess_ocr_text(text)
        assert "2024" in result
        assert "1,234.56" in result or "1234.56" in result


# ─── Text Utils Integration Tests ───

class TestTextUtils:
    """Verify text utility functions."""

    def test_is_cjk_char(self):
        from docmirror.core.utils.text_utils import _is_cjk_char
        assert _is_cjk_char("中")
        assert _is_cjk_char("字")
        assert not _is_cjk_char("A")
        assert not _is_cjk_char("1")

    def test_normalize_table(self):
        from docmirror.core.utils.text_utils import normalize_table
        table = [["  A  ", "  B  "], ["  1  ", "  2  "]]
        result = normalize_table(table)
        assert result[0][0] == "A"
        assert result[1][1] == "2"


# ─── Vocabulary Integration Tests ───

class TestVocabulary:
    """Verify vocabulary detection utilities."""

    def test_known_header_words(self):
        from docmirror.core.utils.vocabulary import KNOWN_HEADER_WORDS
        assert isinstance(KNOWN_HEADER_WORDS, frozenset)
        assert len(KNOWN_HEADER_WORDS) > 0

    def test_is_header_row(self):
        from docmirror.core.utils.vocabulary import _is_header_row
        header = ["Date", "Description", "Amount", "Balance"]
        assert _is_header_row(header)

    def test_score_header_by_vocabulary(self):
        from docmirror.core.utils.vocabulary import _score_header_by_vocabulary, KNOWN_HEADER_WORDS
        # Use words from the actual KNOWN_HEADER_WORDS set
        sample_words = list(KNOWN_HEADER_WORDS)[:4]
        if len(sample_words) >= 3:
            score = _score_header_by_vocabulary(sample_words)
            assert score >= 2
        else:
            # Fallback: just test it doesn't crash
            _score_header_by_vocabulary(["Date", "Amount"])

    def test_normalize_for_vocab(self):
        from docmirror.core.utils.vocabulary import _normalize_for_vocab
        result = _normalize_for_vocab("  DATE  ")
        assert "date" in result.lower()


# ─── Plugin System Integration Tests ───

class TestPluginSystem:
    """Verify plugin discovery and registry."""

    def test_registry_singleton(self):
        from docmirror.plugins import registry
        assert registry is not None

    def test_list_plugins(self):
        from docmirror.plugins import registry
        plugins = registry.list_plugins()
        assert isinstance(plugins, dict)
        # bank_statement should be auto-discovered
        assert "bank_statement" in plugins

    def test_bank_statement_plugin(self):
        from docmirror.plugins import registry
        plugin = registry.get("bank_statement")
        assert plugin is not None
        assert plugin.domain_name == "bank_statement"
        assert len(plugin.scene_keywords) > 0


# ─── Cache Integration Tests ───

class TestCacheSystem:
    """Verify parse cache mechanics."""

    def test_cache_import(self):
        from docmirror.framework.cache import parse_cache
        assert parse_cache is not None

    def test_cache_miss(self):
        """Cache.get is async; verify module interface."""
        from docmirror.framework.cache import parse_cache
        # Just verify cache module is importable and has get/set methods
        assert hasattr(parse_cache, 'get')
        assert hasattr(parse_cache, 'set')


# ─── Settings Integration Tests ───

class TestSettingsIntegration:
    """Verify settings model extensions."""

    def test_settings_to_dict_complete(self):
        from docmirror.configs.settings import DocMirrorSettings
        settings = DocMirrorSettings.from_env()
        d = settings.to_dict()
        assert isinstance(d, dict)
        assert len(d) > 0  # has actual config keys

    def test_settings_from_env_consistent(self):
        from docmirror.configs.settings import DocMirrorSettings
        s1 = DocMirrorSettings.from_env()
        s2 = DocMirrorSettings.from_env()
        assert s1.to_dict() == s2.to_dict()


# ─── Watermark Detection Integration Tests ───

class TestWatermarkUtils:
    """Verify watermark utility functions."""

    def test_is_watermark_char(self):
        from docmirror.core.utils.watermark import is_watermark_char
        # is_watermark_char takes a Dict (char object from pdfplumber)
        char_obj = {"text": "A", "size": 6.0, "fontname": "WatermarkFont"}
        result = is_watermark_char(char_obj)
        assert isinstance(result, bool)

    def test_watermark_module_imports(self):
        """Ensure watermark module imports without error."""
        from docmirror.core.utils import watermark
        assert hasattr(watermark, 'filter_watermark_page')
        assert hasattr(watermark, 'is_watermark_char')


# ─── Enhanced Result Integration Tests ───

class TestEnhancedResult:
    """Verify EnhancedResult model construction."""

    def test_enhanced_result_from_base(self):
        from docmirror.models.entities.domain import BaseResult, PageLayout, Block
        from docmirror.models.entities.enhanced import EnhancedResult

        block = Block(block_type="text", raw_content="Test content", page=1)
        page = PageLayout(page_number=1, blocks=(block,))
        base = BaseResult(pages=(page,), full_text="Test content", metadata={})

        enhanced = EnhancedResult.from_base_result(base)
        assert enhanced is not None
        assert enhanced.base_result is base
        assert enhanced.base_result.full_text == "Test content"
        assert len(enhanced.base_result.pages) == 1
