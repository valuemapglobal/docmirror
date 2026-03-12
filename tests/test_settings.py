"""
Settings configuration tests.
"""

import os
import pytest
from docmirror.configs.settings import DocMirrorSettings


class TestDocMirrorSettings:
    """Test DocMirrorSettings configuration."""

    def test_default_values(self):
        """Default settings should have sensible values."""
        settings = DocMirrorSettings()
        assert settings.default_enhance_mode == "standard"
        assert settings.max_pages == 200
        assert settings.ocr_dpi == 150
        assert settings.fail_strategy == "skip"

    def test_from_env_defaults(self):
        """from_env should use defaults when env vars not set."""
        settings = DocMirrorSettings.from_env()
        assert settings.default_enhance_mode == "standard"

    def test_from_env_override(self, monkeypatch):
        """from_env should respect DOCMIRROR_ env vars."""
        monkeypatch.setenv("DOCMIRROR_ENHANCE_MODE", "full")
        monkeypatch.setenv("DOCMIRROR_MAX_PAGES", "50")

        settings = DocMirrorSettings.from_env()
        assert settings.default_enhance_mode == "full"
        assert settings.max_pages == 50

    def test_to_dict(self):
        """to_dict should return orchestrator-compatible config."""
        settings = DocMirrorSettings()
        d = settings.to_dict()
        assert "enhance_mode" in d
        assert "SceneDetector" in d
        assert "Validator" in d

    def test_model_paths_default_none(self):
        """Model paths should default to None."""
        settings = DocMirrorSettings()
        assert settings.layout_model_path is None
        assert settings.reading_order_model_path is None
        assert settings.formula_model_path is None
