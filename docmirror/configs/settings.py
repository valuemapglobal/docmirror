"""
DocMirror Global Settings
=========================

Centralized system-level configuration. Override defaults via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class DocMirrorSettings:
    """DocMirror global configuration."""

    # Default enhancement mode
    default_enhance_mode: str = "standard"

    # LLM settings
    enable_llm: bool = False
    llm_model: str = "qwen-vl-max"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    # Performance
    max_pages: int = 200
    ocr_dpi: int = 150
    ocr_retry_dpi: int = 300
    ocr_language: str = "auto"  # "auto" = auto-detect, or specify e.g. "zh", "en"

    # Validation
    validator_pass_threshold: float = 0.7

    # Logging
    log_level: str = "INFO"

    # Pipeline strategy
    fail_strategy: str = "skip"  # "skip" | "abort"

    # Model paths (optional, falls back to rule-based when None)
    layout_model_path: Optional[str] = None        # DocLayout-YOLO ONNX path
    reading_order_model_path: Optional[str] = None  # LayoutReader ONNX path
    formula_model_path: Optional[str] = None        # Pix2Tex / UniMERNet ONNX path

    # Model inference parameters
    model_render_dpi: int = 200                     # DocLayout-YOLO page render DPI

    @classmethod
    def from_env(cls) -> DocMirrorSettings:
        """Load configuration from environment variables."""
        return cls(
            default_enhance_mode=os.getenv("DOCMIRROR_ENHANCE_MODE", "standard"),
            enable_llm=os.getenv("DOCMIRROR_ENABLE_LLM", "false").lower() == "true",
            llm_model=os.getenv("DOCMIRROR_LLM_MODEL", "qwen-vl-max"),
            max_pages=int(os.getenv("DOCMIRROR_MAX_PAGES", "200")),
            validator_pass_threshold=float(os.getenv("DOCMIRROR_VALIDATOR_THRESHOLD", "0.7")),
            log_level=os.getenv("DOCMIRROR_LOG_LEVEL", "INFO"),
            fail_strategy=os.getenv("DOCMIRROR_FAIL_STRATEGY", "skip"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for Orchestrator config injection."""
        return {
            "enhance_mode": self.default_enhance_mode,
            "SceneDetector": {"enable_llm": self.enable_llm},
            "ColumnMapper": {},
            "Validator": {"pass_threshold": self.validator_pass_threshold},
            "Repairer": {"enable_llm": self.enable_llm},
        }


# Global default settings instance
default_settings = DocMirrorSettings.from_env()

