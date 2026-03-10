"""
Image Adapter — Image → BaseResult

Uses VLM (via Ollama/OpenAI-compatible API) or OCR to convert images
to structured data.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from docmirror.framework.base import BaseParser, ParserOutput, ParserStatus
from docmirror.models.entities.domain import (
    BaseResult, Block, PageLayout, TextSpan, Style,
)

logger = logging.getLogger(__name__)


class ImageAdapter(BaseParser):
    """Image format adapter — VLM + OCR."""

    async def to_base_result(self, file_path: Path) -> BaseResult:
        """Image → BaseResult."""
        try:
            return await self._vlm_extract(file_path)
        except Exception as e:
            logger.debug(f"[ImageAdapter] VLM unavailable ({e}), falling back to OCR")
            return await self._ocr_fallback(file_path)

    async def _vlm_extract(self, file_path: Path) -> BaseResult:
        """Extract content from image using VLM (Ollama / OpenAI-compatible API)."""
        import httpx

        base_url = os.environ.get("DOCMIRROR_VLM_BASE_URL", "http://localhost:11434")
        model = os.environ.get("DOCMIRROR_VLM_MODEL", "qwen2.5vl:3b")

        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        ext = file_path.suffix.lower().lstrip(".")
        mime_type = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "gif", "webp") else "image/png"

        prompt = (
            "Identify the type of document and extract all text and structured data "
            "(tables, key-value pairs). Return JSON with keys: "
            "'document_type', 'text_content', 'tables' (list of lists), 'key_entities' (dict)."
        )

        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
            "options": {"temperature": 0.1},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{base_url}/api/generate", json=payload)

        if resp.status_code != 200:
            raise RuntimeError(f"VLM returned {resp.status_code}: {resp.text[:200]}")

        content = resp.json().get("response", "")
        blocks: list[Block] = []
        metadata: Dict[str, Any] = {"source_format": "image"}

        # Try to parse JSON from the response
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict):
                    metadata["document_type"] = data.get("document_type", "unknown")
                    if "text_content" in data:
                        blocks.append(Block(
                            block_type="text",
                            raw_content=data["text_content"],
                            page=0,
                        ))
                    if "tables" in data and isinstance(data["tables"], list):
                        for tbl in data["tables"]:
                            if isinstance(tbl, list):
                                blocks.append(Block(
                                    block_type="table",
                                    raw_content=tbl,
                                    page=0,
                                ))
                    if "key_entities" in data and isinstance(data["key_entities"], dict):
                        blocks.append(Block(
                            block_type="key_value",
                            raw_content=data["key_entities"],
                            page=0,
                        ))
            except json.JSONDecodeError:
                pass

        if not blocks:
            blocks.append(Block(block_type="text", raw_content=content, page=0))

        page = PageLayout(page_number=0, blocks=tuple(blocks))
        return BaseResult(
            pages=(page,),
            metadata=metadata,
            full_text=content,
        )

    async def _ocr_fallback(self, file_path: Path) -> BaseResult:
        """OCR fallback path."""
        try:
            from docmirror.core.ocr.vision.rapidocr_engine import get_ocr_engine
            engine = get_ocr_engine()
            if engine is None:
                raise ImportError("OCR engine not available")
            result, _ = engine(str(file_path))
            text = "\n".join(line[1] for line in (result or []))
        except Exception:
            text = ""

        blocks = [Block(block_type="text", raw_content=text, page=0)] if text else []
        page = PageLayout(page_number=0, blocks=tuple(blocks))
        return BaseResult(pages=(page,), full_text=text, metadata={"source_format": "image_ocr"})



