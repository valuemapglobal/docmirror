"""
Image Adapter — Image → BaseResult
====================================

Converts image files (JPG, PNG, TIFF, etc.) into structured data using a
two-tier extraction strategy:

1. **Primary (VLM)**: Sends the image to a Vision-Language Model
   (e.g., Qwen2.5-VL) via the Ollama/OpenAI-compatible REST API.
   The VLM is prompted to return JSON with document_type, text_content,
   tables, and key_entities. If the response contains valid JSON,
   it is parsed into typed Blocks.

2. **Fallback (OCR)**: If the VLM is unavailable or fails, falls back
   to RapidOCR (ONNX Runtime) for plain text extraction. This path
   produces a single text Block without structured table/entity data.

Environment variables:
    DOCMIRROR_VLM_BASE_URL  — Ollama API base URL (default: http://localhost:11434)
    DOCMIRROR_VLM_MODEL     — VLM model name (default: qwen2.5vl:3b)
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
    """
    Image format adapter with VLM primary extraction and OCR fallback.

    The adapter first attempts VLM-based extraction for rich structured output.
    If the VLM endpoint is unreachable or returns an error, it silently
    falls back to OCR for basic text recognition.
    """

    async def to_base_result(self, file_path: Path) -> BaseResult:
        """
        Convert an image file to BaseResult.

        Tries VLM extraction first, falls back to OCR on any failure.
        """
        try:
            return await self._vlm_extract(file_path)
        except Exception as e:
            logger.debug(f"[ImageAdapter] VLM unavailable ({e}), falling back to OCR")
            return await self._ocr_fallback(file_path)

    async def _vlm_extract(self, file_path: Path) -> BaseResult:
        """
        Extract structured content from an image using a Vision-Language Model.

        Sends the base64-encoded image to the Ollama API with a structured
        extraction prompt. Parses the VLM response for JSON containing:
        - document_type: classification of the document
        - text_content: full text extracted from the image
        - tables: list of 2D arrays (table data)
        - key_entities: dict of extracted key-value pairs

        If the VLM response does not contain parseable JSON, the raw
        text response is stored as a single text Block.
        """
        import httpx

        base_url = os.environ.get("DOCMIRROR_VLM_BASE_URL", "http://localhost:11434")
        model = os.environ.get("DOCMIRROR_VLM_MODEL", "qwen2.5vl:3b")

        # Read and base64-encode the image for the API payload
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        # Determine MIME type from file extension
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

        # Attempt to extract JSON from a ```json ... ``` code fence in the response
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict):
                    metadata["document_type"] = data.get("document_type", "unknown")

                    # Create a text block from the extracted text content
                    if "text_content" in data:
                        blocks.append(Block(
                            block_type="text",
                            raw_content=data["text_content"],
                            page=0,
                        ))

                    # Create table blocks from extracted tabular data
                    if "tables" in data and isinstance(data["tables"], list):
                        for tbl in data["tables"]:
                            if isinstance(tbl, list):
                                blocks.append(Block(
                                    block_type="table",
                                    raw_content=tbl,
                                    page=0,
                                ))

                    # Create a key-value block from extracted entities
                    if "key_entities" in data and isinstance(data["key_entities"], dict):
                        blocks.append(Block(
                            block_type="key_value",
                            raw_content=data["key_entities"],
                            page=0,
                        ))
            except json.JSONDecodeError:
                pass

        # If no structured blocks were created, store the raw VLM text response
        if not blocks:
            blocks.append(Block(block_type="text", raw_content=content, page=0))

        page = PageLayout(page_number=0, blocks=tuple(blocks))
        return BaseResult(
            pages=(page,),
            metadata=metadata,
            full_text=content,
        )

    async def _ocr_fallback(self, file_path: Path) -> BaseResult:
        """
        Fallback path: extract text from the image using RapidOCR.

        Returns a BaseResult with a single text Block containing all
        recognized text lines joined by newlines. If OCR is unavailable
        or produces no output, returns an empty result.
        """
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
