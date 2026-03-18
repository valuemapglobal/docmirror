# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
DocMirror Universal Parsing API
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import time
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir

from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from docmirror import DocumentType, __version__, perceive_document
from docmirror.server.schemas import ParseResponse

logger = logging.getLogger(__name__)

# ── API Key authentication (set via DOCMIRROR_API_KEY env var) ──
_API_KEY = os.environ.get("DOCMIRROR_API_KEY", "")

app = FastAPI(
    title="DocMirror Universal Parsing API",
    description="High-performance MultiModal document extraction and enhancement engine.",
    version=__version__,
)


# ── Startup/shutdown lifecycle ──


@app.on_event("startup")
async def _cleanup_stale_temp_files():
    """Remove temporary files older than 1 hour on startup.

    Prevents disk exhaustion if previous instances crashed without cleanup.
    """
    tmp_dir = gettempdir()
    cutoff = time.time() - 3600  # 1 hour ago
    cleaned = 0
    for tmp_file in glob.glob(os.path.join(tmp_dir, "tmp*")):
        try:
            if os.path.getmtime(tmp_file) < cutoff:
                os.unlink(tmp_file)
                cleaned += 1
        except OSError:
            pass
    if cleaned:
        logger.info(f"[Server] Cleaned {cleaned} stale temp file(s) on startup")


@app.on_event("startup")
async def _warmup_ocr_engine():
    """Pre-load OCR ONNX model on startup to avoid cold-start latency.

    First request otherwise pays ~500ms-2s for model loading.
    """
    try:
        from docmirror.core.ocr.vision.rapidocr_engine import get_ocr_engine

        engine = get_ocr_engine()
        if engine:
            logger.info("[Server] OCR engine warmed up on startup")
    except Exception as e:
        logger.debug(f"[Server] OCR warmup skipped: {e}")


@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "version": __version__}


def cleanup_file(filepath: Path):
    """Background task to remove temporary files."""
    try:
        if filepath.exists():
            filepath.unlink()
    except Exception as e:
        logger.error(f"[Server] Failed to cleanup temp file {filepath}: {e}")


def _verify_api_key(authorization: str | None) -> None:
    """Verify API key if DOCMIRROR_API_KEY is configured."""
    if not _API_KEY:
        return  # No key configured — open access
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    # Accept "Bearer <key>" or raw key
    token = authorization.removeprefix("Bearer ").strip()
    if token != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.post("/v1/parse", responses={200: {"model": ParseResponse}}, tags=["Parsing"])
async def parse_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="The document file to parse (PDF, PNG, JPEG, DOCX, etc.)"),
    include_text: bool = Query(default=False, description="Include full markdown text in response"),
    authorization: str | None = Header(default=None),
):
    """
    Parse a document using the core MultiModal engine.
    The file is saved temporarily, processed, and then asynchronously cleaned up.
    """
    _verify_api_key(authorization)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided in upload")

    # Create a secure temporary file with the correct extension
    suffix = Path(file.filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = Path(temp_file.name)

    # Schedule cleanup
    background_tasks.add_task(cleanup_file, temp_path)

    try:
        import uuid

        result = await perceive_document(temp_path, DocumentType.OTHER)

        api_payload = result.to_api_dict(
            include_text=include_text,
            request_id=str(uuid.uuid4()),
        )

        status_code = api_payload.get("code", 200)

        return JSONResponse(status_code=status_code, content=api_payload)

    except Exception as e:
        logger.exception("[Server] Parse failed with uncaught exception")
        raise HTTPException(status_code=500, detail=str(e))
