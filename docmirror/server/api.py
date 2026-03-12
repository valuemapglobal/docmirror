from __future__ import annotations
import shutil
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from docmirror import perceive_document, DocumentType
from docmirror.server.schemas import ParseResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocMirror Universal Parsing API",
    description="High-performance MultiModal document extraction and enhancement engine.",
    version="0.1.0",
)

@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}

def cleanup_file(filepath: Path):
    """Background task to remove temporary files."""
    try:
        if filepath.exists():
            filepath.unlink()
    except Exception as e:
        logger.error(f"Failed to cleanup temp file {filepath}: {e}")

@app.post("/v1/parse", response_model=ParseResponse, tags=["Parsing"])
async def parse_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="The document file to parse (PDF, PNG, JPEG, DOCX, etc.)"),
):
    """
    Parse a document using the core MultiModal engine.
    The file is saved temporarily, processed, and then asynchronously cleaned up.
    """
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
        # Route through the unified entry point
        # The underlying Dispatcher handles file_type routing directly.
        result = await perceive_document(temp_path, DocumentType.OTHER)
        
        # Serialize the 4-layer architecture into a flat API dict
        api_payload = result.to_api_dict()
        
        # Determine HTTP status code based on success
        status_code = 200 if api_payload.get("success") else 422
        
        return JSONResponse(status_code=status_code, content=api_payload)

    except Exception as e:
        logger.exception("Parse failed with uncaught exception")
        raise HTTPException(status_code=500, detail=str(e))
