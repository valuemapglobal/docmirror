from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any

class ParseResponse(BaseModel):
    """
    Standardized HTTP response wrapper for Document Parsing.
    """
    success: bool = Field(..., description="Whether the document was successfully parsed")
    status: str = Field(..., description="Result status: 'success', 'partial', or 'failure'")
    error: str = Field(default="", description="Error message if any")
    
    identity: Dict[str, Any] = Field(default_factory=dict, description="Identified document type and metadata")
    blocks: list[Dict[str, Any]] = Field(default_factory=list, description="Extracted content blocks (text, tables)")
    
    trust: Dict[str, Any] = Field(default_factory=dict, description="Validation scores and forgery detection")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Performance and pipeline diagnostics")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "status": "success",
            "error": "",
            "identity": {"type": "invoice", "page_count": 1},
            "blocks": [{"type": "text", "content": "Total: $100"}],
            "trust": {"validation_score": 100, "is_forged": False},
            "diagnostics": {"elapsed_ms": 1500}
        }
    })
