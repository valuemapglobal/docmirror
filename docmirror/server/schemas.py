# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic response schemas for the DocMirror REST API.

Defines the standardized HTTP response models used by the FastAPI
endpoints in ``docmirror.server.api``.

Response envelope follows ``docs/parser_interface.md`` v1.0::

    Success: {code: 200, message, api_version, request_id, timestamp, data, meta}
    Failure: {code: 422, message, api_version, request_id, timestamp, error, meta}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ParseResponse(BaseModel):
    """Standardized RESTful response for document parsing.

    Aligned with ``ParseResult.to_api_dict()`` output and
    ``docs/parser_interface.md`` v1.0.
    """

    code: int = Field(..., description="HTTP status code (200 or 422)")
    message: str = Field(..., description="'success' or 'error'")
    api_version: str = Field(default="1.0", description="API version")
    request_id: str = Field(default="", description="Request tracing ID (UUID)")
    timestamp: str = Field(default="", description="ISO 8601 UTC response time")

    data: dict[str, Any] | None = Field(
        default=None,
        description="Business payload: {document, quality}. Present on success.",
    )
    error: dict[str, Any] | None = Field(
        default=None,
        description="Error details: {type, detail}. Present on failure.",
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Parser diagnostics and provenance",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": 200,
                "message": "success",
                "api_version": "1.0",
                "request_id": "req_abc123",
                "timestamp": "2026-03-18T10:22:17+00:00",
                "data": {
                    "document": {
                        "type": "bank_statement",
                        "properties": {
                            "organization": "重庆三峡银行",
                            "subject_name": "重庆数宜信信用管理有限公司",
                        },
                        "pages": [],
                    },
                    "quality": {
                        "confidence": 1.0,
                        "trust_score": 1.0,
                        "validation_passed": True,
                        "issues": [],
                    },
                },
                "meta": {
                    "parser": "DocMirror",
                    "version": "0.3.0",
                    "elapsed_ms": 55.3,
                    "extraction_method": "digital",
                    "page_count": 4,
                    "table_count": 1,
                    "row_count": 34,
                },
            }
        }
    )
