# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Email Adapter — .eml → ParseResult
=====================================

Parses email files (.eml format) using Python's standard library ``email``
module with the default policy for modern email handling.

Processing logic:
    1. Opens the .eml file in binary mode and parses using ``email.policy.default``.
    2. Extracts header fields into KeyValuePairs:
       - subject, from, to, date, message_id
    3. Extracts body text into TextBlocks.
    4. Attachment filenames are listed in key_values (content not extracted).

Metadata includes:
    - parser_name: "EmailAdapter"
"""

from __future__ import annotations

import email as email_lib
import logging
from email import policy
from pathlib import Path
from typing import Dict, List

from docmirror.framework.base import BaseParser

logger = logging.getLogger(__name__)


class EmailAdapter(BaseParser):
    """Email (.eml) format adapter — extracts headers, body text, and attachment metadata."""

    async def to_parse_result(self, file_path: Path, **kwargs) -> ParseResult:
        """
        Parse an .eml file into a ParseResult.

        Email headers become KeyValuePairs. The plain-text body
        becomes TextBlocks. Attachment filenames are listed in the
        key_value pairs but their contents are not extracted.
        """
        from docmirror.models.entities.parse_result import (
            KeyValuePair,
            PageContent,
            ParseResult,
            ParserInfo,
            TextBlock,
            TextLevel,
        )

        logger.info(f"[EmailAdapter] Starting extraction for email: {file_path}")
        with open(file_path, "rb") as f:
            msg = email_lib.message_from_binary_file(f, policy=policy.default)

        # Extract standard email header fields
        key_values: list[KeyValuePair] = [
            KeyValuePair(key="subject", value=msg["subject"] or ""),
            KeyValuePair(key="from", value=msg["from"] or ""),
            KeyValuePair(key="to", value=msg["to"] or ""),
            KeyValuePair(key="date", value=msg["date"] or ""),
            KeyValuePair(key="message_id", value=msg.get("message-id", "")),
        ]

        text_parts: list[str] = []
        attachments: list[str] = []

        if msg.is_multipart():
            for part in msg.walk():
                disp = str(part.get("Content-Disposition", ""))
                if "attachment" in disp:
                    fname = part.get_filename()
                    if fname:
                        attachments.append(fname)
                elif part.get_content_type() == "text/plain":
                    try:
                        text_parts.append(part.get_content())
                    except Exception as exc:
                        logger.debug(f"operation: suppressed {exc}")
        else:
            try:
                text_parts.append(msg.get_content())
            except Exception as exc:
                logger.debug(f"operation: suppressed {exc}")

        if attachments:
            key_values.append(KeyValuePair(key="attachments", value=", ".join(attachments)))

        texts: list[TextBlock] = []
        for part_text in text_parts:
            if part_text.strip():
                texts.append(TextBlock(content=part_text, level=TextLevel.BODY))

        page = PageContent(
            page_number=0,
            texts=texts,
            key_values=key_values,
        )

        return ParseResult(
            pages=[page],
            parser_info=ParserInfo(
                parser_name="EmailAdapter",
                page_count=1,
                overall_confidence=1.0,
            ),
        )
