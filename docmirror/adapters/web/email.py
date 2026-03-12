"""
Email Adapter — .eml → BaseResult
===================================

Parses email files (.eml format) using Python's standard library ``email``
module with the default policy for modern email handling.

Processing logic:
    1. Opens the .eml file in binary mode and parses using ``email.policy.default``.
    2. Extracts header fields into a key-value dict:
       - subject, from, to, date, message_id
    3. Extracts body text:
       - For multipart messages, walks all parts:
         * Parts with "attachment" Content-Disposition → recorded as attachment
           filenames (not extracted, only listed).
         * Parts with content type "text/plain" → appended to body text.
       - For non-multipart messages, reads the content directly.
    4. If attachments are found, their filenames are joined into a single
       comma-separated string in the key-value dict.
    5. Output structure:
       - A ``key_value`` Block with email headers (and attachment list if any).
       - A ``text`` Block with the concatenated body text (if non-empty).

Metadata includes:
    - source_format: "email"
    - attachment_count: number of detected attachments
"""
from __future__ import annotations


import email as email_lib
from email import policy
import logging
from pathlib import Path
from typing import Dict

from docmirror.framework.base import BaseParser
from docmirror.models.domain import BaseResult, Block, PageLayout

logger = logging.getLogger(__name__)


class EmailAdapter(BaseParser):
    """Email (.eml) format adapter — extracts headers, body text, and attachment metadata."""

    async def to_base_result(self, file_path: Path) -> BaseResult:
        """
        Parse an .eml file into a BaseResult.

        Email headers become a key_value Block. The plain-text body
        becomes a text Block. Attachment filenames are listed in the
        key_value pairs but their contents are not extracted.
        """
        with open(file_path, "rb") as f:
            msg = email_lib.message_from_binary_file(f, policy=policy.default)

        # Extract standard email header fields
        kv: Dict[str, str] = {
            "subject": msg["subject"] or "",
            "from": msg["from"] or "",
            "to": msg["to"] or "",
            "date": msg["date"] or "",
            "message_id": msg.get("message-id", ""),
        }

        text_parts = []
        attachments = []

        if msg.is_multipart():
            # Walk all MIME parts to find body text and attachment names
            for part in msg.walk():
                disp = str(part.get("Content-Disposition", ""))
                if "attachment" in disp:
                    # Record attachment filename (content not extracted)
                    fname = part.get_filename()
                    if fname:
                        attachments.append(fname)
                elif part.get_content_type() == "text/plain":
                    try:
                        text_parts.append(part.get_content())
                    except Exception as exc:
                        logger.debug(f"operation: suppressed {exc}")
        else:
            # Non-multipart: read content directly
            try:
                text_parts.append(msg.get_content())
            except Exception as exc:
                logger.debug(f"operation: suppressed {exc}")

        full_text = "\n\n".join(text_parts)

        # Add attachment filenames to key-value data if any were found
        if attachments:
            kv["attachments"] = ", ".join(attachments)

        # Build output blocks: key-value header first, then body text
        blocks = [
            Block(block_type="key_value", raw_content=kv, page=0),
        ]
        if full_text:
            blocks.append(Block(block_type="text", raw_content=full_text, page=0))

        page = PageLayout(page_number=0, blocks=tuple(blocks))
        return BaseResult(
            pages=(page,),
            full_text=full_text,
            metadata={"source_format": "email", "attachment_count": len(attachments)},
        )
