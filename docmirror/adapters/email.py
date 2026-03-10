"""
Email Adapter — Email → BaseResult

提取邮件正文、标头和附件元数据。
"""

from __future__ import annotations

import email as email_lib
from email import policy
import logging
from pathlib import Path
from typing import Dict, Any

from docmirror.framework.base import BaseParser, ParserOutput, ParserStatus
from docmirror.models.domain import BaseResult, Block, PageLayout

logger = logging.getLogger(__name__)


class EmailAdapter(BaseParser):
    """Email (.eml) 格式适配器。"""

    async def to_base_result(self, file_path: Path) -> BaseResult:
        """Email → BaseResult。"""
        with open(file_path, "rb") as f:
            msg = email_lib.message_from_binary_file(f, policy=policy.default)

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
            for part in msg.walk():
                disp = str(part.get("Content-Disposition", ""))
                if "attachment" in disp:
                    fname = part.get_filename()
                    if fname:
                        attachments.append(fname)
                elif part.get_content_type() == "text/plain":
                    try:
                        text_parts.append(part.get_content())
                    except Exception:
                        pass
        else:
            try:
                text_parts.append(msg.get_content())
            except Exception:
                pass

        full_text = "\n\n".join(text_parts)
        if attachments:
            kv["attachments"] = ", ".join(attachments)

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


