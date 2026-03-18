# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# Author: Adam Lin <adamlin@valuemapglobal.com>
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
HTML utility functions for extraction.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


def strip_html_to_plain_text(html_text: str, drop_tables: bool = False) -> str:
    """Strip HTML tags and normalize whitespace so output is plain text.

    Used when external OCR returns HTML (e.g. AI Studio markdown.text); we
    emit plain text into the comprehensive result instead of raw HTML.

    When drop_tables is True, removes <table>…</table> segments before
    stripping, so the text block does not duplicate table content (tables
    are represented only in the key_value block).
    """
    if not html_text:
        return ""
    if drop_tables:
        # Remove entire table blocks so they are not turned into flat text
        html_text = re.sub(r"<table[^>]*>.*?</table>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
    # Remove remaining tags and collapse whitespace
    text = re.sub(r"<[^>]+>", " ", html_text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_html_tables_to_key_value(html_text: str) -> dict[str, str] | None:
    """Parse two-column HTML <table> in text into a single key-value dict.

    Used when external OCR returns markdown/HTML with <table>…</table> so we
    can emit a key_value block for downstream (APIs, entities). Rows with
    exactly two <td>s become one pair; multi-table or multi-row are merged.
    Returns None if no table or no pairs.
    """
    if not html_text or "<table" not in html_text.lower():
        return None
    # Extract cell text in order (handles simple <td>...</td>)
    cells = re.findall(r"<td[^>]*>([^<]*)</td>", html_text, re.IGNORECASE)
    cells = [c.strip() for c in cells if c.strip()]
    if len(cells) < 2:
        return None
    out: dict[str, str] = {}
    for i in range(0, len(cells) - 1, 2):
        k, v = cells[i], cells[i + 1]
        if k:
            out[k] = v
    return out if out else None
