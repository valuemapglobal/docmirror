# Quick Start

## Parse a PDF

```python
import asyncio
from docmirror import perceive_document

async def main():
    result = await perceive_document("statement.pdf")

    # Check status
    print(f"Status: {result.status}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Scene: {result.scene}")  # "bank_statement", "invoice", etc.

    # Full text (Markdown format)
    print(result.content.text)

    # Iterate content blocks
    for block in result.content.blocks:
        if block.type == "table":
            print(f"Table: {block.table.headers}")
            for row in block.table.rows:
                print(f"  {row}")
        elif block.type == "key_value":
            for k, v in block.key_value.pairs.items():
                print(f"  {k}: {v}")

    # Domain-specific data (if detected)
    if result.domain:
        print(f"Domain: {result.domain.document_type}")

asyncio.run(main())
```

## Parse an Image

```python
result = await perceive_document("receipt.jpg")
print(result.content.text)  # OCR-extracted text
```

## CLI Usage

```bash
# Basic parse
python3 -m docmirror invoice.pdf

# Force re-parse (skip Redis cache)
python3 -m docmirror --skip-cache invoice.pdf

# Don't save output to disk
python3 -m docmirror --no-save invoice.pdf
```

## API Output Structure

The `to_api_dict()` method returns a flat JSON structure:

```json
{
  "success": true,
  "scene": "bank_statement",
  "identity": {
    "document_type": "bank_statement",
    "page_count": 12,
    "properties": {
      "institution": "Example National Bank",
      "account_holder": "Acme Technology Co., Ltd.",
      "account_number": "6200000000001234567"
    }
  },
  "blocks": [...],
  "trust": {
    "validation_score": 1.0,
    "validation_passed": true,
    "validation_details": {...},
    "image_quality": {...}
  },
  "diagnostics": {
    "parser": "PDFAdapter",
    "elapsed_ms": 12000
  }
}
```

## Batch Processing

```python
from pathlib import Path

async def batch_parse(folder: str):
    for path in Path(folder).glob("*.pdf"):
        result = await perceive_document(str(path))
        print(f"{path.name}: {result.status}, {len(result.content.blocks)} blocks")
```

## Configuration via Environment

```bash
export DOCMIRROR_ENHANCE_MODE=standard
export DOCMIRROR_MAX_PAGES=100
export DOCMIRROR_OCR_DPI=200
```
