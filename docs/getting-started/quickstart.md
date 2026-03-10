# Quick Start

## Parse a PDF

```python
import asyncio
from docmirror import perceive_document

async def main():
    result = await perceive_document(
        "statement.pdf",
        enhance_mode="standard",  # raw | standard | full
    )

    # Check status
    print(f"Status: {result.status}")
    print(f"Confidence: {result.confidence:.0%}")

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
export DOCMIRROR_ENHANCE_MODE=full
export DOCMIRROR_ENABLE_LLM=true
export DOCMIRROR_MAX_PAGES=100
export DOCMIRROR_OCR_DPI=200
```
