# 📄 DocMirror

**Universal document parsing engine** — extract structured data from any document format.

<div class="grid cards" markdown>

- :material-file-pdf-box: **PDF** — Digital & scanned, with layout analysis
- :material-image: **Image** — OCR + VLM extraction
- :material-file-word: **Office** — Word, Excel, PowerPoint
- :material-email: **Email** — EML/MSG with attachments
- :material-table: **Tables** — Multi-strategy extraction
- :material-shield-check: **Security** — Forgery detection

</div>

## Quick Start

```bash
pip install docmirror[pdf,ocr]
```

```python
from docmirror import perceive_document

result = await perceive_document("invoice.pdf")

# Structured output
print(result.content.text)          # Full text (Markdown)
print(result.content.entities)      # Key-value entities
for block in result.content.blocks: # Content blocks
    print(block.type, block.table or block.text)
```

## Why DocMirror?

| Feature | DocMirror | Traditional OCR |
|---------|-----------|----------------|
| Format support | 8+ formats | PDF only |
| Table extraction | 4-tier progressive | Single strategy |
| Layout analysis | AI + rules | None |
| Plugin system | Extensible domains | Hardcoded |
| OCR engine | RapidOCR (ONNX) | Tesseract |

## Next Steps

- [Installation](getting-started/installation.md)
- [Quick Start Guide](getting-started/quickstart.md)
- [Architecture Overview](guide/architecture.md)
- [Creating Plugins](plugins/creating-plugins.md)
