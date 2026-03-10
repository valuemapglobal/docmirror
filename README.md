<p align="center">
  <h1 align="center">📄 DocMirror</h1>
  <p align="center">
    <em>Universal document parsing engine — extract structured data from any document format.</em>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/docmirror/"><img src="https://img.shields.io/pypi/v/docmirror?color=blue" alt="PyPI"></a>
    <a href="https://pypi.org/project/docmirror/"><img src="https://img.shields.io/pypi/pyversions/docmirror" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  </p>
</p>

---

DocMirror is a Python library that parses documents into structured, machine-readable data. It supports **8 file formats** out of the box and provides a modular pipeline with OCR, layout analysis, table extraction, and document forgery detection.

## ✨ Features

| Capability | Description |
|---|---|
| 📑 **Multi-format** | PDF, Image, Word, Excel, PowerPoint, Email, Web, JSON/XML/CSV |
| 🔍 **OCR** | RapidOCR (ONNX) with automatic language detection |
| 📐 **Layout Analysis** | DocLayout-YOLO / RapidLayout for page segmentation |
| 📊 **Table Extraction** | Multi-strategy: rule-based, PDFPlumber, RapidTable, VLM fallback |
| 🧮 **Formula Recognition** | LaTeX-OCR for mathematical formula extraction |
| 🛡️ **Forgery Detection** | PDF metadata & image tamper analysis |
| 🤖 **VLM Integration** | Ollama-based Vision LLM for complex page understanding |
| 🧩 **Plugin System** | Extensible middleware pipeline for domain-specific extraction |

## 🚀 Quick Start

### Installation

```bash
# Core only (minimal dependencies)
pip install docmirror

# With PDF + OCR support
pip install docmirror[pdf,ocr]

# Everything
pip install docmirror[all]
```

### Basic Usage

```python
import asyncio
from docmirror import perceive_document, DocumentType

async def main():
    result = await perceive_document("invoice.pdf", DocumentType.OTHER)

    print(f"Status: {result.status}")
    print(f"Text: {result.content.text[:200]}")
    print(f"Tables: {len(result.tables)}")
    print(f"Entities: {result.content.entities}")

asyncio.run(main())
```

### Supported Formats

| Format | Extensions | Adapter |
|---|---|---|
| PDF | `.pdf` | `PDFAdapter` — PyMuPDF + OCR + Layout + Table |
| Image | `.png` `.jpg` `.jpeg` `.tiff` `.bmp` | `ImageAdapter` — VLM + OCR fallback |
| Word | `.doc` `.docx` | `WordAdapter` — python-docx |
| Excel | `.xls` `.xlsx` | `ExcelAdapter` — openpyxl |
| PowerPoint | `.ppt` `.pptx` | `PPTAdapter` — python-pptx |
| Email | `.eml` `.msg` | `EmailAdapter` — stdlib email |
| Web | `.html` `.htm` | `WebAdapter` |
| Structured | `.json` `.xml` `.csv` | `StructuredAdapter` |

## 🏗️ Architecture

```
perceive_document()
    │
    ▼
┌──────────────────┐
│  ParserDispatcher │ ← L0 file type routing (magic number + extension)
│  (framework/)     │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ PDF    │ │ Image  │ ... (8 adapters)
│Adapter │ │Adapter │
└───┬────┘ └────────┘
    │
    ▼
┌──────────────────┐
│  CoreExtractor   │ ← Foundation + Layout + OCR + Table engines
│  (core/)         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Orchestrator    │ ← Middleware pipeline
│  (framework/)    │
│                  │
│  Detection ──▶   │
│  Extraction ──▶  │
│  Alignment ──▶   │
│  Validation ──▶  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ PerceptionResult │ ← Unified 4-layer output model
│ (models/)        │
└──────────────────┘
```

## 📦 Optional Dependencies

DocMirror uses modular optional dependencies. Install only what you need:

```bash
pip install docmirror[pdf]        # PyMuPDF + pdfplumber
pip install docmirror[ocr]        # RapidOCR + OpenCV + NumPy
pip install docmirror[layout]     # RapidLayout
pip install docmirror[table]      # RapidTable
pip install docmirror[formula]    # LaTeX-OCR
pip install docmirror[office]     # python-docx, openpyxl, python-pptx
pip install docmirror[vlm]        # httpx (Ollama VLM client)
pip install docmirror[security]   # pikepdf (forgery detection)
pip install docmirror[cache]      # Redis caching
pip install docmirror[all]        # Everything above
```

## 🔧 Configuration

DocMirror is configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `DOCMIRROR_ENHANCE_MODE` | `standard` | Enhancement mode: `raw`, `standard`, `full` |
| `DOCMIRROR_ENABLE_LLM` | `false` | Enable LLM-powered middlewares |
| `DOCMIRROR_MAX_PAGES` | `200` | Maximum pages to process |
| `DOCMIRROR_VLM_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DOCMIRROR_VLM_MODEL` | `qwen2.5vl:3b` | VLM model name |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for caching |

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

DocMirror is licensed under the [Apache License 2.0](LICENSE).
