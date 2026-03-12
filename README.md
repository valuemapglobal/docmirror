<p align="center">
  <h1 align="center">📄 DocMirror</h1>
  <p align="center">
    <em>Universal document parsing engine — extract structured data from any document format.</em>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/docmirror/"><img src="https://img.shields.io/pypi/v/docmirror?color=blue" alt="PyPI"></a>
    <a href="https://pypi.org/project/docmirror/"><img src="https://img.shields.io/pypi/pyversions/docmirror" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
    <a href="https://github.com/valuemapglobal/docmirror/actions"><img src="https://github.com/valuemapglobal/docmirror/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://codecov.io/gh/valuemapglobal/docmirror"><img src="https://codecov.io/gh/valuemapglobal/docmirror/branch/main/graph/badge.svg" alt="Coverage"></a>
    <a href="https://valuemapglobal.github.io/docmirror"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Docs"></a>
  </p>
</p>

---

DocMirror is a Python library that parses documents into structured, machine-readable data. It supports **8 file formats** out of the box and provides a modular pipeline with OCR, layout analysis, table extraction, and trust scoring.

## ✨ Features

| Capability | Description |
|---|---|
| 📑 **Multi-format** | PDF, Image, Word, Excel, PowerPoint, Email, Web, JSON/XML/CSV |
| 🔍 **OCR** | RapidOCR (ONNX) with multi-scale fusion and dynamic color slicing |
| 📐 **Layout Analysis** | DocLayout-YOLO / RapidLayout for page segmentation |
| 📊 **Table Extraction** | Multi-strategy: rule-based, PDFPlumber, RapidTable, VLM fallback |
| 🧮 **Formula Recognition** | LaTeX-OCR for mathematical formula extraction |
| 🛡️ **Forgery Detection** | PDF metadata & image tamper analysis (ELA) |
| ✅ **Trust Scoring** | 7-dimension mirror fidelity validation with confidence scores |
| 🧩 **Plugin System** | Extensible domain plugins for business-specific extraction |
| 💾 **Caching** | Redis-based parse result caching with full state persistence |

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
from docmirror import perceive_document

async def main():
    result = await perceive_document("invoice.pdf")

    print(f"Status: {result.status}")
    print(f"Scene: {result.scene}")          # "bank_statement", "invoice", etc.
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Text: {result.content.text[:200]}")

    # Iterate structured blocks
    for block in result.content.blocks:
        if block.type == "table":
            print(f"Table: {block.table.headers}")
        elif block.type == "key_value":
            print(f"Entities: {block.key_value.pairs}")

    # Extracted entities
    print(f"Entities: {result.content.entities}")

asyncio.run(main())
```

### CLI

```bash
# Parse a document
python3 -m docmirror statement.pdf

# Force re-parse (skip cache)
python3 -m docmirror --skip-cache statement.pdf

# Show contributors
python3 -m docmirror --authors
```

### Supported Formats

| Format | Extensions | Adapter |
|---|---|---|
| PDF | `.pdf` | `PDFAdapter` — PyMuPDF + OCR + Layout + Table |
| Image | `.png` `.jpg` `.jpeg` `.tiff` `.bmp` | `ImageAdapter` — OCR extraction |
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
│  ParserDispatcher │ ← L0 file type routing + Redis cache
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
│  CoreExtractor   │ ← Layout + OCR + Table + Formula engines
│  (core/)         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Orchestrator    │ ← Middleware pipeline
│  (framework/)    │
│                  │
│  SceneDetector ──▶│
│  EntityExtractor ▶│
│  InstitutionDet. ▶│
│  Validator ──────▶│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ PerceptionResult │ ← Unified 4-layer output model
│ (models/)        │   (status + content + trust + diagnostics)
└──────────────────┘
```

## 📦 Optional Dependencies

DocMirror uses modular optional dependencies — install only what you need:

```bash
pip install docmirror[pdf]        # PyMuPDF + pdfplumber
pip install docmirror[ocr]        # RapidOCR + OpenCV + NumPy
pip install docmirror[layout]     # DocLayout-YOLO
pip install docmirror[table]      # RapidTable
pip install docmirror[formula]    # LaTeX-OCR
pip install docmirror[office]     # python-docx, openpyxl, python-pptx
pip install docmirror[security]   # pikepdf (forgery detection)
pip install docmirror[cache]      # Redis caching
pip install docmirror[all]        # Everything above
```

## 🔧 Configuration

DocMirror is configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `DOCMIRROR_ENHANCE_MODE` | `standard` | Enhancement mode: `raw`, `standard` |
| `DOCMIRROR_MAX_PAGES` | `200` | Maximum pages to process |
| `DOCMIRROR_OCR_DPI` | `150` | OCR rendering resolution |
| `DOCMIRROR_FAIL_STRATEGY` | `skip` | Error handling: `skip`, `raise`, `fallback` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for caching |

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
pip install -e ".[dev,all]"
pytest tests/ -v          # 109 tests
ruff check docmirror/     # Lint
ruff format docmirror/    # Format
```

## 📄 License

DocMirror is licensed under the [Apache License 2.0](LICENSE).
