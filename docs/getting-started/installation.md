# Installation

## Basic Install

```bash
pip install docmirror
```

## With Format Support

Install only the extras you need:

```bash
# PDF parsing
pip install docmirror[pdf]

# OCR (for scanned documents)
pip install docmirror[ocr]

# Office formats (Word, Excel, PowerPoint)
pip install docmirror[office]

# Everything
pip install docmirror[all]
```

## Available Extras

| Extra | Packages | Use Case |
|-------|----------|----------|
| `pdf` | PyMuPDF, pdfplumber | Digital & scanned PDFs |
| `ocr` | RapidOCR, OpenCV | Scanned document text recognition |
| `layout` | rapid-layout | AI-powered layout analysis |
| `table` | rapid-table | Advanced table structure recognition |
| `formula` | rapid-latex-ocr | Mathematical formula recognition |
| `office` | python-docx, openpyxl, python-pptx | Word, Excel, PowerPoint |
| `security` | pikepdf | PDF forgery detection |
| `cache` | redis | Parse result caching |
| `langdetect` | fast-langdetect | Language detection |
| `all` | All of the above | Full installation |
| `dev` | pytest, ruff, mypy, coverage | Development tools |

## Requirements

- **Python**: 3.10+
- **OS**: Linux, macOS, Windows
