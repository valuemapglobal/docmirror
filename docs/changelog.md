# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-11

### Added
- Initial open-source release of DocMirror
- 8 format adapters: PDF, Image, Word, Excel, PowerPoint, Email, Web, Structured
- Core extraction engine with PyMuPDF and pdfplumber backends
- OCR support via RapidOCR (ONNX Runtime)
- Layout analysis with DocLayout-YOLO and rule-based fallback
- Multi-strategy table extraction (character-based, PDFPlumber, RapidTable, VLM)
- Formula recognition via LaTeX-OCR
- PDF forgery & tamper detection (ELA + metadata analysis)
- VLM integration via Ollama HTTP API
- Middleware pipeline: SceneDetector, EntityExtractor, ColumnMapper, Validator, Repairer
- Redis-based parse result caching
- `pyproject.toml` with modular optional dependencies
- Test suite (28 tests) with pytest
- GitHub Actions CI/CD (lint, test on Python 3.10-3.13, build)
- Apache 2.0 license
