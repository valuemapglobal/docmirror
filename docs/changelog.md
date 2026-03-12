# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-12

### Added
- `--skip-cache` CLI flag for forcing full re-parse
- `scene` persisted as a Pydantic field on `PerceptionResult` (survives cache serialization)
- Trust scoring system: mirror fidelity validation with 7 artifact dimensions
- Image quality assessment with VLM recommendation flags
- Generic `Validator` with document-type-agnostic scoring dimensions
- Bilingual entity extraction (Chinese/English concatenated label-value patterns)
- 110 test cases with full pipeline coverage

### Changed
- Middleware pipeline simplified: removed ColumnMapper and Repairer
- Entity extraction reads from `enhanced.enhanced_data` (was incorrectly using `base_result.metadata`)
- `to_api_dict()` uses persisted `self.scene` instead of transient `_enhanced` attribute
- Block output cleaned: removed internal `markdown`/`bbox` fields from table blocks
- Empty image blocks (logos/stamps) filtered from API output

### Removed
- `docmirror/integrations/` — LangChain/LlamaIndex loaders (planned for v0.3)
- `docmirror/middlewares/alignment/column_mapper.py` — standardization logic removed
- `docmirror/middlewares/alignment/repairer.py` — LLM-powered repair removed
- `output_file` field from API response (CLI-only concern)
- `scripts/classify_docs.py` — unused standalone script

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
- Middleware pipeline: SceneDetector, EntityExtractor, InstitutionDetector, Validator
- Redis-based parse result caching
- `pyproject.toml` with modular optional dependencies
- Test suite with pytest
- GitHub Actions CI/CD (lint, test on Python 3.10-3.13, build)
- Apache 2.0 license
