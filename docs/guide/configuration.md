# Configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCMIRROR_ENHANCE_MODE` | `standard` | Pipeline mode: `raw`, `standard` |
| `DOCMIRROR_MAX_PAGES` | `200` | Maximum pages to process |
| `DOCMIRROR_OCR_DPI` | `150` | OCR rendering resolution |
| `DOCMIRROR_FAIL_STRATEGY` | `skip` | Error handling: `skip`, `raise`, `fallback` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for parse result caching |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint for VLM |

## Programmatic Configuration

```python
from docmirror.configs.settings import DocMirrorSettings

settings = DocMirrorSettings(
    default_enhance_mode="standard",
    max_pages=100,
    ocr_dpi=200,
)
```

## Enhancement Modes

| Mode | Middlewares | Use Case |
|------|-----------|----------|
| `raw` | None | Fast preview, format conversion |
| `standard` | SceneDetector + EntityExtractor + InstitutionDetector + Validator | Production parsing with full entity extraction and trust scoring |

## Pipeline Configuration

The middleware pipeline is configured per-format in `docmirror/configs/pipeline_registry.py`:

| Format | `raw` | `standard` |
|--------|-------|-----------|
| PDF | — | SceneDetector → EntityExtractor → InstitutionDetector → Validator |
| Image | — | LanguageDetector → GenericEntityExtractor |
| Word | — | LanguageDetector → GenericEntityExtractor |
| Excel | — | GenericEntityExtractor |
| Other | — | LanguageDetector |

## CLI Options

```bash
python3 -m docmirror <file>                    # Parse a document
python3 -m docmirror --skip-cache <file>       # Force re-parse (skip Redis)
python3 -m docmirror --format json <file>      # Output format
python3 -m docmirror --no-save <file>          # Don't save to disk
python3 -m docmirror --output-dir ./out <file> # Custom output directory
python3 -m docmirror --authors                 # Show contributors
```
