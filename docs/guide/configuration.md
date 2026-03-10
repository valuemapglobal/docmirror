# Configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCMIRROR_ENHANCE_MODE` | `standard` | Pipeline mode: `raw`, `standard`, `full` |
| `DOCMIRROR_ENABLE_LLM` | `false` | Enable LLM-powered validation |
| `DOCMIRROR_MAX_PAGES` | `200` | Maximum pages to process |
| `DOCMIRROR_OCR_DPI` | `150` | OCR rendering resolution |
| `DOCMIRROR_FAIL_STRATEGY` | `skip` | Error handling: `skip`, `raise`, `fallback` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for parse result caching |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint for VLM |

## Programmatic Configuration

```python
from docmirror.configs.settings import DocMirrorSettings

settings = DocMirrorSettings(
    default_enhance_mode="full",
    enable_llm=True,
    max_pages=100,
    ocr_dpi=200,
)
```

## Enhancement Modes

| Mode | Middlewares | Use Case |
|------|-----------|----------|
| `raw` | Scene detection only | Fast preview, format conversion |
| `standard` | Detection + Entity + Column mapping + Repair | Production parsing |
| `full` | Standard + LLM validation | High-accuracy, audit-grade |
