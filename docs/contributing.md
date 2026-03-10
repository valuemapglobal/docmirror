# Contributing to DocMirror

Thank you for your interest in contributing to DocMirror! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/valuemapglobal/docmirror.git
cd docmirror

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with dev dependencies
pip install -e ".[dev,all]"

# Verify setup
pytest tests/ -v
```

## Development Workflow

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make changes** — follow the coding standards below.

3. **Run checks** before committing:
   ```bash
   ruff check docmirror/        # Lint
   ruff format docmirror/       # Format
   pytest tests/ -v             # Test
   ```

4. **Commit** with [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add new PDF table extraction strategy
   fix: correct column alignment for merged cells
   docs: update README with new configuration options
   chore: update dependencies
   ```

5. **Submit a Pull Request** against `main`.

## Coding Standards

- **Python 3.10+** — use modern syntax (`match/case`, `X | Y` unions)
- **Type hints** on all public functions
- **English** for all comments, docstrings, and variable names
- **Docstrings** in Google style for public API
- **Line length** — 120 characters max (enforced by ruff)

## Project Structure

```
docmirror/
├── adapters/       # Format-specific adapters (PDF, Image, Office, ...)
├── configs/        # Settings, YAML configs, registries
├── core/           # Core engines (extraction, layout, OCR, table, security)
├── framework/      # Dispatcher, orchestrator, base classes
├── middlewares/    # Pipeline middlewares (detection, extraction, alignment, validation)
└── models/         # Data models (domain, enhanced, perception_result)
```

## Adding a New Adapter

1. Create `docmirror/adapters/your_format.py`
2. Subclass `BaseParser` from `docmirror.framework.base`
3. Implement `to_base_result(file_path) -> BaseResult`
4. Register in the dispatcher's `_get_parser()` method
5. Add tests in `tests/test_your_format.py`

## Adding a New Middleware

1. Create `docmirror/middlewares/your_category/your_middleware.py`
2. Subclass `BaseMiddleware` from `docmirror.middlewares.base`
3. Implement `process(result) -> result`
4. Register in `docmirror/configs/pipeline_registry.py`
5. Add tests

## Reporting Issues

- Use [GitHub Issues](https://github.com/valuemapglobal/docmirror/issues)
- Include: Python version, OS, DocMirror version, minimal reproduction steps
- For document parsing issues, include a sample file if possible (redact sensitive data)

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
