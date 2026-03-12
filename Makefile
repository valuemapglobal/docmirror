.PHONY: help install lint format test clean

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies for development"
	@echo "  make format   - Format code using ruff"
	@echo "  make lint     - Run static analysis using ruff and mypy"
	@echo "  make test     - Run tests with pytest"
	@echo "  make clean    - Remove build artifacts and cache directories"

install:
	pip install -e ".[all,dev,docs]"
	pre-commit install

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	mypy docmirror/

test:
	pytest

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
