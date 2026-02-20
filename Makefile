.PHONY: install install-dev test lint run run-step clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install with dev dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v

lint:  ## Run linter
	ruff check src/ tests/
	ruff format --check src/ tests/

format:  ## Auto-format code
	ruff format src/ tests/

run:  ## Run full pipeline
	gss run

run-step:  ## Run a single step (usage: make run-step STEP=s01_extract_frames)
	gss run-step $(STEP)

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
