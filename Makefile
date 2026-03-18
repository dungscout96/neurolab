.PHONY: install install-dev lint test clean

# Install in editable mode
install:
	pip install -e .

# Install with dev dependencies
install-dev:
	pip install -e ".[dev]"

lint:
	ruff check neurolab/ tests/ examples/
	black --check neurolab/ tests/ examples/

format:
	ruff check --fix neurolab/ tests/ examples/
	black neurolab/ tests/ examples/

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
