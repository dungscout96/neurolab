.PHONY: install install-dev lint test clean

# Install all sub-libraries in editable mode
install:
	pip install -e ./libs/config -e ./libs/jobs -e ./libs/data

# Install with dev dependencies
install-dev:
	pip install -e "./libs/config[dev]" -e "./libs/jobs[dev]" -e "./libs/data[dev]" -e ".[dev]"

lint:
	ruff check libs/ tests/ examples/
	black --check libs/ tests/ examples/

format:
	ruff check --fix libs/ tests/ examples/
	black libs/ tests/ examples/

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
