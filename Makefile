.PHONY: ci

ci:
	uvx ruff check .
	uvx ruff format --check .
	uv run pytest
