.PHONY: env env_dry_run test

env_dry_run:
	UV_PROJECT_ENVIRONMENT=.venv uv sync --group openai --group gemini --dry-run

env:
	UV_PROJECT_ENVIRONMENT=.venv uv sync --group openai --group gemini

test:
	python -m pytest tests/ -v
