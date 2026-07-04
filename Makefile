.PHONY: ci clean-clearml

ci:
	uvx ruff check .
	uvx ruff format --check .
	uv run pytest

# Deletes all clearml runs
clean-clearml:
	uv run python -c "from clearml import Task; \
n = sum(bool(t.delete(delete_artifacts_and_models=True, skip_models_used_by_other_tasks=True, raise_on_error=False)) for t in Task.get_tasks(project_name='HeartWaveML')); \
print('deleted:', n, 'remaining:', len(Task.get_tasks(project_name='HeartWaveML')))"
