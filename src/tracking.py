import os

PROJECT_NAME = "HeartWaveML"


def _clearml_configured() -> bool:
    if os.getenv("CLEARML_OFFLINE", "").lower() in ("1", "true"):
        return True
    if os.getenv("CLEARML_API_HOST") or os.getenv("CLEARML_WEB_HOST"):
        return True
    return os.path.exists(os.path.expanduser("~/clearml.conf"))


def init_clearml(task_name: str, task_type: str = "training", tags: list[str] | None = None):
    """Start a ClearML Task. Returns the Task, or None if ClearML is missing/unconfigured
    (tracking failures never break training)."""
    if not _clearml_configured():
        return None
    try:
        from clearml import Task
    except ImportError:
        return None
    try:
        if os.getenv("CLEARML_OFFLINE", "").lower() in ("1", "true"):
            Task.set_offline(True)
        return Task.init(
            project_name=PROJECT_NAME,
            task_name=task_name,
            task_type=task_type,
            tags=tags,
            auto_connect_frameworks=True,
            reuse_last_task_id=False,
            output_uri=False,
        )
    except Exception as e:
        print(f"[ClearML] init skipped: {e}")
        return None


def clearml_log_params(params: dict, name: str = "General") -> None:
    from clearml import Task

    task = Task.current_task()
    if task is None:
        return
    task.connect(dict(params), name=name)


def clearml_log_metrics(metrics: dict) -> None:
    from clearml import Task

    task = Task.current_task()
    if task is None:
        return
    logger = task.get_logger()
    for k, v in metrics.items():
        logger.report_single_value(name=k, value=float(v))


def clearml_get_best_params(experiment_name: str, run_id: str | None = None) -> dict:
    from clearml import Task

    if run_id is not None:
        task = Task.get_task(task_id=run_id)
        return task.get_parameters_as_dict(cast=True).get("General", {})

    tasks = Task.get_tasks(
        project_name=PROJECT_NAME,
        tags=[experiment_name],
        task_filter={"type": ["optimizer"], "status": ["completed"]},
    )
    scored = []
    for t in tasks:
        m = t.get_last_scalar_metrics()
        val = m.get("Summary", {}).get("val_f1_macro", {}).get("last")
        if val is not None:
            scored.append((val, t))
    if not scored:
        raise ValueError(f"No completed tuning runs found for '{experiment_name}'.")
    best_task = max(scored, key=lambda x: x[0])[1]
    return best_task.get_parameters_as_dict(cast=True).get("General", {})
