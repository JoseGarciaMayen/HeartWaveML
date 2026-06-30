import os

PROJECT_NAME = "HeartWaveML"


def _clearml_configured() -> bool:
    if os.getenv("CLEARML_OFFLINE", "").lower() in ("1", "true"):
        return True
    if os.getenv("CLEARML_API_HOST") or os.getenv("CLEARML_WEB_HOST"):
        return True
    return os.path.exists(os.path.expanduser("~/clearml.conf"))


def init_clearml(task_name: str, task_type: str = "training", tags: list[str] | None = None):
    """Start a ClearML Task alongside MLflow. Returns the Task or None.

    No-op if ClearML is missing or unconfigured, so the existing MLflow-only
    pipeline keeps working. Tracking failures never break training.
    """
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
