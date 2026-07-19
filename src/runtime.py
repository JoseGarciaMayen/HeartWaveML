from __future__ import annotations

import os
from pathlib import Path

DATASET_DEFAULT = "josegarciamayen/heartwaveml-features"
PROJECT_NAME = "HeartWaveML"


def configure_runtime() -> str:
    os.environ["HEARTWAVEML_REQUIRE_CLEARML"] = "1"
    runtime = os.getenv("REMOTE_BRIDGE_RUNTIME")
    if not runtime and os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        runtime = "kaggle"
    runtime = runtime or "local"
    os.environ["HEARTWAVEML_RUNTIME"] = runtime

    if runtime == "kaggle":
        os.environ.setdefault("HEARTWAVEML_RUNTIME", "kaggle")
        device = os.getenv("PROJECT_DEVICE", "cuda")
        os.environ.setdefault("HEARTWAVEML_ENABLE_GPU", "1" if device == "cuda" else "0")
        os.environ.setdefault("HEARTWAVEML_XGB_DEVICE", device)
        handle = os.getenv(
            "REMOTE_BRIDGE_DATASET_HANDLE",
            os.getenv("HEARTWAVEML_DATASET_HANDLE", DATASET_DEFAULT),
        )
        os.environ["HEARTWAVEML_DATASET_HANDLE"] = handle
        mount_processed_dataset(handle)

    require_clearml()
    return runtime


def mount_processed_dataset(handle: str, project_root: Path | None = None) -> Path:
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "Kaggle runtime requires kagglehub. Install the Kaggle dependencies first."
        ) from error

    project_root = project_root or Path(__file__).resolve().parents[1]
    try:
        dataset_root = Path(kagglehub.dataset_download(handle))
    except Exception as error:
        raise RuntimeError(
            f"Kaggle dataset '{handle}' was not found or could not be downloaded. "
            "Set REMOTE_BRIDGE_DATASET_HANDLE or publish it with "
            "remote-bridge dataset upload."
        ) from error
    configured_root = os.getenv("PROJECT_DATA_ROOT")
    processed_root = (
        Path(configured_root).expanduser()
        if configured_root
        else project_root / "data" / "processed"
    )
    if not processed_root.is_absolute():
        processed_root = project_root / processed_root
    processed_root.parent.mkdir(parents=True, exist_ok=True)

    if processed_root.is_symlink():
        processed_root.unlink()
    elif processed_root.exists():
        if not processed_root.is_dir() or any(processed_root.iterdir()):
            raise RuntimeError(f"Cannot mount dataset over non-empty path: {processed_root}")
        processed_root.rmdir()

    if (dataset_root / "feat").is_dir() or (dataset_root / "base").is_dir():
        processed_root.symlink_to(dataset_root, target_is_directory=True)
    else:
        processed_root.mkdir()
        feature_files = tuple(dataset_root.glob("*features.csv"))
        base_files = tuple(dataset_root.glob("mitbih_*.csv"))
        if feature_files:
            (processed_root / "feat").symlink_to(dataset_root, target_is_directory=True)
        if base_files and not feature_files:
            (processed_root / "base").symlink_to(dataset_root, target_is_directory=True)
        if not feature_files and not base_files:
            processed_root.rmdir()
            processed_root.symlink_to(dataset_root, target_is_directory=True)
    print(f"dataset={handle}", flush=True)
    print(f"data_root={processed_root}", flush=True)
    return processed_root


def require_clearml() -> None:
    load_kaggle_clearml_secrets()
    try:
        from clearml import Task
    except ImportError as error:
        raise RuntimeError("ClearML is required but is not installed") from error

    if not _clearml_configured():
        raise RuntimeError(
            "ClearML is required. Configure clearml.conf or CLEARML_* environment variables."
        )
    if os.getenv("CLEARML_OFFLINE", "").lower() in {"1", "true"}:
        return
    try:
        Task.get_tasks(project_name=PROJECT_NAME)
    except Exception as error:
        raise RuntimeError(f"ClearML connection failed: {error}") from error


def load_kaggle_clearml_secrets() -> None:
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return

    client = UserSecretsClient()
    for name in (
        "CLEARML_API_HOST",
        "CLEARML_WEB_HOST",
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY",
    ):
        if os.getenv(name):
            continue
        try:
            value = client.get_secret(name)
        except Exception:
            continue
        if value:
            os.environ[name] = _clean_secret_value(value)


def _clean_secret_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def _clearml_configured() -> bool:
    return bool(
        os.getenv("CLEARML_OFFLINE", "").lower() in {"1", "true"}
        or os.getenv("CLEARML_API_HOST")
        or os.getenv("CLEARML_WEB_HOST")
        or os.path.exists(os.path.expanduser("~/clearml.conf"))
    )
