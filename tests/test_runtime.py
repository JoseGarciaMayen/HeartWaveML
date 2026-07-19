import sys
import types

import pytest

from src import runtime


def test_mount_processed_dataset_replaces_empty_data_directory(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "sample.csv").write_text("feature\n1\n")

    fake_kagglehub = types.SimpleNamespace(dataset_download=lambda handle: str(dataset_root))
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)

    data_root = tmp_path / "data" / "processed"
    data_root.mkdir(parents=True)
    result = runtime.mount_processed_dataset("owner/dataset", project_root=tmp_path)

    assert result == data_root
    assert result.is_symlink()
    assert result.resolve() == dataset_root


def test_mount_processed_dataset_rejects_non_empty_directory(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        types.SimpleNamespace(dataset_download=lambda handle: str(dataset_root)),
    )

    data_root = tmp_path / "data" / "processed"
    data_root.mkdir(parents=True)
    (data_root / "local.csv").write_text("local\n")

    with pytest.raises(RuntimeError, match="non-empty path"):
        runtime.mount_processed_dataset("owner/dataset", project_root=tmp_path)


def test_mount_processed_dataset_maps_root_feature_csvs_to_feat(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    for name in (
        "mitbih_train_features.csv",
        "mitbih_cv_features.csv",
        "mitbih_test_features.csv",
    ):
        (dataset_root / name).write_text("class\n0\n")
    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        types.SimpleNamespace(dataset_download=lambda handle: str(dataset_root)),
    )

    result = runtime.mount_processed_dataset("owner/dataset", project_root=tmp_path)

    assert result.is_dir()
    assert (result / "feat").is_symlink()
    assert (result / "feat" / "mitbih_train_features.csv").is_file()


def test_configure_runtime_requires_clearml(monkeypatch):
    monkeypatch.delenv("CLEARML_API_HOST", raising=False)
    monkeypatch.delenv("CLEARML_WEB_HOST", raising=False)
    monkeypatch.delenv("CLEARML_OFFLINE", raising=False)
    monkeypatch.setattr(runtime, "load_kaggle_clearml_secrets", lambda: None)
    monkeypatch.setattr(runtime, "_clearml_configured", lambda: False)

    with pytest.raises(RuntimeError, match="ClearML is required"):
        runtime.configure_runtime()
