import numpy as np
import pandas as pd
import pytest

from src.data import generate_sequences
from src.data.splitter import CV_RECORDS, DS1_RECORDS, DS2_RECORDS, TRAIN_RECORDS


def test_ds1_ds2_records_disjoint():
    assert DS1_RECORDS & DS2_RECORDS == set()


def test_train_records_excludes_ds2():
    assert TRAIN_RECORDS & DS2_RECORDS == set()


def test_cv_records_disjoint_from_train_and_test():
    assert CV_RECORDS & TRAIN_RECORDS == set()
    assert CV_RECORDS & DS2_RECORDS == set()


def test_train_and_cv_partition_ds1():
    assert TRAIN_RECORDS | CV_RECORDS == DS1_RECORDS
    assert TRAIN_RECORDS & CV_RECORDS == set()


@pytest.fixture
def two_record_csv(tmp_path, monkeypatch):
    rec_a, rec_b = sorted(TRAIN_RECORDS)[:2]
    rec_cv = next(iter(CV_RECORDS))
    rec_test = next(iter(DS2_RECORDS))
    rows = [
        {"marker": float(rec_cv), "class": 0, "record": rec_cv},
        {"marker": float(rec_test), "class": 0, "record": rec_test},
    ]
    for _ in range(10):
        rows.append({"marker": float(rec_a), "class": 0, "record": rec_a})
        rows.append({"marker": float(rec_b), "class": 0, "record": rec_b})
    df = pd.DataFrame(rows)

    csv_path = tmp_path / "features_only.csv"
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr(generate_sequences, "INTERIM_PATH", str(csv_path))
    monkeypatch.setattr(generate_sequences.joblib, "dump", lambda *a, **k: None)

    return rec_a, rec_b


def test_generate_sequences_windows_never_cross_records(tmp_path, two_record_csv):
    rec_a, rec_b = two_record_csv
    out_dir = tmp_path / "out"

    generate_sequences.generate_sequences(window=5, out_dir=str(out_dir))

    X = np.load(out_dir / "train_X.npy")  # (N, 5, 1)
    marker = X[:, :, 0]

    assert np.allclose(marker, marker[:, :1])
