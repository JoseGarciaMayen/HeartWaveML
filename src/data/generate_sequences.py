"""Generate beat-sequence datasets for the BiLSTM model.

Reads data/interim/mitbih_features_only.csv (46 features + class + record),
applies the same class filtering as splitter.py, scales features (fit on train),
builds windows of W consecutive beats per patient with replication padding at
boundaries, and saves numpy arrays to data/processed/seq/.

Run after the 'preprocess' DVC stage (which generates the interim CSV):
    python -m src.data.generate_sequences
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import SEQUENCES
from src.data.splitter import CV_RECORDS, DS2_RECORDS, TRAIN_RECORDS

WINDOW = SEQUENCES.get("window", 5)
INTERIM_PATH = "data/interim/mitbih_features_only.csv"
OUT_DIR = "data/processed/seq"
SCALER_PATH = "src/saved_models/scaler_seq.joblib"


def generate_sequences(window: int = WINDOW, out_dir: str = OUT_DIR) -> None:
    K = window // 2

    df = pd.read_csv(INTERIM_PATH)
    df = df[df["class"] != 4].reset_index(drop=True)
    df["class"] = df["class"].replace({3: 2})

    feature_cols = [c for c in df.columns if c not in ("class", "record")]
    record = df["record"].astype(str)

    train_mask = record.isin(TRAIN_RECORDS).values
    cv_mask = record.isin(CV_RECORDS).values
    test_mask = record.isin(DS2_RECORDS).values

    X_all = df[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_all[train_mask] = scaler.fit_transform(X_all[train_mask])
    X_all[cv_mask] = scaler.transform(X_all[cv_mask])
    X_all[test_mask] = scaler.transform(X_all[test_mask])

    os.makedirs("src/saved_models", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    os.makedirs(out_dir, exist_ok=True)

    split_defs = [
        ("train", train_mask),
        ("cv", cv_mask),
        ("test", test_mask),
    ]

    for split_name, mask in split_defs:
        split_X = X_all[mask]
        split_y = df["class"].values[mask]
        split_record = record.values[mask]

        X_seqs, y_seqs, y_seq_seqs = [], [], []
        for rec in pd.unique(split_record):
            rec_pos = np.where(split_record == rec)[0]
            rec_X = split_X[rec_pos]
            rec_y = split_y[rec_pos]
            n = len(rec_X)
            for i in range(n):
                idxs = np.clip(np.arange(i - K, i + K + 1), 0, n - 1)
                X_seqs.append(rec_X[idxs])
                y_seqs.append(rec_y[i])
                y_seq_seqs.append(rec_y[idxs])

        X_arr = np.array(X_seqs, dtype=np.float32)
        y_arr = np.array(y_seqs, dtype=np.int32)
        y_seq_arr = np.array(y_seq_seqs, dtype=np.int32)
        np.save(f"{out_dir}/{split_name}_X.npy", X_arr)
        np.save(f"{out_dir}/{split_name}_y.npy", y_arr)
        np.save(f"{out_dir}/{split_name}_y_seq.npy", y_seq_arr)

        counts = np.bincount(y_arr, minlength=3)
        print(f"{split_name}: {X_arr.shape}  y_seq:{y_seq_arr.shape}  N={counts[0]}  S={counts[1]}  V={counts[2]}")

    print(f"\nScaler saved to {SCALER_PATH}")
    print(f"Sequences saved to {out_dir}/")


if __name__ == "__main__":
    generate_sequences()
