import functools

import joblib
import numpy as np
import pandas as pd

from src.config import SEQUENCES
from src.data.features import (
    _compute_rr_features,
    extract_features_from_beat,
    extract_features_from_dataframe,
)
from src.data.splitter import (
    CV_RECORDS,
    DS1_RECORDS,
    DS2_RECORDS,
    TRAIN_RECORDS,
    split_data,
)

__all__ = [
    "extract_features_from_beat",
    "extract_features_from_dataframe",
    "split_data",
    "preprocess_record",
    "DS1_RECORDS",
    "DS2_RECORDS",
    "CV_RECORDS",
    "TRAIN_RECORDS",
]

WINDOW = SEQUENCES.get("window", 45)


@functools.lru_cache(maxsize=1)
def _get_scaler():
    return joblib.load("src/saved_models/scaler_seq.joblib")


def preprocess_record(beats: list[dict]) -> np.ndarray:
    """Builds (n_beats, WINDOW, 46) scaled sliding windows over an ordered ECG recording.

    RR features are computed over the *entire* recording, exactly matching how
    training builds them (grouped by record) instead of an isolated fragment.
    """
    if len(beats) == 0:
        raise ValueError("beats must not be empty")

    feature_rows = [extract_features_from_beat(np.asarray(b["signal"])) for b in beats]
    features_df = pd.DataFrame(feature_rows)

    rr_input = pd.DataFrame(
        {
            "record": ["record"] * len(beats),
            "beat_center": [b["r_peak_sample"] for b in beats],
        }
    )
    rr_df = _compute_rr_features(rr_input)

    combined = pd.concat([features_df, rr_df.reset_index(drop=True)], axis=1)
    scaled = _get_scaler().transform(combined.values)

    n = len(beats)
    K = WINDOW // 2
    idxs = np.clip(np.arange(n)[:, None] + np.arange(-K, K + 1)[None, :], 0, n - 1)
    return scaled[idxs].astype(np.float32)
