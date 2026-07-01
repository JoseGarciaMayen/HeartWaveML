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
    "preprocess_sequence",
    "DS1_RECORDS",
    "DS2_RECORDS",
    "CV_RECORDS",
    "TRAIN_RECORDS",
]

WINDOW = SEQUENCES.get("window", 45)


def preprocess_sequence(beats: list[dict]) -> np.ndarray:
    """Builds a (1, WINDOW, 46) scaled sequence for Seq2Seq/Transformer inference.

    Args:
        beats: ordered list of exactly WINDOW consecutive, already-segmented beats,
            each ``{"signal": [...187 floats...], "r_peak_sample": int}``.

    Returns:
        numpy.ndarray of shape (1, WINDOW, 46), scaled with the training-time
        `scaler_seq.joblib` (46 morphological + RR features, same column order
        as `data/interim/mitbih_features_only.csv`).
    """
    if len(beats) != WINDOW:
        raise ValueError(f"Expected exactly {WINDOW} beats, got {len(beats)}")

    feature_rows = [extract_features_from_beat(np.asarray(b["signal"])) for b in beats]
    features_df = pd.DataFrame(feature_rows)

    rr_input = pd.DataFrame(
        {
            "record": ["session"] * len(beats),
            "beat_center": [b["r_peak_sample"] for b in beats],
        }
    )
    rr_df = _compute_rr_features(rr_input)

    combined = pd.concat([features_df, rr_df.reset_index(drop=True)], axis=1)

    scaler = joblib.load("src/saved_models/scaler_seq.joblib")
    scaled = scaler.transform(combined.values)
    return scaled.reshape(1, len(beats), -1).astype(np.float32)
