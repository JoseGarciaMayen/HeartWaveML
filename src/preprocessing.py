import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from src.data.features import extract_features_from_beat, extract_features_from_dataframe
from src.data.splitter import (
    CV_RECORDS,
    DS1_RECORDS,
    DS2_RECORDS,
    TRAIN_RECORDS,
    feature_extracting,
    split_data,
)
from src.utils import apply_filter, get_filter_coeffs

__all__ = [
    "extract_features_from_beat",
    "extract_features_from_dataframe",
    "split_data",
    "feature_extracting",
    "DS1_RECORDS",
    "DS2_RECORDS",
    "CV_RECORDS",
    "TRAIN_RECORDS",
    "preprocess_xgb",
    "preprocess_convxgb",
]


def preprocess_xgb(beat_signal):
    """
    Preprocesses a single heartbeat signal for prediction.
    Args:
        beat_signal (numpy.ndarray): The heartbeat signal.
    Returns:
        pandas.DataFrame: The preprocessed heartbeat signal with extracted features.
    """
    features = extract_features_from_beat(beat_signal)
    column_beats = [f"sample_{i}" for i in range(len(beat_signal))]
    column_feats = list(features.keys())
    features = np.array(list(features.values()))

    combined = np.concatenate([beat_signal, features])
    combined = combined.reshape(1, -1)

    scaler = joblib.load("src/saved_models/scaler.joblib")

    b, a = get_filter_coeffs()
    combined = np.apply_along_axis(apply_filter, axis=1, arr=combined, b=b, a=a)
    combined = scaler.transform(combined)
    combined = pd.DataFrame(combined, columns=column_beats + column_feats)

    return combined


def preprocess_convxgb(beat_signal):
    """
    Preprocesses a single heartbeat signal for prediction using a convolutional feature extractor.
    Args:
        beat_signal (numpy.ndarray): The heartbeat signal.
    Returns:
        pandas.DataFrame: The features extracted by the feature extractor.
    """
    scaler = joblib.load("src/saved_models/scaler_convxgb.joblib")

    b, a = get_filter_coeffs()
    beat_signal = np.apply_along_axis(apply_filter, axis=0, arr=beat_signal, b=b, a=a)
    beat_signal = scaler.transform(beat_signal.reshape(1, -1))

    feature_extractor = ort.InferenceSession("src/saved_models/feature_extractor.onnx")

    beat_signal = beat_signal.astype(np.float32).reshape(1, -1, 1)
    inputs = {"input": beat_signal}
    beat_signal = feature_extractor.run(None, inputs)[0]

    column_beats = [f"{i}" for i in range(beat_signal.shape[1])]
    beat_signal = pd.DataFrame(beat_signal, columns=column_beats)

    return beat_signal
