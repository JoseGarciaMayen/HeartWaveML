import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew


def extract_features_from_beat(beat_signal, fs=360):
    """
    Extracts features from a single heartbeat signal.

    Args:
        beat_signal (numpy.ndarray): The heartbeat signal.
        fs (int): The sampling frequency of the signal. Default is 360.

    Returns:
        dict: A dictionary containing the extracted features.
    """
    features = {}

    if len(beat_signal) < 10 or np.all(beat_signal == 0):
        return features
    peaks, _ = find_peaks(beat_signal, height=np.max(beat_signal) * 0.1, distance=int(0.05 * fs))
    total_area = np.trapz(np.abs(beat_signal))
    features["mean"] = np.mean(beat_signal)
    features["std"] = np.std(beat_signal)
    features["var"] = np.var(beat_signal)
    features["median"] = np.median(beat_signal)
    features["mad"] = np.median(np.abs(beat_signal - np.median(beat_signal)))
    features["skewness"] = skew(beat_signal)
    features["kurtosis"] = kurtosis(beat_signal)
    features["max_val"] = np.max(beat_signal)
    features["min_val"] = np.min(beat_signal)
    features["range"] = np.max(beat_signal) - np.min(beat_signal)
    features["peak_to_peak"] = np.ptp(beat_signal)
    features["energy"] = np.sum(beat_signal**2)
    features["power"] = np.mean(beat_signal**2)
    features["rms"] = np.sqrt(np.mean(beat_signal**2))
    features["zero_crossings"] = len(np.where(np.diff(np.signbit(beat_signal)))[0])
    features["mean_crossings"] = len(
        np.where(np.diff(np.signbit(beat_signal - np.mean(beat_signal))))[0]
    )
    features["r_peak_std_ratio"] = features["max_val"] / (features["var"] ** 0.5 + 1e-6)
    features["num_peaks"] = len(peaks)
    features["r_peak_amplitude"] = np.max(beat_signal) if len(beat_signal) > 0 else 0
    features["r_peak_position"] = (
        np.argmax(beat_signal) / len(beat_signal) if len(beat_signal) > 0 else 0
    )
    features["total_area"] = total_area

    n_segments = 5
    segment_length = len(beat_signal) // n_segments

    for i in range(n_segments):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length if i < n_segments - 1 else len(beat_signal)
        segment = beat_signal[start_idx:end_idx]

        features[f"segment_{i}_mean"] = np.mean(segment)
        features[f"segment_{i}_std"] = np.std(segment)
        features[f"segment_{i}_area"] = np.trapz(np.abs(segment))

    return features


def _compute_rr_features(df):
    """
    Computes RR interval features from beat_center positions grouped by record.
    Returns a DataFrame with rr_pre, rr_post, rr_ratio aligned to df's index.
    """
    rr = pd.DataFrame(index=df.index, columns=["rr_pre", "rr_post", "rr_ratio"], dtype=float)
    for _, group in df.groupby("record", sort=False):
        centers = group["beat_center"].values
        n = len(centers)
        pre = np.empty(n)
        post = np.empty(n)
        pre[1:] = (centers[1:] - centers[:-1]) / 360.0
        post[:-1] = (centers[1:] - centers[:-1]) / 360.0
        pre[0] = post[0] if n > 1 else 0.0
        post[-1] = pre[-1] if n > 1 else 0.0
        ratio = pre / (post + 1e-6)
        rr.loc[group.index, "rr_pre"] = pre
        rr.loc[group.index, "rr_post"] = post
        rr.loc[group.index, "rr_ratio"] = ratio
    return rr


def extract_features_from_dataframe():
    """
    Extracts features from a DataFrame of heartbeat signals.
    """
    features = []

    df = pd.read_csv("data/interim/mitbih_combined_records.csv")
    signal_cols = [c for c in df.columns if c.startswith("sample_")]
    X = df[signal_cols]

    print("Extracting features from dataset...")
    for i in range(len(X)):
        beat_signal = X.iloc[i].values
        features.append(extract_features_from_beat(beat_signal))
        if i % 5000 == 0:
            print(f"  Processed {i} of {len(X)} training heartbeats.")

    features = pd.DataFrame(features)

    if "beat_center" in df.columns:
        rr = _compute_rr_features(df)
        features = pd.concat([features, rr.reset_index(drop=True)], axis=1)

    print("Features shape:", features.shape)

    X_concat = pd.concat(
        [
            df[signal_cols].reset_index(drop=True),
            features,
            df[["class", "record"]].reset_index(drop=True),
        ],
        axis=1,
    )

    os.makedirs("data/interim", exist_ok=True)
    X_concat.to_csv("data/interim/mitbih_features.csv", index=False)

    feat_only = features.copy()
    feat_only["class"] = df["class"].reset_index(drop=True)
    feat_only["record"] = df["record"].reset_index(drop=True)
    feat_only.to_csv("data/interim/mitbih_features_only.csv", index=False)

    print("Saved in the following archives:")
    print(" - data/interim/mitbih_features.csv")
    print(" - data/interim/mitbih_features_only.csv")
