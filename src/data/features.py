import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew


def extract_features_from_beat(beat_signal, fs=360):
    """
    Extracts features from a single heartbeat signal.
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


_RR_COLS = [
    "rr_pre",
    "rr_post",
    "rr_ratio",
    "rr_local_mean",
    "rr_global_mean",
    "rr_pre_norm_local",
    "rr_post_norm_local",
    "rr_pre_norm_global",
    "rr_local_std",
    "rr_rmssd",
]


def _compute_rr_features(df):
    """
    Computes 10 HRV-based RR interval features from beat_center positions
    grouped by record.
    """
    rr = pd.DataFrame(index=df.index, columns=_RR_COLS, dtype=float)

    for _, group in df.groupby("record", sort=False):
        centers = group["beat_center"].values
        n = len(centers)

        pre = np.empty(n)
        post = np.empty(n)
        if n > 1:
            intervals = (centers[1:] - centers[:-1]) / 360.0
            pre[1:] = intervals
            post[:-1] = intervals
            pre[0] = pre[1]
            post[-1] = post[-2]
        else:
            pre[0] = 0.0
            post[0] = 0.0

        ratio = pre / (post + 1e-6)

        s = pd.Series(pre)
        local_mean = s.rolling(window=10, min_periods=1).mean().to_numpy()
        global_mean = s.expanding(min_periods=1).mean().to_numpy()
        local_std = s.rolling(window=10, min_periods=2).std().fillna(0.0).to_numpy()
        rmssd = (
            s.diff()
            .pow(2)
            .rolling(window=5, min_periods=1)
            .mean()
            .apply(np.sqrt)
            .fillna(0.0)
            .to_numpy()
        )

        idx = group.index
        rr.loc[idx, "rr_pre"] = pre
        rr.loc[idx, "rr_post"] = post
        rr.loc[idx, "rr_ratio"] = ratio
        rr.loc[idx, "rr_local_mean"] = local_mean
        rr.loc[idx, "rr_global_mean"] = global_mean
        rr.loc[idx, "rr_pre_norm_local"] = pre / (local_mean + 1e-6)
        rr.loc[idx, "rr_post_norm_local"] = post / (local_mean + 1e-6)
        rr.loc[idx, "rr_pre_norm_global"] = pre / (global_mean + 1e-6)
        rr.loc[idx, "rr_local_std"] = local_std
        rr.loc[idx, "rr_rmssd"] = rmssd

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
