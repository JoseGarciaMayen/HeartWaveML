import os

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

from src.utils import apply_filter, get_filter_coeffs

# Inter-patient split following de Chazal et al. (2004). DS1 is used for
# training/validation and DS2 for testing; no patient appears in both, which
# avoids inter-patient data leakage. The 4 paced records (102, 104, 107, 217)
# are excluded per the AAMI recommendation, so any record not listed here is
# dropped from the dataset.
DS1_RECORDS = {
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
}
DS2_RECORDS = {
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
}
# Validation records carved out of DS1. Chosen so all 5 classes (including the
# rare fusion class 3) are represented in cv, while keeping record 208 (which
# holds most of the fusion beats) in the training set.
CV_RECORDS = {"108", "205", "223"}
TRAIN_RECORDS = DS1_RECORDS - CV_RECORDS


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
    peaks, _ = find_peaks(beat_signal, height=np.max(beat_signal) * 0.3, distance=int(0.2 * fs))
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


def extract_features_from_dataframe():
    """
    Extracts features from a DataFrame of heartbeat signals.
    """
    features = []

    df = pd.read_csv("data/interim/mitbih_combined_records.csv")
    X = df.drop(["class", "record"], axis=1)

    print("Extracting features from dataset...")
    for i in range(len(X)):
        beat_signal = X.iloc[i].values
        features.append(extract_features_from_beat(beat_signal))
        if i % 5000 == 0:
            print(f"  Processed {i} of {len(X)} training heartbeats.")

    features = pd.DataFrame(features)
    print("Features shape:", features.shape)

    X_concat = pd.concat([df, features], axis=1)

    os.makedirs("data/interim", exist_ok=True)
    X_concat.to_csv("data/interim/mitbih_features.csv", index=False)

    features["class"] = df["class"]
    features["record"] = df["record"]
    features.to_csv("data/interim/mitbih_features_only.csv", index=False)

    print("Saved in the following archives:")
    print(" - data/interim/mitbih_features.csv")
    print(" - data/interim/mitbih_features_only.csv")


def split_data(path="data/interim/mitbih_combined_records.csv"):
    """
    Splits the dataset into training, validation and testing sets.
    Then, applies SMOTE, filtering and scaling.

    The split is patient-wise following the de Chazal DS1/DS2 partition (see
    ``DS1_RECORDS``/``DS2_RECORDS``): training and validation come from DS1 and
    the test set is DS2, so no patient appears in more than one set. This is the
    standard inter-patient paradigm and avoids inter-patient data leakage.
    """
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(path)
    record = df["record"].astype(str)
    X = df.drop(["class", "record"], axis=1)
    y = df["class"]

    train_mask = record.isin(TRAIN_RECORDS)
    cv_mask = record.isin(CV_RECORDS)
    test_mask = record.isin(DS2_RECORDS)
    # Beats from records in neither list (the paced ones) are dropped.

    X_train, y_train = X[train_mask], y[train_mask]
    X_cv, y_cv = X[cv_mask], y[cv_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    sampling_strategy_dict = {1: 5000, 2: 5000, 3: 5000, 4: 5000}

    smote = SMOTE(sampling_strategy=sampling_strategy_dict, random_state=42, k_neighbors=5)

    X_train, y_train = smote.fit_resample(X_train, y_train)

    b, a = get_filter_coeffs()
    X_train_filtered = np.apply_along_axis(apply_filter, axis=1, arr=X_train, b=b, a=a)
    X_cv_filtered = np.apply_along_axis(apply_filter, axis=1, arr=X_cv, b=b, a=a)
    X_test_filtered = np.apply_along_axis(apply_filter, axis=1, arr=X_test, b=b, a=a)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_cv_scaled = scaler.transform(X_cv_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)

    os.makedirs("src/saved_models", exist_ok=True)
    if path == "data/interim/mitbih_features.csv":
        joblib.dump(scaler, "src/saved_models/scaler.joblib")
    elif path == "data/interim/mitbih_combined_records.csv":
        joblib.dump(scaler, "src/saved_models/scaler_convxgb.joblib")

    train_df_processed = pd.DataFrame(X_train_scaled, columns=X.columns)
    cv_df_processed = pd.DataFrame(X_cv_scaled, columns=X.columns)
    test_df_processed = pd.DataFrame(X_test_scaled, columns=X.columns)

    train_df_processed["class"] = y_train.reset_index(drop=True)
    cv_df_processed["class"] = y_cv.reset_index(drop=True)
    test_df_processed["class"] = y_test.reset_index(drop=True)

    if path == "data/interim/mitbih_combined_records.csv":
        os.makedirs("data/processed/base", exist_ok=True)
        train_df_processed.to_csv("data/processed/base/mitbih_train.csv", index=False)
        cv_df_processed.to_csv("data/processed/base/mitbih_cv.csv", index=False)
        test_df_processed.to_csv("data/processed/base/mitbih_test.csv", index=False)

        print("Filtered and scaled data saved in:")
        print(" - data/processed/base/mitbih_train.csv")
        print(" - data/processed/base/mitbih_cv.csv")
        print(" - data/processed/base/mitbih_test.csv")

    elif path == "data/interim/mitbih_features.csv":
        os.makedirs("data/processed/feat", exist_ok=True)
        train_df_processed.to_csv("data/processed/feat/mitbih_train_features.csv", index=False)
        cv_df_processed.to_csv("data/processed/feat/mitbih_cv_features.csv", index=False)
        test_df_processed.to_csv("data/processed/feat/mitbih_test_features.csv", index=False)

        print("Filtered and scaled data saved in:")
        print(" - data/processed/feat/mitbih_train_features.csv")
        print(" - data/processed/feat/mitbih_cv_features.csv")
        print(" - data/processed/feat/mitbih_test_features.csv")

    elif path == "data/interim/mitbih_features_only.csv":
        os.makedirs("data/processed/feat_only", exist_ok=True)
        train_df_processed.to_csv(
            "data/processed/feat_only/mitbih_train_features_only.csv", index=False
        )
        cv_df_processed.to_csv("data/processed/feat_only/mitbih_cv_features_only.csv", index=False)
        test_df_processed.to_csv(
            "data/processed/feat_only/mitbih_test_features_only.csv", index=False
        )

        print("Filtered and scaled data saved in:")
        print(" - data/processed/feat_only/mitbih_train_features_only.csv")
        print(" - data/processed/feat_only/mitbih_cv_features_only.csv")
        print(" - data/processed/feat_only/mitbih_test_features_only.csv")


def feature_extracting():
    """
    Extracts features using a pre-trained CNN feature extractor and saves them to CSV files.
    """
    import tensorflow as tf

    feature_extractor = tf.keras.models.load_model("src/saved_models/feature_extractor.keras")
    train_df = pd.read_csv("data/processed/base/mitbih_train.csv")
    cv_df = pd.read_csv("data/processed/base/mitbih_cv.csv")
    test_df = pd.read_csv("data/processed/base/mitbih_test.csv")

    X_train = train_df.drop("class", axis=1)
    y_train = train_df["class"]
    X_cv = cv_df.drop("class", axis=1)
    y_cv = cv_df["class"]
    X_test = test_df.drop("class", axis=1)
    y_test = test_df["class"]

    X_train_features = feature_extractor.predict(X_train, batch_size=64, verbose=0)
    X_cv_features = feature_extractor.predict(X_cv, batch_size=64, verbose=0)
    X_test_features = feature_extractor.predict(X_test, batch_size=64, verbose=0)

    X_train_features_flattened = X_train_features.reshape(X_train_features.shape[0], -1)
    X_train_features = pd.DataFrame(X_train_features_flattened)
    X_train_features["class"] = y_train.reset_index(drop=True)
    X_cv_features_flattened = X_cv_features.reshape(X_cv_features.shape[0], -1)
    X_cv_features = pd.DataFrame(X_cv_features_flattened)
    X_cv_features["class"] = y_cv.reset_index(drop=True)
    X_test_features_flattened = X_test_features.reshape(X_test_features.shape[0], -1)
    X_test_features = pd.DataFrame(X_test_features_flattened)
    X_test_features["class"] = y_test.reset_index(drop=True)

    os.makedirs("data/processed/cnn", exist_ok=True)
    X_train_features.to_csv("data/processed/cnn/mitbih_train_cnn.csv", index=False)
    X_cv_features.to_csv("data/processed/cnn/mitbih_cv_cnn.csv", index=False)
    X_test_features.to_csv("data/processed/cnn/mitbih_test_cnn.csv", index=False)

    print("Features saved in:")
    print(" - data/processed/cnn/mitbih_train_cnn.csv")
    print(" - data/processed/cnn/mitbih_cv_cnn.csv")
    print(" - data/processed/cnn/mitbih_test_cnn.csv")


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
