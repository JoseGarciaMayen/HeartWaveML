import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import apply_filter, get_filter_coeffs

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

    df = pd.read_csv(path)
    df = df[df["class"] != 4].reset_index(drop=True)
    record = df["record"].astype(str)
    X = df.drop(["class", "record"], axis=1)
    y = df["class"]

    train_mask = record.isin(TRAIN_RECORDS)
    cv_mask = record.isin(CV_RECORDS)
    test_mask = record.isin(DS2_RECORDS)

    X_train, y_train = X[train_mask], y[train_mask]
    X_cv, y_cv = X[cv_mask], y[cv_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    sampling_strategy_dict = {1: 2500, 3: 2500}

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
