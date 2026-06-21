import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from src.config import DATA

EVAL_CONFIG = {
    "xgb": {
        "model_path": "src/saved_models/modelXGB.joblib",
        "test_data": DATA["feat_test"],
        "predict": lambda model, X: model.predict(X),
    },
    "cnn_mlp": {
        "model_path": "src/saved_models/modelCNNMLP.keras",
        "test_data": DATA["feat_test"],
        "predict": lambda model, X: np.argmax(
            model.predict([X.iloc[:, :187].values.reshape(-1, 187, 1), X.iloc[:, 187:].values]),
            axis=1,
        ),
    },
    "convxgb": {
        "model_path": "src/saved_models/modelCONVXGB.joblib",
        "test_data": DATA["cnn_test"],
        "predict": lambda model, X: model.predict(X),
    },
    "extratrees": {
        "model_path": "src/saved_models/modelExtraTrees.joblib",
        "test_data": DATA["feat_test"],
        "predict": lambda model, X: model.predict(X),
    },
    "lgbm": {
        "model_path": "src/saved_models/modelLGBM.joblib",
        "test_data": DATA["feat_test"],
        "predict": lambda model, X: model.predict(X),
    },
    "catboost": {
        "model_path": "src/saved_models/modelCatBoost.joblib",
        "test_data": DATA["feat_test"],
        "predict": lambda model, X: model.predict(X),
    },
    "ensemble": {
        "model_path": "src/saved_models/modelEnsemble.joblib",
        "test_data": DATA["feat_test"],
        "predict": lambda model, X: model.predict(X),
    },
}


def load_test_data(path: str):
    df = pd.read_csv(path)
    return df.drop("class", axis=1), df["class"]


def _load_model(model_path: str):
    if model_path.endswith(".keras"):
        import tensorflow as tf

        return tf.keras.models.load_model(model_path)
    return joblib.load(model_path)


CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

_NON_SCALAR_KEYS = ("cm", "labels")


def _compute_metrics(y_test, y_pred) -> dict:
    # Derive labels from the classes actually present so the confusion matrix and
    # its display labels always line up, even if a model predicts a class that is
    # no longer in the test set (e.g. an older 5-class model on 4-class data).
    labels = sorted(set(int(c) for c in y_test) | set(int(c) for c in y_pred))
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f2_macro": fbeta_score(y_test, y_pred, beta=2, average="macro"),
        "f2_weighted": fbeta_score(y_test, y_pred, beta=2, average="weighted"),
        "labels": labels,
        "cm": confusion_matrix(y_test, y_pred, labels=labels, normalize="true"),
    }


def _print_metrics(name: str, metrics: dict):
    print(f"\nModelo {name}:")
    for key, val in metrics.items():
        if key not in _NON_SCALAR_KEYS:
            print(f"  {key}: {val:.4f}")


def _save_artifacts(name: str, metrics: dict):
    os.makedirs("src/saved_models/metrics", exist_ok=True)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["cm"],
        display_labels=[CLASS_NAMES.get(c, str(c)) for c in metrics["labels"]],
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix {name}")
    plt.savefig(f"src/saved_models/metrics/confusion_matrix_{name.lower()}.png")
    plt.close()


def evaluate_model(key: str) -> dict:
    cfg = EVAL_CONFIG[key]
    X_test, y_test = load_test_data(cfg["test_data"])
    model = _load_model(cfg["model_path"])
    y_pred = cfg["predict"](model, X_test)
    metrics = _compute_metrics(y_test, y_pred)
    _print_metrics(key, metrics)
    _save_artifacts(key, metrics)
    return metrics


def evaluate_all() -> dict:
    results = {}
    for key in EVAL_CONFIG:
        model_path = EVAL_CONFIG[key]["model_path"]
        if not os.path.exists(model_path):
            print(f"\nSkipping {key}: model artifact not found at {model_path}")
            continue
        results[key] = evaluate_model(key)

    if not results:
        raise SystemExit("No model artifacts found to evaluate. Train at least one model first")

    summary = {
        k: {m: f"{v:.4f}" for m, v in metrics.items() if m not in _NON_SCALAR_KEYS}
        for k, metrics in results.items()
    }
    with open("src/saved_models/metrics/metrics.json", "w") as f:
        json.dump(summary, f, indent=4)
    return results


if __name__ == "__main__":
    evaluate_all()
