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
}


def load_test_data(path: str):
    df = pd.read_csv(path)
    return df.drop("class", axis=1), df["class"]


def _load_model(model_path: str):
    if model_path.endswith(".keras"):
        import tensorflow as tf

        return tf.keras.models.load_model(model_path)
    return joblib.load(model_path)


def _compute_metrics(y_test, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f2_macro": fbeta_score(y_test, y_pred, beta=2, average="macro"),
        "f2_weighted": fbeta_score(y_test, y_pred, beta=2, average="weighted"),
        "cm": confusion_matrix(y_test, y_pred, normalize="true"),
    }


def _print_metrics(name: str, metrics: dict):
    print(f"\nModelo {name}:")
    for key, val in metrics.items():
        if key != "cm":
            print(f"  {key}: {val:.4f}")


def _save_artifacts(name: str, metrics: dict):
    os.makedirs("src/saved_models/metrics", exist_ok=True)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["cm"],
        display_labels=["N", "S", "V", "F"],
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
    results = {key: evaluate_model(key) for key in EVAL_CONFIG}
    summary = {
        k: {m: f"{v:.4f}" for m, v in metrics.items() if m != "cm"}
        for k, metrics in results.items()
    }
    with open("src/saved_models/metrics/metrics.json", "w") as f:
        json.dump(summary, f, indent=4)
    return results


if __name__ == "__main__":
    evaluate_all()
