import json
import os
import shutil

import joblib
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
    "cnn_mlp": {
        "model_path": "src/saved_models/modelCNNMLP.keras",
        "test_data": DATA["feat_test"],
        "predict": lambda model, X: np.argmax(
            model.predict([X.iloc[:, :187].values.reshape(-1, 187, 1), X.iloc[:, 187:].values]),
            axis=1,
        ),
    },
    "lstm": {
        "model_path": "src/saved_models/modelLSTM.keras",
        "load_data": lambda: (
            np.load("data/processed/seq_lstm/test_X.npy"),
            np.load("data/processed/seq_lstm/test_y.npy"),
        ),
        "predict": lambda model, X: np.argmax(model.predict(X, batch_size=512, verbose=0), axis=1),
    },
    "transformer": {
        "model_path": "src/saved_models/modelTransformer.keras",
        "load_data": lambda: (
            np.load("data/processed/seq/test_X.npy"),
            np.load("data/processed/seq/test_y.npy"),
        ),
        "predict": lambda model, X: np.argmax(
            model.predict(X, batch_size=512, verbose=0)[:, X.shape[1] // 2, :], axis=1
        ),
    },
    "seq2seq": {
        "model_path": "src/saved_models/modelSeq2Seq.keras",
        "load_data": lambda: (
            np.load("data/processed/seq/test_X.npy"),
            np.load("data/processed/seq/test_y.npy"),
        ),
        "predict": lambda model, X: np.argmax(
            model.predict(X, batch_size=512, verbose=0)[:, X.shape[1] // 2, :], axis=1
        ),
    },
}


def load_test_data(path: str):
    df = pd.read_csv(path)
    return df.drop("class", axis=1), df["class"]


def _load_model(model_path: str):
    if model_path.endswith(".keras"):
        import tensorflow as tf

        from src.utils import FocalLoss

        loss_instance = FocalLoss()
        return tf.keras.models.load_model(
            model_path,
            custom_objects={type(loss_instance).__name__: type(loss_instance)},
        )
    return joblib.load(model_path)


CLASS_NAMES = {0: "N", 1: "S", 2: "V"}

_NON_SCALAR_KEYS = ("cm", "labels")


def _compute_metrics(y_test, y_pred) -> dict:
    labels = sorted(set(int(c) for c in np.ravel(y_test)) | set(int(c) for c in np.ravel(y_pred)))
    f1_per = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_N": float(f1_per[0]),
        "f1_S": float(f1_per[1]),
        "f1_V": float(f1_per[2]),
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
    import matplotlib.pyplot as plt

    os.makedirs("src/saved_models/metrics", exist_ok=True)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["cm"],
        display_labels=[CLASS_NAMES.get(c, str(c)) for c in metrics["labels"]],
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix {name}")
    plt.savefig(f"src/saved_models/metrics/confusion_matrix_{name.lower()}.png")
    plt.close()


def evaluate_model(key: str, model_path: str | None = None) -> dict:
    cfg = EVAL_CONFIG[key]
    model_path = model_path or cfg["model_path"]
    if "load_data" in cfg:
        X_test, y_test_arr = cfg["load_data"]()
        y_test = pd.Series(y_test_arr)
    else:
        X_test, y_test = load_test_data(cfg["test_data"])
    model = _load_model(model_path)
    y_pred = cfg["predict"](model, X_test)
    metrics = _compute_metrics(y_test, y_pred)
    _print_metrics(key, metrics)
    _save_artifacts(key, metrics)
    return metrics


CANDIDATES_DIR = "src/saved_models/candidates"


def evaluate_all() -> dict:
    metrics_path = "src/saved_models/metrics/metrics.json"
    prev_summary = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            prev_summary = json.load(f)

    results = {}
    candidates = {}
    for key in EVAL_CONFIG:
        prod_path = EVAL_CONFIG[key]["model_path"]
        candidate_path = f"{CANDIDATES_DIR}/{os.path.basename(prod_path)}"
        has_candidate = os.path.exists(candidate_path)
        eval_path = candidate_path if has_candidate else prod_path
        if not os.path.exists(eval_path):
            print(f"\nSkipping {key}: model artifact not found at {eval_path}")
            continue
        try:
            results[key] = evaluate_model(key, eval_path)
            if has_candidate:
                candidates[key] = candidate_path
        except Exception as e:
            print(f"\nSkipping {key}: evaluation failed ({type(e).__name__}: {e})")
            continue

    if not results:
        raise SystemExit("No model artifacts found to evaluate. Train at least one model first")

    summary = dict(prev_summary)
    for key, metrics in results.items():
        if key not in candidates:
            print(f"[info] {key}: no candidate pending, evaluated current production model")
            continue

        candidate_path = candidates[key]
        prod_path = EVAL_CONFIG[key]["model_path"]
        new_f1 = metrics["f1_macro"]
        prev_f1 = float(prev_summary.get(key, {}).get("f1_macro", "-inf"))

        if new_f1 > prev_f1:
            summary[key] = {m: f"{v:.4f}" for m, v in metrics.items() if m not in _NON_SCALAR_KEYS}
            shutil.copy2(candidate_path, prod_path)
            print(f"[promoted] {key}: f1_macro {new_f1:.4f} > best {prev_f1:.4f}")
        else:
            print(f"[discarded] {key}: f1_macro {new_f1:.4f} <= best {prev_f1:.4f}")

        os.remove(candidate_path)

    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=4)
    return results


if __name__ == "__main__":
    evaluate_all()
