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
        "data_path": lambda split: DATA[f"feat_{split}"],
        "predict": lambda model, X: np.argmax(
            model.predict([X.iloc[:, :187].values.reshape(-1, 187, 1), X.iloc[:, 187:].values]),
            axis=1,
        ),
    },
    "lstm": {
        "model_path": "src/saved_models/modelLSTM.keras",
        "load_data": lambda split: (
            np.load(f"data/processed/seq_lstm/{split}_X.npy"),
            np.load(f"data/processed/seq_lstm/{split}_y.npy"),
        ),
        "predict": lambda model, X: np.argmax(model.predict(X, batch_size=512, verbose=0), axis=1),
    },
    "transformer": {
        "model_path": "src/saved_models/modelTransformer.keras",
        "load_data": lambda split: (
            np.load(f"data/processed/seq/{split}_X.npy"),
            np.load(f"data/processed/seq/{split}_y.npy"),
        ),
        "predict": lambda model, X: np.argmax(
            model.predict(X, batch_size=512, verbose=0)[:, X.shape[1] // 2, :], axis=1
        ),
    },
    "seq2seq": {
        "model_path": "src/saved_models/modelSeq2Seq.keras",
        "load_data": lambda split: (
            np.load(f"data/processed/seq/{split}_X.npy"),
            np.load(f"data/processed/seq/{split}_y.npy"),
        ),
        "predict": lambda model, X: np.argmax(
            model.predict(X, batch_size=512, verbose=0)[:, X.shape[1] // 2, :], axis=1
        ),
    },
}


def load_csv_split(path: str):
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


def evaluate_model(key: str, model_path: str | None = None, split: str = "test") -> dict:
    cfg = EVAL_CONFIG[key]
    model_path = model_path or cfg["model_path"]
    if "load_data" in cfg:
        X, y_arr = cfg["load_data"](split)
        y = pd.Series(y_arr)
    else:
        X, y = load_csv_split(cfg["data_path"](split))
    model = _load_model(model_path)
    y_pred = cfg["predict"](model, X)
    metrics = _compute_metrics(y, y_pred)
    _print_metrics(f"{key} ({split})", metrics)
    if split == "test":
        _save_artifacts(key, metrics)
    return metrics


CANDIDATES_DIR = "src/saved_models/candidates"


def evaluate_all() -> dict:
    """Selection uses the CV split only. Test is evaluated once per run, purely
    for reporting, and never decides promotion (avoids test-set leakage)."""
    metrics_path = "src/saved_models/metrics/metrics.json"
    prev_summary = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            prev_summary = json.load(f)

    summary = dict(prev_summary)
    results = {}
    for key in EVAL_CONFIG:
        prod_path = EVAL_CONFIG[key]["model_path"]
        candidate_path = f"{CANDIDATES_DIR}/{os.path.basename(prod_path)}"
        has_candidate = os.path.exists(candidate_path)
        prod_exists = os.path.exists(prod_path)

        if not has_candidate and not prod_exists:
            print(f"\nSkipping {key}: model artifact not found")
            continue

        cv_f1 = None
        if has_candidate:
            try:
                cand_cv = evaluate_model(key, candidate_path, split="cv")["f1_macro"]
            except Exception as e:
                print(f"\nSkipping candidate {key}: cv evaluation failed ({type(e).__name__}: {e})")
                cand_cv = None

            if cand_cv is not None:
                prev_cv = float("-inf")
                if prod_exists:
                    prev_cv = evaluate_model(key, prod_path, split="cv")["f1_macro"]

                if cand_cv > prev_cv:
                    shutil.copy2(candidate_path, prod_path)
                    prod_exists = True
                    cv_f1 = cand_cv
                    print(f"[promoted] {key}: cv_f1_macro {cand_cv:.4f} > {prev_cv:.4f}")
                else:
                    cv_f1 = prev_cv
                    print(
                        f"[discarded] {key}: candidate cv_f1_macro {cand_cv:.4f} <= {prev_cv:.4f}"
                    )

            os.remove(candidate_path)

        if not prod_exists:
            print(
                f"\nSkipping {key}: candidate failed cv evaluation and no production model exists"
            )
            continue

        try:
            test_metrics = evaluate_model(key, prod_path, split="test")
        except Exception as e:
            print(f"\nSkipping {key}: test evaluation failed ({type(e).__name__}: {e})")
            continue

        if cv_f1 is None:
            cv_f1 = evaluate_model(key, prod_path, split="cv")["f1_macro"]

        results[key] = test_metrics
        summary[key] = {
            m: round(float(v), 4) for m, v in test_metrics.items() if m not in _NON_SCALAR_KEYS
        }
        summary[key]["cv_f1_macro"] = round(float(cv_f1), 4)

    if not results:
        raise SystemExit("No model artifacts found to evaluate. Train at least one model first")

    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=4)
    return results


if __name__ == "__main__":
    evaluate_all()
