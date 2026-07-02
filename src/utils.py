import os
import re

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.signal import butter, filtfilt

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")


def get_record_numbers():
    """
    Returns a list of record numbers from the MIT-BIH Arrhythmia Database.
    """
    database_path = "data/raw/mit-bih-arrhythmia-database-1.0.0"

    record_numbers = []
    for filename in os.listdir(database_path):
        if filename.endswith(".atr"):
            match = re.match(r"^(\d+)\.atr$", filename)
            if match:
                record_numbers.append(match.group(1))

    return record_numbers


def get_class_mapping():
    """
    Returns a mapping of class labels to their corresponding integer values.
    """
    class_mapping = {
        "N": 0,
        "·": 0,
        "L": 0,
        "R": 0,
        "e": 0,
        "j": 0,
        "A": 1,
        "a": 1,
        "J": 1,
        "S": 1,
        "V": 2,
        "E": 2,
        "F": 3,
        "/": 4,
        "f": 4,
        "x": 4,
        "Q": 4,
        "|": 4,
        "~": 4,
    }
    return class_mapping


def get_filter_coeffs(cutoff_freq=40, fs=360, order=5):
    """
    Designs a Butterworth low-pass filter and returns its coefficients.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def apply_filter(data, b, a):
    """
    Applies a Butterworth filter to the given data.
    """
    return filtfilt(b, a, data)


def make_class_weight_array(y: np.ndarray, shape_2d: tuple | None = None) -> np.ndarray:
    """sample_weight (N, W) para modelos Seq2Seq. Cada posición hereda el peso del beat central."""
    cw = get_class_weights(y)
    w_flat = np.array([cw[int(c)] for c in y], dtype=np.float32)
    if shape_2d is None:
        return w_flat
    return np.tile(w_flat[:, None], (1, shape_2d[1]))


def get_class_weights(y_train):
    """
    Computes balanced class weights from the given training labels to handle
    class imbalance
    """
    y = pd.Series(y_train).reset_index(drop=True)
    unique_classes = sorted(y.unique())
    n, k = len(y), len(unique_classes)

    return {cls: n / (k * (y == cls).sum()) for cls in unique_classes}


def notify_telegram(msg):
    """
    Sends a notification message to a specified Telegram chat.
    """
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram notification skipped: TELEGRAM_TOKEN or CHAT_ID not set.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Telegram notification failed: {e}")


def load_splits(train_path: str, cv_path: str):
    """Load train/cv CSVs. Return X_train, y_train, X_cv, y_cv."""
    train_df = pd.read_csv(train_path)
    cv_df = pd.read_csv(cv_path)
    return (
        train_df.drop("class", axis=1),
        train_df["class"],
        cv_df.drop("class", axis=1),
        cv_df["class"],
    )


def FocalLoss(gamma: float = 2.0, **kwargs):
    """Return a sparse categorical focal loss instance TF 2.10 compatible."""
    import tensorflow as tf

    class _FocalLoss(tf.keras.losses.Loss):
        def __init__(self, gamma: float = 2.0, **kw):
            super().__init__(**kw)
            self.gamma = float(gamma)

        def call(self, y_true, y_pred):
            probs = tf.nn.softmax(y_pred, axis=-1)
            y_true_i = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            num_classes = tf.shape(y_pred)[-1]
            p_t = tf.reduce_sum(
                probs * tf.one_hot(y_true_i, num_classes, dtype=probs.dtype),
                axis=-1,
            )
            focal_weight = tf.pow(1.0 - p_t, self.gamma)
            xent = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
            return tf.reduce_mean(focal_weight * xent)

        def get_config(self):
            cfg = super().get_config()
            cfg["gamma"] = self.gamma
            return cfg

    return _FocalLoss(gamma=gamma, **kwargs)


def make_best_f1_restorer(X_val, y_val, center_idx=None, batch_size=512):
    """
    Keras callback that selects the best epoch by macro-F1
    """
    import tensorflow as tf
    from sklearn.metrics import f1_score

    class _BestF1Restorer(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = -1.0
            self.best_weights = None
            self.best_epoch = -1

        def on_epoch_end(self, epoch, logs=None):
            logs = logs if logs is not None else {}
            logits = self.model.predict(X_val, batch_size=batch_size, verbose=0)
            if center_idx is not None:
                logits = logits[:, center_idx, :]
            y_pred = np.argmax(logits, axis=1)
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            logs["val_f1_macro"] = f1
            if f1 > self.best:
                self.best = f1
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch

        def on_train_end(self, logs=None):
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                print(
                    f"\n[BestF1] restored epoch {self.best_epoch + 1} "
                    f"(val_f1_macro={self.best:.4f})"
                )

    return _BestF1Restorer()


def make_clearml_epoch_logger(task=None):
    """Keras callback that reports per-epoch train/val metrics to the given (or active) ClearML Task."""
    import tensorflow as tf

    class _ClearMLEpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            from clearml import Task

            active_task = task or Task.current_task()
            if active_task is None or not logs:
                return
            logger = active_task.get_logger()
            for k, v in logs.items():
                series, title = ("val", k[4:]) if k.startswith("val_") else ("train", k)
                logger.report_scalar(title=title, series=series, value=float(v), iteration=epoch)

    return _ClearMLEpochLogger()
