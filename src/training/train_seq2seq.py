import gc
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Input,
    TimeDistributed,
)
from tensorflow.keras.models import Model  # type: ignore

from src.config import configure_tf_threads
from src.tracking import clearml_log_metrics, clearml_log_params
from src.training.trainer import TrainerBase
from src.utils import (
    make_best_f1_restorer,
    make_class_weight_array,
    make_clearml_epoch_logger,
    notify_telegram,
)

configure_tf_threads()

SEQ_DIR = "data/processed/seq"


class TrainerSeq2Seq(TrainerBase):
    def __init__(self, model_name="modelSeq2Seq", experiment_name="ECG_Seq2Seq"):
        super().__init__(model_name, experiment_name)

    def get_typed_params(self, best_params):
        types = {
            "units1": int,
            "units2": int,
            "dense_units": int,
            "dropout": float,
            "learning_rate": float,
            "epochs": int,
        }
        return {k: types.get(k, str)(v) for k, v in best_params.items()}

    def load_data(self, *_):
        X_train = np.load(f"{SEQ_DIR}/train_X.npy")  # (N, W, 46)
        X_cv = np.load(f"{SEQ_DIR}/cv_X.npy")
        y_train_center = np.load(f"{SEQ_DIR}/train_y.npy")  # (N,)
        y_cv_center = np.load(f"{SEQ_DIR}/cv_y.npy")
        y_train_seq = np.load(f"{SEQ_DIR}/train_y_seq.npy")  # (N, W)
        y_cv_seq = np.load(f"{SEQ_DIR}/cv_y_seq.npy")
        sw_train = make_class_weight_array(y_train_center, shape_2d=y_train_seq.shape)
        sw_cv = make_class_weight_array(y_cv_center, shape_2d=y_cv_seq.shape)
        return X_train, y_train_center, X_cv, y_cv_center, y_train_seq, y_cv_seq, sw_train, sw_cv

    def create_model(self, window, n_features, p):
        clearml_log_params({**p, "window": window, "n_features": n_features})
        inp = Input(shape=(window, n_features))
        x = Bidirectional(LSTM(p["units1"], return_sequences=True))(inp)
        x = Dropout(p["dropout"])(x)
        x = Bidirectional(LSTM(p["units2"], return_sequences=True))(x)
        x = Dropout(p["dropout"])(x)
        x = TimeDistributed(Dense(p["dense_units"], activation="relu"))(x)
        x = TimeDistributed(Dropout(p["dropout"]))(x)
        output = TimeDistributed(Dense(3, activation="linear"))(x)
        model = Model(inputs=inp, outputs=output)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(p["learning_rate"]),
            metrics=["accuracy"],
            weighted_metrics=[],
            jit_compile=True,
        )
        return model

    def save_model(self, model, model_name):
        os.makedirs("src/saved_models/candidates", exist_ok=True)
        model.save(f"src/saved_models/candidates/{model_name}.keras")

    def run_training(
        self,
        model,
        X_train,
        y_train_center,
        X_cv,
        y_cv_center,
        y_train_seq,
        y_cv_seq,
        sw_train,
        sw_cv,
        p,
    ):
        center_idx = X_cv.shape[1] // 2
        best_f1 = make_best_f1_restorer(X_cv, y_cv_center, center_idx=center_idx)
        cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_macro", mode="max", patience=10, restore_best_weights=False
        )
        model.fit(
            X_train,
            y_train_seq,
            validation_data=(X_cv, y_cv_seq, sw_cv),
            epochs=p.get("epochs", 50),
            batch_size=256,
            callbacks=[best_f1, cb, make_clearml_epoch_logger()],
            sample_weight=sw_train,
            verbose=1,
        )

        logits = model.predict(X_cv, batch_size=512, verbose=0)  # (N, W, 3)
        y_pred = np.argmax(logits[:, center_idx, :], axis=1)
        val_f1 = f1_score(y_cv_center, y_pred, average="macro", zero_division=0)
        val_f1_per = f1_score(y_cv_center, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

        clearml_log_metrics(
            {
                "val_f1_macro": val_f1,
                "val_f1_N": float(val_f1_per[0]),
                "val_f1_S": float(val_f1_per[1]),
                "val_f1_V": float(val_f1_per[2]),
            }
        )
        print(
            f"\nSeq2Seq CV - F1-N:{val_f1_per[0]:.4f}  F1-S:{val_f1_per[1]:.4f}  "
            f"F1-V:{val_f1_per[2]:.4f}  macro:{val_f1:.4f}"
        )

        self.save_model(model, self.model_name)

        del model
        gc.collect()
        return val_f1

    def train(self):
        (X_train, y_train_center, X_cv, y_cv_center, y_train_seq, y_cv_seq, sw_train, sw_cv) = (
            self.load_data()
        )

        p = self.get_typed_params(self.get_params())
        window, n_features = X_train.shape[1], X_train.shape[2]
        model = self.create_model(window, n_features, p)
        val_f1 = self.run_training(
            model,
            X_train,
            y_train_center,
            X_cv,
            y_cv_center,
            y_train_seq,
            y_cv_seq,
            sw_train,
            sw_cv,
            p,
        )
        notify_telegram(f"Seq2Seq trained - val_f1_macro: {val_f1:.4f}")
        return val_f1


if __name__ == "__main__":
    TrainerSeq2Seq().train()
