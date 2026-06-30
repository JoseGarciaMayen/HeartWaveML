import gc

import mlflow
import numpy as np
import optuna
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

from src.tuning.tuner import TunerBase
from src.utils import make_best_f1_restorer, make_class_weight_array, notify_telegram

SEQ_DIR = "data/processed/seq"
N_TRIALS = 100


class TunerSeq2Seq(TunerBase):
    def __init__(self):
        super().__init__("ECG_Seq2Seq_tuning", n_trials=N_TRIALS)

        self.X_train = np.load(f"{SEQ_DIR}/train_X.npy")  # (N, W, 46)
        self.X_cv = np.load(f"{SEQ_DIR}/cv_X.npy")

        self.y_train_center = np.load(f"{SEQ_DIR}/train_y.npy")  # (N,) - for metrics
        self.y_cv_center = np.load(f"{SEQ_DIR}/cv_y.npy")

        self.y_train_seq = np.load(f"{SEQ_DIR}/train_y_seq.npy")  # (N, W) - for loss
        self.y_cv_seq = np.load(f"{SEQ_DIR}/cv_y_seq.npy")

        # sample_weight (N, 9): cada posición hereda el peso de la clase del beat central
        self.sw_train = make_class_weight_array(
            self.y_train_center, shape_2d=self.y_train_seq.shape
        )
        self.sw_cv = make_class_weight_array(self.y_cv_center, shape_2d=self.y_cv_seq.shape)

        self.window = self.X_train.shape[1]
        self.n_features = self.X_train.shape[2]
        self.center_idx = self.window // 2

    def _build_model(self, units1, units2, dense_units, dropout, lr):
        inp = Input(shape=(self.window, self.n_features))
        x = Bidirectional(LSTM(units1, return_sequences=True))(inp)
        x = Dropout(dropout)(x)
        x = Bidirectional(LSTM(units2, return_sequences=True))(x)
        x = Dropout(dropout)(x)
        x = TimeDistributed(Dense(dense_units, activation="relu"))(x)
        x = TimeDistributed(Dropout(dropout))(x)
        output = TimeDistributed(Dense(3, activation="linear"))(x)  # (N, 9, 3)
        model = Model(inputs=inp, outputs=output)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(lr),
            metrics=["accuracy"],
            weighted_metrics=[],
        )
        return model

    def objective(self, trial) -> float:
        units1 = trial.suggest_int("units1", 32, 256)
        units2 = trial.suggest_int("units2", 16, 128)
        dense_units = trial.suggest_int("dense_units", 16, 128)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_categorical("epochs", [20, 30, 50])

        params = {
            "units1": units1,
            "units2": units2,
            "dense_units": dense_units,
            "dropout": dropout,
            "learning_rate": lr,
            "epochs": epochs,
        }

        mlflow.tensorflow.autolog(log_models=False, log_datasets=False, silent=True)
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            model = self._build_model(units1, units2, dense_units, dropout, lr)
            best_f1 = make_best_f1_restorer(self.X_cv, self.y_cv_center, center_idx=self.center_idx)
            cb = tf.keras.callbacks.EarlyStopping(
                monitor="val_f1_macro", mode="max", patience=7, restore_best_weights=False
            )
            model.fit(
                self.X_train,
                self.y_train_seq,
                validation_data=(self.X_cv, self.y_cv_seq, self.sw_cv),
                epochs=epochs,
                batch_size=256,
                callbacks=[best_f1, cb],
                sample_weight=self.sw_train,
                verbose=0,
            )

            logits = model.predict(self.X_cv, batch_size=512, verbose=0)  # (N, W, 3)
            y_pred = np.argmax(logits[:, self.center_idx, :], axis=1)
            val_f1 = f1_score(self.y_cv_center, y_pred, average="macro", zero_division=0)
            val_f1_per = f1_score(
                self.y_cv_center, y_pred, average=None, labels=[0, 1, 2], zero_division=0
            )

            val_f1_sv = (float(val_f1_per[1]) + float(val_f1_per[2])) / 2
            mlflow.log_metrics(
                {
                    "val_f1_macro": val_f1,
                    "val_f1_sv": val_f1_sv,
                    "val_f1_N": float(val_f1_per[0]),
                    "val_f1_S": float(val_f1_per[1]),
                    "val_f1_V": float(val_f1_per[2]),
                }
            )
            notify_telegram(
                f"Seq2Seq trial - sv:{val_f1_sv:.4f} macro:{val_f1:.4f} "
                f"N:{val_f1_per[0]:.3f} S:{val_f1_per[1]:.3f} V:{val_f1_per[2]:.3f}"
            )

            del model
            gc.collect()
            tf.keras.backend.clear_session()
            return val_f1


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    TunerSeq2Seq().run(pruner=pruner)
