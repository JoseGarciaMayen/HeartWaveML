import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model  # type: ignore

from src.config import TUNING
from src.tuning.tuner import TunerBase
from src.utils import FocalLoss, get_class_weights, make_best_f1_restorer, notify_telegram

# LSTM many-to-one uses its own W=5 window, matching train_lstm.py.
SEQ_DIR = "data/processed/seq_lstm"


def _build_model(window, n_features, params, num_classes=3):
    inp = Input(shape=(window, n_features))
    x = Bidirectional(LSTM(params["units1"], return_sequences=True))(inp)
    x = Dropout(params["dropout"])(x)
    x = Bidirectional(LSTM(params["units2"]))(x)
    x = Dropout(params["dropout"])(x)
    x = Dense(params["dense_units"], activation="relu")(x)
    x = Dropout(params["dropout"])(x)
    output = Dense(num_classes, activation="linear")(x)
    model = Model(inputs=inp, outputs=output)
    model.compile(
        loss=FocalLoss(gamma=params["gamma"]),
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        metrics=["accuracy"],
    )
    return model


class TunerLSTM(TunerBase):
    def __init__(self):
        cfg = TUNING["lstm"]
        super().__init__("ECG_LSTM_tuning", n_trials=cfg["n_trials"])
        self.X_train = np.load(f"{SEQ_DIR}/train_X.npy")
        self.y_train = np.load(f"{SEQ_DIR}/train_y.npy")
        self.X_cv = np.load(f"{SEQ_DIR}/cv_X.npy")
        self.y_cv = np.load(f"{SEQ_DIR}/cv_y.npy")
        self.class_weights = get_class_weights(self.y_train)
        self.window = self.X_train.shape[1]
        self.n_features = self.X_train.shape[2]
        self.search_space = cfg["search_space"]
        self._patience = cfg.get("patience", 10)

    def objective(self, trial) -> float:
        import gc

        ss = self.search_space
        params = {
            "units1": trial.suggest_int("units1", *ss["units1"]),
            "units2": trial.suggest_int("units2", *ss["units2"]),
            "dense_units": trial.suggest_int("dense_units", *ss["dense_units"]),
            "dropout": trial.suggest_float("dropout", *ss["dropout"]),
            "learning_rate": trial.suggest_float("learning_rate", *ss["learning_rate"], log=True),
            "gamma": trial.suggest_float("gamma", *ss["gamma"]),
            "epochs": trial.suggest_categorical("epochs", ss["epochs"]),
        }

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            model = _build_model(self.window, self.n_features, params)
            best_f1 = make_best_f1_restorer(self.X_cv, self.y_cv, center_idx=None)
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_f1_macro",
                mode="max",
                patience=self._patience,
                restore_best_weights=False,
            )
            model.fit(
                self.X_train,
                self.y_train,
                class_weight=self.class_weights,
                validation_data=(self.X_cv, self.y_cv),
                epochs=params["epochs"],
                batch_size=256,
                callbacks=[best_f1, early_stop],
                verbose=2,
            )
            y_pred = np.argmax(model.predict(self.X_cv, batch_size=512, verbose=0), axis=1)
            val_f1 = f1_score(self.y_cv, y_pred, average="macro", zero_division=0)
            mlflow.log_metric("val_f1_macro", val_f1)

            notify_telegram(f"LSTM trial - f1_macro: {val_f1:.4f}")

            del model
            gc.collect()
            tf.keras.backend.clear_session()

        return val_f1


if __name__ == "__main__":
    TunerLSTM().run()
