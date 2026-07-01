import gc
import os

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

from src.training.trainer import TrainerBase
from src.utils import FocalLoss, get_class_weights, make_best_f1_restorer, notify_telegram

# BiLSTM is many-to-one, uses its own small window (W=5), separate from the
# W=45 sequences that the many-to-many Transformer/Seq2Seq models consume.
SEQ_DIR = "data/processed/seq_lstm"


class TrainerLSTM(TrainerBase):
    def __init__(self, model_name="modelLSTM", mlflow_experiment_name="ECG_LSTM"):
        super().__init__(model_name, mlflow_experiment_name)

    def get_typed_params(self, best_params):
        types = {
            "units1": int,
            "units2": int,
            "dense_units": int,
            "dropout": float,
            "learning_rate": float,
            "gamma": float,
            "epochs": int,
        }
        return {k: types.get(k, str)(v) for k, v in best_params.items()}

    def load_data(self, *_):
        X_train = np.load(f"{SEQ_DIR}/train_X.npy")
        y_train = np.load(f"{SEQ_DIR}/train_y.npy")
        X_cv = np.load(f"{SEQ_DIR}/cv_X.npy")
        y_cv = np.load(f"{SEQ_DIR}/cv_y.npy")
        return X_train, y_train, X_cv, y_cv, get_class_weights(y_train)

    def create_model(self, window, n_features, num_classes=3):
        p = self.get_typed_params(self.get_params())
        mlflow.log_params({**p, "window": window, "n_features": n_features})

        inp = Input(shape=(window, n_features))
        x = Bidirectional(LSTM(p["units1"], return_sequences=True))(inp)
        x = Dropout(p["dropout"])(x)
        x = Bidirectional(LSTM(p["units2"]))(x)
        x = Dropout(p["dropout"])(x)
        x = Dense(p["dense_units"], activation="relu")(x)
        x = Dropout(p["dropout"])(x)
        output = Dense(num_classes, activation="linear")(x)

        model = Model(inputs=inp, outputs=output)
        model.compile(
            loss=FocalLoss(gamma=p["gamma"]),
            optimizer=tf.keras.optimizers.Adam(p["learning_rate"]),
            metrics=["accuracy"],
        )
        return model, p

    def save_model(self, model, model_name):
        os.makedirs("src/saved_models/candidates", exist_ok=True)
        model.save(f"src/saved_models/candidates/{model_name}.keras")

    def mlflow_start(self, model, X_train, y_train, X_cv, y_cv, class_weights, p):
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False, silent=True)

        with mlflow.start_run(nested=True):
            best_f1 = make_best_f1_restorer(X_cv, y_cv, center_idx=None)
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_f1_macro", mode="max", patience=10, restore_best_weights=False
            )
            model.fit(
                X_train,
                y_train,
                class_weight=class_weights,
                validation_data=(X_cv, y_cv),
                epochs=p.get("epochs", 50),
                batch_size=256,
                callbacks=[best_f1, early_stop],
                verbose=1,
            )

            y_cv_pred = np.argmax(model.predict(X_cv, batch_size=512, verbose=0), axis=1)
            val_f1 = f1_score(y_cv, y_cv_pred, average="macro")
            val_f1_per = f1_score(y_cv, y_cv_pred, average=None, labels=[0, 1, 2], zero_division=0)

            mlflow.log_metrics(
                {
                    "val_f1_macro": val_f1,
                    "val_f1_N": float(val_f1_per[0]),
                    "val_f1_S": float(val_f1_per[1]),
                    "val_f1_V": float(val_f1_per[2]),
                }
            )

            print(
                f"\nLSTM CV -> F1-N:{val_f1_per[0]:.4f}  F1-S:{val_f1_per[1]:.4f}  "
                f"F1-V:{val_f1_per[2]:.4f}  macro:{val_f1:.4f}"
            )

            mlflow.keras.log_model(model, self.model_name, registered_model_name=self.model_name)
            self.save_model(model, self.model_name)

            del model
            gc.collect()
            return val_f1

    def train(self):
        X_train, y_train, X_cv, y_cv, class_weights = self.load_data()
        window, n_features = X_train.shape[1], X_train.shape[2]
        model, p = self.create_model(window, n_features)
        val_f1 = self.mlflow_start(model, X_train, y_train, X_cv, y_cv, class_weights, p)
        notify_telegram(f"LSTM - val_f1_macro: {val_f1:.4f}")
        return val_f1


if __name__ == "__main__":
    TrainerLSTM().train()
