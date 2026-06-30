import gc

import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.layers import (  # type: ignore
    Add,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    TimeDistributed,
)
from tensorflow.keras.models import Model  # type: ignore

from src.training.trainer import TrainerBase
from src.utils import make_class_weight_array, notify_telegram

SEQ_DIR = "data/processed/seq"


def _transformer_block(x, num_heads, key_dim, ff_dim, dropout):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
    x2 = attn(x, x)
    x = LayerNormalization()(Add()([x, x2]))
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(x.shape[-1])(ff)
    x = LayerNormalization()(Add()([x, ff]))
    return x


class TrainerTransformer(TrainerBase):
    def __init__(self, model_name="modelTransformer", mlflow_experiment_name="ECG_Transformer"):
        super().__init__(model_name, mlflow_experiment_name)

    def get_typed_params(self, best_params):
        types = {
            "d_model": int,
            "num_heads": int,
            "key_dim": int,
            "ff_dim": int,
            "num_blocks": int,
            "dropout": float,
            "learning_rate": float,
            "epochs": int,
        }
        return {k: types.get(k, str)(v) for k, v in best_params.items()}

    def load_data(self, *_):
        X_train = np.load(f"{SEQ_DIR}/train_X.npy")
        X_cv = np.load(f"{SEQ_DIR}/cv_X.npy")
        y_train_center = np.load(f"{SEQ_DIR}/train_y.npy")
        y_cv_center = np.load(f"{SEQ_DIR}/cv_y.npy")
        y_train_seq = np.load(f"{SEQ_DIR}/train_y_seq.npy")
        y_cv_seq = np.load(f"{SEQ_DIR}/cv_y_seq.npy")
        sw_train = make_class_weight_array(y_train_center, shape_2d=y_train_seq.shape)
        sw_cv = make_class_weight_array(y_cv_center, shape_2d=y_cv_seq.shape)
        return X_train, y_train_center, X_cv, y_cv_center, y_train_seq, y_cv_seq, sw_train, sw_cv

    def create_model(self, window, n_features, p):
        mlflow.log_params({**p, "window": window, "n_features": n_features})
        inp = Input(shape=(window, n_features))
        x = Dense(p["d_model"])(inp)
        for _ in range(p["num_blocks"]):
            x = _transformer_block(x, p["num_heads"], p["key_dim"], p["ff_dim"], p["dropout"])
        x = TimeDistributed(Dense(32, activation="relu"))(x)
        x = TimeDistributed(Dropout(p["dropout"]))(x)
        output = TimeDistributed(Dense(3, activation="linear"))(x)
        model = Model(inputs=inp, outputs=output)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(p["learning_rate"]),
            metrics=["accuracy"],
            weighted_metrics=[],
        )
        return model

    def save_model(self, model, model_name):
        model.save(f"src/saved_models/{model_name}.keras")

    def mlflow_start(
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
        center_idx = X_train.shape[1] // 2
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False, silent=True)
        with mlflow.start_run(nested=True):
            cb = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
            model.fit(
                X_train,
                y_train_seq,
                validation_data=(X_cv, y_cv_seq, sw_cv),
                epochs=p.get("epochs", 50),
                batch_size=256,
                callbacks=[cb],
                sample_weight=sw_train,
                verbose=1,
            )

            logits = model.predict(X_cv, batch_size=512, verbose=0)
            y_pred = np.argmax(logits[:, center_idx, :], axis=1)
            val_f1 = f1_score(y_cv_center, y_pred, average="macro", zero_division=0)
            val_f1_per = f1_score(
                y_cv_center, y_pred, average=None, labels=[0, 1, 2], zero_division=0
            )

            mlflow.log_metrics(
                {
                    "val_f1_macro": val_f1,
                    "val_f1_N": float(val_f1_per[0]),
                    "val_f1_S": float(val_f1_per[1]),
                    "val_f1_V": float(val_f1_per[2]),
                }
            )
            print(
                f"\nTransformer CV - F1-N:{val_f1_per[0]:.4f}  F1-S:{val_f1_per[1]:.4f}  "
                f"F1-V:{val_f1_per[2]:.4f}  macro:{val_f1:.4f}"
            )

            mlflow.keras.log_model(model, self.model_name, registered_model_name=self.model_name)
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
        val_f1 = self.mlflow_start(
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
        notify_telegram(f"Transformer trained - val_f1_macro: {val_f1:.4f}")
        return val_f1


if __name__ == "__main__":
    TrainerTransformer().train()
