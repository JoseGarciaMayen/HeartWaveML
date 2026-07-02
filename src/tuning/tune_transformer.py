import gc

import numpy as np
import optuna
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.layers import (  # type: ignore
    Dense,
    Dropout,
    Input,
    TimeDistributed,
)
from tensorflow.keras.models import Model  # type: ignore

from src.config import TUNING, configure_tf_threads
from src.tracking import clearml_log_metrics, clearml_log_params
from src.training.train_transformer import _transformer_block
from src.tuning.tuner import TunerBase
from src.utils import (
    make_best_f1_restorer,
    make_class_weight_array,
    make_clearml_epoch_logger,
    notify_telegram,
)

configure_tf_threads()

SEQ_DIR = "data/processed/seq"


class TunerTransformer(TunerBase):
    def __init__(self):
        cfg = TUNING["transformer"]
        super().__init__("ECG_Transformer_tuning", n_trials=cfg["n_trials"])
        self.search_space = cfg["search_space"]

        self.X_train = np.load(f"{SEQ_DIR}/train_X.npy")
        self.X_cv = np.load(f"{SEQ_DIR}/cv_X.npy")
        self.y_train_center = np.load(f"{SEQ_DIR}/train_y.npy")
        self.y_cv_center = np.load(f"{SEQ_DIR}/cv_y.npy")
        self.y_train_seq = np.load(f"{SEQ_DIR}/train_y_seq.npy")
        self.y_cv_seq = np.load(f"{SEQ_DIR}/cv_y_seq.npy")
        self.sw_train = make_class_weight_array(
            self.y_train_center, shape_2d=self.y_train_seq.shape
        )
        self.sw_cv = make_class_weight_array(self.y_cv_center, shape_2d=self.y_cv_seq.shape)
        self.window = self.X_train.shape[1]
        self.n_features = self.X_train.shape[2]
        self.center_idx = self.window // 2

    def _build_model(self, d_model, num_heads, key_dim, ff_dim, num_blocks, dropout, lr):
        inp = Input(shape=(self.window, self.n_features))
        x = Dense(d_model)(inp)
        for _ in range(num_blocks):
            x = _transformer_block(x, num_heads, key_dim, ff_dim, dropout)
        x = TimeDistributed(Dense(32, activation="relu"))(x)
        x = TimeDistributed(Dropout(dropout))(x)
        output = TimeDistributed(Dense(3, activation="linear"))(x)
        model = Model(inputs=inp, outputs=output)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(lr),
            metrics=["accuracy"],
            weighted_metrics=[],
            jit_compile=True,
        )
        return model

    def objective(self, trial) -> float:
        ss = self.search_space
        d_model = trial.suggest_categorical("d_model", ss["d_model"])
        num_heads = trial.suggest_categorical("num_heads", ss["num_heads"])
        key_dim = d_model // num_heads
        ff_dim = trial.suggest_categorical("ff_dim", ss["ff_dim"])
        num_blocks = trial.suggest_int("num_blocks", *ss["num_blocks"])
        dropout = trial.suggest_float("dropout", *ss["dropout"])
        lr = trial.suggest_float("learning_rate", *ss["learning_rate"], log=True)
        epochs = trial.suggest_categorical("epochs", ss["epochs"])

        params = {
            "d_model": d_model,
            "num_heads": num_heads,
            "key_dim": key_dim,
            "ff_dim": ff_dim,
            "num_blocks": num_blocks,
            "dropout": dropout,
            "learning_rate": lr,
            "epochs": epochs,
        }

        clearml_log_params(params)
        model = self._build_model(d_model, num_heads, key_dim, ff_dim, num_blocks, dropout, lr)
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
            callbacks=[best_f1, cb, make_clearml_epoch_logger()],
            sample_weight=self.sw_train,
            verbose=2,
        )

        logits = model.predict(self.X_cv, batch_size=512, verbose=0)
        y_pred = np.argmax(logits[:, self.center_idx, :], axis=1)
        val_f1 = f1_score(self.y_cv_center, y_pred, average="macro", zero_division=0)
        val_f1_per = f1_score(
            self.y_cv_center, y_pred, average=None, labels=[0, 1, 2], zero_division=0
        )

        val_f1_sv = (float(val_f1_per[1]) + float(val_f1_per[2])) / 2
        clearml_log_metrics(
            {
                "val_f1_macro": val_f1,
                "val_f1_sv": val_f1_sv,
                "val_f1_N": float(val_f1_per[0]),
                "val_f1_S": float(val_f1_per[1]),
                "val_f1_V": float(val_f1_per[2]),
            }
        )
        notify_telegram(
            f"Transformer trial - sv:{val_f1_sv:.4f} macro:{val_f1:.4f} "
            f"N:{val_f1_per[0]:.3f} S:{val_f1_per[1]:.3f} V:{val_f1_per[2]:.3f}"
        )

        del model
        gc.collect()
        tf.keras.backend.clear_session()
        return val_f1


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    TunerTransformer().run(pruner=pruner)
