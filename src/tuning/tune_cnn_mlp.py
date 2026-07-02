import gc

import numpy as np
import optuna
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import regularizers  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    concatenate,
)
from tensorflow.keras.models import Model  # type: ignore

from src.config import DATA, TUNING, configure_tf_threads
from src.tracking import clearml_log_metrics, clearml_log_params
from src.tuning.tuner import TunerBase
from src.utils import (
    FocalLoss,
    get_class_weights,
    make_best_f1_restorer,
    make_clearml_epoch_logger,
    notify_telegram,
)

configure_tf_threads()


class TunerCNNMLP(TunerBase):
    def __init__(self):
        cfg = TUNING["cnn_mlp"]
        super().__init__("ECG_CNNMLP_tuning", n_trials=cfg["n_trials"])
        self.search_space = cfg["search_space"]
        self._epochs_choices = cfg["epochs"]
        self._patience = cfg["patience"]
        self.X_train, self.y_train, self.X_cv, self.y_cv = self.load_data(
            DATA["feat_train"], DATA["feat_cv"]
        )
        self.X_train_cnn = self.X_train.iloc[:, :187].values.reshape(-1, 187, 1)
        self.X_train_mlp = self.X_train.iloc[:, 187:].values
        self.X_cv_cnn = self.X_cv.iloc[:, :187].values.reshape(-1, 187, 1)
        self.X_cv_mlp = self.X_cv.iloc[:, 187:].values
        self.n_mlp_features = self.X_train_mlp.shape[1]
        self.num_classes = self.y_train.nunique()
        self.class_weights = get_class_weights(self.y_train)

    def _build_model(
        self, trial, input_shape_cnn=(187, 1), input_shape_mlp=(36,), num_classes=3, task=None
    ):
        search_space = self.search_space
        l2 = trial.suggest_float("l2", *search_space["l2"], log=True)
        dropout = trial.suggest_float("dropout", *search_space["dropout"])
        learning_rate = trial.suggest_float(
            "learning_rate", *search_space["learning_rate"], log=True
        )
        gamma = trial.suggest_float("gamma", *search_space["gamma"])
        filters = trial.suggest_categorical("filters", [64])
        filter_multiplier = trial.suggest_categorical("filter_multiplier", [2])
        units_mlp1 = trial.suggest_categorical("units_mlp1", [512])
        units_mlp2 = trial.suggest_categorical("units_mlp2", [256])
        units_mlp3 = trial.suggest_categorical("units_mlp3", [128])

        clearml_log_params(
            {
                "input_shape_cnn": input_shape_cnn,
                "input_shape_mlp": input_shape_mlp,
                "num_classes": num_classes,
                "conv_layers": 4,
                "l2": l2,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "filters": filters,
                "filter_multiplier": filter_multiplier,
                "units_mlp1": units_mlp1,
                "units_mlp2": units_mlp2,
                "units_mlp3": units_mlp3,
            },
            task=task,
        )

        conv_filters = [
            filters,
            filters * filter_multiplier,
            filters * filter_multiplier**2,
            filters * filter_multiplier**2,
        ]
        input_cnn = Input(input_shape_cnn)
        x = input_cnn
        for f in conv_filters:
            x = Conv1D(
                f, 5, activation="relu", kernel_regularizer=regularizers.l2(l2), padding="same"
            )(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(dropout)(x)
        x = Flatten()(x)

        input_mlp = Input(input_shape_mlp)
        y = Dense(units_mlp1, activation="relu", kernel_regularizer=regularizers.l2(l2))(input_mlp)
        y = Dense(units_mlp2, activation="relu", kernel_regularizer=regularizers.l2(l2))(y)
        y = Dense(units_mlp3, activation="relu", kernel_regularizer=regularizers.l2(l2))(y)
        y = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2))(y)
        y = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2))(y)

        combined = concatenate([x, y])
        z = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2))(combined)
        z = Dropout(dropout)(z)
        z = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2))(z)
        z = Dropout(dropout)(z)
        output = Dense(num_classes, activation="linear")(z)

        model = Model(inputs=[input_cnn, input_mlp], outputs=output)
        model.compile(
            loss=FocalLoss(gamma=gamma),
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=["accuracy"],
            jit_compile=True,
        )
        return model

    def objective(self, trial, task=None) -> float:
        epochs = trial.suggest_categorical("epochs", self._epochs_choices)
        clearml_log_params({"epochs": epochs}, task=task)

        model = self._build_model(
            trial,
            input_shape_mlp=(self.n_mlp_features,),
            num_classes=self.num_classes,
            task=task,
        )
        best_f1 = make_best_f1_restorer([self.X_cv_cnn, self.X_cv_mlp], self.y_cv, center_idx=None)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_macro",
            mode="max",
            patience=self._patience,
            restore_best_weights=False,
        )
        model.fit(
            [self.X_train_cnn, self.X_train_mlp],
            self.y_train,
            class_weight=self.class_weights,
            validation_data=([self.X_cv_cnn, self.X_cv_mlp], self.y_cv),
            epochs=epochs,
            callbacks=[best_f1, early_stopping, make_clearml_epoch_logger(task=task)],
            verbose=2,
        )

        loss, acc = model.evaluate([self.X_train_cnn, self.X_train_mlp], self.y_train, verbose=0)
        val_loss, val_acc = model.evaluate([self.X_cv_cnn, self.X_cv_mlp], self.y_cv, verbose=0)
        y_cv_pred = np.argmax(model.predict([self.X_cv_cnn, self.X_cv_mlp]), axis=1)
        val_f1 = f1_score(self.y_cv, y_cv_pred, average="macro")
        val_f1_weighted = f1_score(self.y_cv, y_cv_pred, average="weighted")

        clearml_log_metrics(
            {
                "accuracy": acc,
                "loss": loss,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "val_f1_macro": val_f1,
                "val_f1_weighted": val_f1_weighted,
            },
            task=task,
        )
        notify_telegram(f"CNNMLP trial - f1: {val_f1:.4f}, f1_w: {val_f1_weighted:.4f}")

        del model
        gc.collect()
        tf.keras.backend.clear_session()
        return val_f1


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    TunerCNNMLP().run(pruner=pruner)
