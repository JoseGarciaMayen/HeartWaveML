import gc

import mlflow
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

from src.config import DATA
from src.tuning.tuner import TunerBase
from src.utils import get_class_weights, notify_telegram


class TunerCNNMLP(TunerBase):
    def __init__(self):
        super().__init__("ECG_CNNMLP_tuning", n_trials=50)
        self.X_train, self.y_train, self.X_cv, self.y_cv = self.load_data(
            DATA["feat_train"], DATA["feat_cv"]
        )
        self.X_train_cnn = self.X_train.iloc[:, :187].values.reshape(-1, 187, 1)
        self.X_train_mlp = self.X_train.iloc[:, 187:].values
        self.X_cv_cnn = self.X_cv.iloc[:, :187].values.reshape(-1, 187, 1)
        self.X_cv_mlp = self.X_cv.iloc[:, 187:].values
        self.class_weights = get_class_weights(self.y_train)

    def _build_model(self, trial, input_shape_cnn=(187, 1), input_shape_mlp=(36,), num_classes=4):
        l2 = trial.suggest_float("l2", 0.0001, 0.0002, log=True)
        dropout = trial.suggest_float("dropout", 0, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.5, 5.0)
        filters = trial.suggest_categorical("filters", [64])
        filter_multiplier = trial.suggest_categorical("filter_multiplier", [2])
        units_mlp1 = trial.suggest_categorical("units_mlp1", [512])
        units_mlp2 = trial.suggest_categorical("units_mlp2", [256])
        units_mlp3 = trial.suggest_categorical("units_mlp3", [128])

        mlflow.log_params(
            {
                "input_shape_cnn": input_shape_cnn,
                "input_shape_mlp": input_shape_mlp,
                "num_classes": num_classes,
                "l2": l2,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "filters": filters,
                "filter_multiplier": filter_multiplier,
                "units_mlp1": units_mlp1,
                "units_mlp2": units_mlp2,
                "units_mlp3": units_mlp3,
            }
        )

        input_cnn = Input(input_shape_cnn)
        x = Conv1D(
            filters, 5, activation="relu", kernel_regularizer=regularizers.l2(l2), padding="same"
        )(input_cnn)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(dropout)(x)
        x = Conv1D(
            filters * filter_multiplier,
            5,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2),
            padding="same",
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
            loss=tf.keras.losses.SparseCategoricalFocalCrossentropy(from_logits=True, gamma=gamma),
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=["accuracy"],
        )
        return model

    def objective(self, trial) -> float:
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False, silent=True)
        with mlflow.start_run(nested=True):
            epochs = trial.suggest_categorical("epochs", [25, 50, 75])
            mlflow.log_param("epochs", epochs)

            model = self._build_model(trial)
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
            model.fit(
                [self.X_train_cnn, self.X_train_mlp],
                self.y_train,
                class_weight=self.class_weights,
                validation_data=([self.X_cv_cnn, self.X_cv_mlp], self.y_cv),
                epochs=epochs,
                callbacks=[early_stopping],
            )

            loss, acc = model.evaluate(
                [self.X_train_cnn, self.X_train_mlp], self.y_train, verbose=0
            )
            val_loss, val_acc = model.evaluate([self.X_cv_cnn, self.X_cv_mlp], self.y_cv, verbose=0)
            y_cv_pred = np.argmax(model.predict([self.X_cv_cnn, self.X_cv_mlp]), axis=1)
            val_f1 = f1_score(self.y_cv, y_cv_pred, average="macro")
            val_f1_weighted = f1_score(self.y_cv, y_cv_pred, average="weighted")

            mlflow.log_metrics(
                {
                    "accuracy": acc,
                    "loss": loss,
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "val_f1_macro": val_f1,
                    "val_f1_weighted": val_f1_weighted,
                }
            )
            notify_telegram(f"CNNMLP trial - f1: {val_f1:.4f}, f1_w: {val_f1_weighted:.4f}")

            del model
            gc.collect()
            tf.keras.backend.clear_session()
            return val_f1


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    TunerCNNMLP().run(pruner=pruner)
