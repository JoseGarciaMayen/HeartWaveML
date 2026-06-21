import gc

import mlflow
import numpy as np
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
from src.training.trainer import TrainerBase
from src.utils import notify_telegram


class TrainerCNNMLP(TrainerBase):
    def __init__(self, model_name="modelCNNMLP", mlflow_experiment_name="ECG_CNNMLP"):
        super().__init__(model_name, mlflow_experiment_name)

    def get_typed_params(self, best_params):
        param_types = {
            "l2": float,
            "dropout": float,
            "learning_rate": float,
            "filters": int,
            "filter_multiplier": int,
            "units_mlp1": int,
            "units_mlp2": int,
            "units_mlp3": int,
        }
        return {k: param_types.get(k, str)(v) for k, v in best_params.items()}

    def _split_inputs(self, X):
        return [X.iloc[:, :187].values.reshape(-1, 187, 1), X.iloc[:, 187:].values]

    def create_model(self, input_shape_cnn=(187, 1), input_shape_mlp=(36,), num_classes=4):
        p = self.get_typed_params(self.get_params())

        mlflow.log_params(
            {
                "input_shape_cnn": input_shape_cnn,
                "input_shape_mlp": input_shape_mlp,
                "num_classes": num_classes,
                "conv_layers": 4,
                **p,
            }
        )

        conv_filters = [
            p["filters"],
            p["filters"] * p["filter_multiplier"],
            p["filters"] * p["filter_multiplier"] ** 2,
            p["filters"] * p["filter_multiplier"] ** 2,
        ]
        x = Input(input_shape_cnn)
        input_cnn = x
        for f in conv_filters:
            x = Conv1D(
                f, 5, activation="relu", kernel_regularizer=regularizers.l2(p["l2"]), padding="same"
            )(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(p["dropout"])(x)
        x = Flatten()(x)

        input_mlp = Input(input_shape_mlp)
        y = input_mlp
        for units in [p["units_mlp1"], p["units_mlp2"], p["units_mlp3"], 64, 32]:
            y = Dense(units, activation="relu", kernel_regularizer=regularizers.l2(p["l2"]))(y)

        combined = concatenate([x, y])
        z = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(p["l2"]))(combined)
        z = Dropout(p["dropout"])(z)
        z = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(p["l2"]))(z)
        z = Dropout(p["dropout"])(z)
        output = Dense(num_classes, activation="linear")(z)

        model = Model(inputs=[input_cnn, input_mlp], outputs=output)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(p["learning_rate"]),
            metrics=["accuracy"],
        )
        return model

    def save_model(self, model, model_name):
        model.save(f"src/saved_models/{model_name}.keras")

    def mlflow_start(self, model, X_train, y_train, X_cv, y_cv, class_weights):
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False, silent=True)

        with mlflow.start_run(nested=True):
            model.fit(
                self._split_inputs(X_train),
                y_train,
                class_weight=class_weights,
                validation_data=(self._split_inputs(X_cv), y_cv),
                epochs=50,
            )

            loss, acc = model.evaluate(self._split_inputs(X_train), y_train, verbose=0)
            val_loss, val_acc = model.evaluate(self._split_inputs(X_cv), y_cv, verbose=0)
            y_cv_pred = np.argmax(model.predict(self._split_inputs(X_cv)), axis=1)
            val_f1 = f1_score(y_cv, y_cv_pred, average="macro")
            val_f1_weighted = f1_score(y_cv, y_cv_pred, average="weighted")

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

            mlflow.keras.log_model(model, self.model_name, registered_model_name=self.model_name)
            self.save_model(model, self.model_name)

            del model
            gc.collect()
            return val_f1

    def train(self):
        X_train, y_train, X_cv, y_cv, class_weights = self.load_data(
            DATA["feat_train"], DATA["feat_cv"]
        )
        model = self.create_model()
        val_f1 = self.mlflow_start(model, X_train, y_train, X_cv, y_cv, class_weights)
        return val_f1


if __name__ == "__main__":
    trainer = TrainerCNNMLP()
    val_f1 = trainer.train()
    notify_telegram(f"Nuevo modelo entrenado con f1Score de: {val_f1}")
