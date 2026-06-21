import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Model  # type: ignore

np.random.seed(42)
tf.random.set_seed(42)


def create_model_cnn(
    input_shape=(187, 1), X_train=None, y_train=None, X_cv=None, y_cv=None, params=None
):
    """
    Function to create the CNNMLP model with dynamic hyperparameters.
    """
    if params is None:
        params = {}

    input_cnn = Input(input_shape)
    x = Conv1D(filters=params["filters1"], kernel_size=5, activation="relu", padding="same")(
        input_cnn
    )
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=params["filters2"], kernel_size=5, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=params["filters3"], kernel_size=5, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    features = GlobalAveragePooling1D()(x)

    features_reduced = Dense(64, activation="relu")(features)
    predictions = Dense(4, activation="linear")(features_reduced)

    model = Model(inputs=input_cnn, outputs=predictions)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(params["learning_rate_cnn"]),
        metrics=["accuracy"],
    )

    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_cv, y_cv), verbose=1)

    feature_extractor = Model(inputs=model.input, outputs=features)

    return feature_extractor


train_df = pd.read_csv("data/processed/base/mitbih_train.csv")
cv_df = pd.read_csv("data/processed/base/mitbih_cv.csv")

X_train = train_df.drop("class", axis=1)
y_train = train_df["class"]

X_cv = cv_df.drop("class", axis=1)
y_cv = cv_df["class"]

params_cnn = {
    "l2": 0,
    "dropout": 0,
    "learning_rate_cnn": 0.01,
    "filters1": 16,
    "filters2": 32,
    "filters3": 64,
}

feature_extractor = create_model_cnn(
    input_shape=(187, 1), X_train=X_train, y_train=y_train, X_cv=X_cv, y_cv=y_cv, params=params_cnn
)

os.makedirs("src/saved_models", exist_ok=True)
tf.keras.models.save_model(feature_extractor, "src/saved_models/feature_extractor.keras")

# Convert to ONNX to use it in API and avoid tensorflow dependency

spec = (tf.TensorSpec((None, 187, 1), tf.float32, name="input"),)
output_path = "src/saved_models/feature_extractor.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    feature_extractor, input_signature=spec, opset=13, output_path=output_path
)
