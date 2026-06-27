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

from src.config import CNN_ARCH, DATA, FEATURE_EXTRACTOR
from src.utils import get_class_weights

np.random.seed(42)
tf.random.set_seed(42)


def _matched_filter_init(model, X_train_np, y_train_np):
    """Initialize the first 3 filters of the first Conv1D with per-class average waveforms.

    Each class average is subsampled to kernel_size=5 points and assigned to one
    filter slot, giving the network a head start on learning class-discriminative
    patterns (matched filter theory, Farag 2023).
    """
    first_conv = next(layer for layer in model.layers if isinstance(layer, Conv1D))
    W, b = first_conv.get_weights()  # W: (kernel_size=5, in_channels=1, n_filters)

    classes = sorted(np.unique(y_train_np))
    indices = np.linspace(0, X_train_np.shape[1] - 1, W.shape[0], dtype=int)
    for i, cls in enumerate(classes[: W.shape[2]]):  # only as many filters as classes
        template = X_train_np[y_train_np == cls].mean(axis=0)
        W[:, 0, i] = template[indices]

    first_conv.set_weights([W, b])


def create_model_cnn(
    input_shape=(187, 1), X_train=None, y_train=None, X_cv=None, y_cv=None, params=None
):
    """Builds and trains the CNN, then returns the feature extractor."""
    if params is None:
        params = CNN_ARCH

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
    num_classes = len(np.unique(y_train)) if y_train is not None else 3
    predictions = Dense(num_classes, activation="linear")(features_reduced)

    model = Model(inputs=input_cnn, outputs=predictions)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        metrics=["accuracy"],
    )

    # matched filter init
    X_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_np = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
    _matched_filter_init(model, X_np, y_np)

    # class-weighted training so the CNN learns S-class representations
    class_weights = get_class_weights(y_train)

    model.fit(
        X_train,
        y_train,
        class_weight=class_weights,
        epochs=FEATURE_EXTRACTOR["epochs"],
        batch_size=FEATURE_EXTRACTOR["batch_size"],
        validation_data=(X_cv, y_cv),
        verbose=1,
    )

    feature_extractor = Model(inputs=model.input, outputs=features)

    return feature_extractor


def main():
    train_df = pd.read_csv(DATA["base_train"])
    cv_df = pd.read_csv(DATA["base_cv"])

    X_train = train_df.drop("class", axis=1)
    y_train = train_df["class"]

    X_cv = cv_df.drop("class", axis=1)
    y_cv = cv_df["class"]

    feature_extractor = create_model_cnn(
        input_shape=(187, 1),
        X_train=X_train,
        y_train=y_train,
        X_cv=X_cv,
        y_cv=y_cv,
        params=CNN_ARCH,
    )

    os.makedirs("src/saved_models", exist_ok=True)
    tf.keras.models.save_model(feature_extractor, "src/saved_models/feature_extractor.keras")

    # Convert to ONNX to use it in the API and avoid the TensorFlow dependency there.
    spec = (tf.TensorSpec((None, 187, 1), tf.float32, name="input"),)
    output_path = "src/saved_models/feature_extractor.onnx"
    tf2onnx.convert.from_keras(
        feature_extractor, input_signature=spec, opset=13, output_path=output_path
    )


if __name__ == "__main__":
    main()
