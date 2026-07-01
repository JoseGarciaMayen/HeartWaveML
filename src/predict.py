import numpy as np

from src.config import SEQUENCES
from src.evaluate import _load_model
from src.preprocessing import preprocess_sequence

CENTER_IDX = SEQUENCES.get("window", 45) // 2


def predict(beats: list[dict], model: str = "src/saved_models/modelTransformer.keras") -> int:
    """Classifies the center beat of an ordered window of beats.

    Args:
        beats: ordered list of exactly WINDOW consecutive, already-segmented beats,
            each ``{"signal": [...187 floats...], "r_peak_sample": int}``.
        model: path to a Many-to-Many .keras model (Transformer/Seq2Seq).
    """
    X = preprocess_sequence(beats)
    keras_model = _load_model(model)
    logits = keras_model.predict(X, verbose=0)
    y_pred = np.argmax(logits[:, CENTER_IDX, :], axis=1)
    return int(y_pred[0])
