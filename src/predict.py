import functools

import numpy as np

from src.config import SEQUENCES
from src.evaluate import _load_model
from src.preprocessing import preprocess_record

CENTER_IDX = SEQUENCES.get("window", 45) // 2


@functools.lru_cache(maxsize=4)
def _cached_model(model_path: str):
    return _load_model(model_path)


def predict_record(
    beats: list[dict], model: str = "src/saved_models/modelTransformer.keras"
) -> list[int]:
    """Classifies every beat of an ordered ECG recording."""
    X = preprocess_record(beats)
    keras_model = _cached_model(model)
    logits = keras_model.predict(X, batch_size=512, verbose=0)
    y_pred = np.argmax(logits[:, CENTER_IDX, :], axis=1)
    return y_pred.astype(int).tolist()
