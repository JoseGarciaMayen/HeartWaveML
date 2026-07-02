import numpy as np
import pytest

import src.predict as predict_module
from src.predict import CENTER_IDX, predict_record

MODEL = "src/saved_models/modelTransformer.keras"


@pytest.fixture(autouse=True)
def _clear_model_cache():
    yield
    predict_module._cached_model.cache_clear()


def _dummy_beats(n=50):
    return [{"signal": [0.0] * 187, "r_peak_sample": i * 300} for i in range(n)]


def test_predict_record_returns_one_label_per_beat(monkeypatch):
    n_beats = 50

    class StubModel:
        def predict(self, X, batch_size=512, verbose=0):
            n_classes = 3
            logits = np.zeros((X.shape[0], X.shape[1], n_classes))
            logits[:, CENTER_IDX, 1] = 10.0  # every window's center beat -> class S
            return logits

    monkeypatch.setattr(
        predict_module,
        "preprocess_record",
        lambda beats: np.zeros((len(beats), 45, 46)),
    )
    monkeypatch.setattr(predict_module, "_load_model", lambda path: StubModel())

    labels = predict_record(_dummy_beats(n_beats), model=MODEL)

    assert len(labels) == n_beats
    assert all(label == 1 for label in labels)
    assert all(isinstance(label, int) for label in labels)


def test_predict_record_rejects_empty_beats():
    with pytest.raises(ValueError):
        predict_record([], model=MODEL)
