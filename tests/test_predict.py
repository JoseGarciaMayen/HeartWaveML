import numpy as np
import pytest

import src.predict as predict_module
from src.predict import CENTER_IDX, predict

MODEL = "src/saved_models/modelTransformer.keras"


@pytest.fixture(autouse=True)
def _clear_model_cache():
    yield
    predict_module._cached_model.cache_clear()


def _dummy_beats(n=45):
    return [{"signal": [0.0] * 187, "r_peak_sample": i * 300} for i in range(n)]


def test_predict_returns_center_beat_label(monkeypatch):
    class StubModel:
        def predict(self, X, verbose=0):
            n_classes = 3
            logits = np.zeros((1, X.shape[1], n_classes))
            logits[0, CENTER_IDX, 1] = 10.0  # center beat -> class S
            return logits

    monkeypatch.setattr(
        predict_module, "preprocess_sequence", lambda beats: np.zeros((1, len(beats), 46))
    )
    monkeypatch.setattr(predict_module, "_load_model", lambda path: StubModel())

    label = predict(_dummy_beats(), model=MODEL)

    assert label == 1
    assert isinstance(label, int)


def test_predict_rejects_wrong_window_length():
    import pytest

    with pytest.raises(ValueError):
        predict(_dummy_beats(n=10), model=MODEL)
