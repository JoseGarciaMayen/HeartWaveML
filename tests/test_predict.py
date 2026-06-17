import numpy as np

import src.predict as predict_module
from src.predict import predict

MODEL = "src/saved_models/modelCONVXGB.joblib"


def test_predict_returns_classifier_label(monkeypatch):
    class StubModel:
        def predict(self, X):
            return np.array([3])

    monkeypatch.setattr(predict_module.joblib, "load", lambda path: StubModel())
    monkeypatch.setattr(predict_module, "preprocess_convxgb", lambda beat: beat)

    label = predict(np.zeros(187), model=MODEL)

    assert label == 3
    assert isinstance(label, int)


def test_predict_rejects_unknown_model():
    import pytest

    with pytest.raises(ValueError):
        predict(np.zeros(187), model="does_not_exist.joblib")
