import os

import numpy as np
import pytest

MODEL = "src/saved_models/modelCONVXGB.joblib"
SCALER = "src/saved_models/scaler_convxgb.joblib"
EXTRACTOR = "src/saved_models/feature_extractor.onnx"
SAMPLES = "testHWML.csv"

pytestmark = pytest.mark.integration

models_present = all(os.path.exists(p) for p in (MODEL, SCALER, EXTRACTOR, SAMPLES))
needs_models = pytest.mark.skipif(not models_present, reason="model files not pulled (run `dvc pull`)")


def _load_sample_beats():
    beats = []
    with open(SAMPLES, encoding="utf-8-sig") as f:
        for line in f:
            values = [v for v in line.strip().split(";") if v]
            if len(values) == 187:
                beats.append(np.array([float(v) for v in values]))
    return beats


@needs_models
def test_predict_returns_valid_labels():
    from src.predict import predict

    for beat in _load_sample_beats():
        label = predict(beat, model=MODEL)
        assert isinstance(label, int)
        assert label in {0, 1, 2, 3, 4}
