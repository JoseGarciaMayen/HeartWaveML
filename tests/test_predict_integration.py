import os

import numpy as np
import pytest

from src.config import SEQUENCES

MODEL = "src/saved_models/modelTransformer.keras"
SCALER = "src/saved_models/scaler_seq.joblib"
WINDOW = SEQUENCES.get("window", 45)

pytestmark = pytest.mark.integration

models_present = all(os.path.exists(p) for p in (MODEL, SCALER))
needs_models = pytest.mark.skipif(
    not models_present, reason="model files not pulled (run `dvc pull`)"
)


def _synthetic_window(window=WINDOW, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"signal": rng.uniform(-0.6, 0.6, 187).tolist(), "r_peak_sample": i * 300}
        for i in range(window)
    ]


@needs_models
def test_predict_returns_valid_label():
    from src.predict import predict

    label = predict(_synthetic_window(), model=MODEL)
    assert isinstance(label, int)
    assert label in {0, 1, 2}
