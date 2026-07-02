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


def _synthetic_recording(n=3 * WINDOW, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {"signal": rng.uniform(-0.6, 0.6, 187).tolist(), "r_peak_sample": i * 300} for i in range(n)
    ]


@needs_models
def test_predict_record_returns_valid_labels():
    from src.predict import predict_record

    labels = predict_record(_synthetic_recording(), model=MODEL)
    assert len(labels) == 3 * WINDOW
    assert all(isinstance(label, int) for label in labels)
    assert all(label in {0, 1, 2} for label in labels)
