import numpy as np

from src.ensemble import EnsembleModel


class StubModel:
    """Returns fixed class probabilities regardless of X."""

    def __init__(self, proba):
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return np.repeat(self._proba[None, :], len(X), axis=0)


def test_predict_proba_applies_weights():
    a = StubModel([1.0, 0.0, 0.0, 0.0])
    b = StubModel([0.0, 0.0, 0.0, 1.0])
    ensemble = EnsembleModel([a, b], weights=[3, 1])

    X = np.zeros((2, 5))
    proba = ensemble.predict_proba(X)

    # Weighted average: 3/4 on class 0, 1/4 on class 3.
    assert proba.shape == (2, 4)
    np.testing.assert_allclose(proba[0], [0.75, 0.0, 0.0, 0.25])


def test_predict_returns_weighted_argmax():
    a = StubModel([0.1, 0.6, 0.2, 0.1])
    b = StubModel([0.7, 0.1, 0.1, 0.1])
    # With equal weights the averaged argmax favours class 0 (0.4 vs 0.35).
    ensemble = EnsembleModel([a, b], weights=[1, 1])

    preds = ensemble.predict(np.zeros((3, 5)))

    assert preds.tolist() == [0, 0, 0]


def test_score_matches_manual_accuracy():
    a = StubModel([0.9, 0.1, 0.0, 0.0])
    ensemble = EnsembleModel([a], weights=[1])

    y = np.array([0, 1, 0])
    assert ensemble.score(np.zeros((3, 5)), y) == 2 / 3


def test_ensemble_registered_in_eval_config():
    import src.evaluate as ev

    assert "ensemble" in ev.EVAL_CONFIG
    assert ev.EVAL_CONFIG["ensemble"]["model_path"].endswith("modelEnsemble.joblib")
