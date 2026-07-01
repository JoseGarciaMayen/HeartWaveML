import numpy as np


class EnsembleModel:
    """Soft-voting wrapper. Compatible with compute_and_log_metrics (predict_proba/predict/score).

    Kept in this lightweight module (numpy only) so a persisted ensemble can be reloaded for
    evaluation or serving without importing the training stack.
    """

    def __init__(self, models: list, weights: list):
        self.models = models
        self.weights = weights

    def predict_proba(self, X):
        return np.average([m.predict_proba(X) for m in self.models], weights=self.weights, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))
