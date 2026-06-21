import joblib
import mlflow
import numpy as np

from src.config import DATA, ENSEMBLE
from src.training.trainer import TrainerBase
from src.utils import compute_and_log_metrics, notify_telegram

MODEL_FILES = {
    "xgb": "modelXGB.joblib",
    "extratrees": "modelExtraTrees.joblib",
    "lgbm": "modelLGBM.joblib",
    "catboost": "modelCatBoost.joblib",
}


class EnsembleModel:
    """Soft-voting wrapper. Compatible with compute_and_log_metrics (predict_proba/predict/score)."""

    def __init__(self, models: list, weights: list):
        self.models = models
        self.weights = weights

    def predict_proba(self, X):
        return np.average(
            [m.predict_proba(X) for m in self.models], weights=self.weights, axis=0
        )

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))


class TrainerEnsemble(TrainerBase):
    def __init__(self, mlflow_experiment_name="ECG_Ensemble"):
        super().__init__("modelEnsemble", mlflow_experiment_name)

    def train(self):
        model_keys = ENSEMBLE["models"]
        weights = [ENSEMBLE["weights"][k] for k in model_keys]

        models = [
            joblib.load(f"src/saved_models/{MODEL_FILES[k]}") for k in model_keys
        ]
        ensemble = EnsembleModel(models, weights)

        X_train, y_train, X_cv, y_cv, _ = self.load_data(DATA["feat_train"], DATA["feat_cv"])

        with mlflow.start_run():
            mlflow.log_param("models", model_keys)
            mlflow.log_param("weights", dict(zip(model_keys, weights)))
            metrics = compute_and_log_metrics(ensemble, X_train, y_train, X_cv, y_cv)
            joblib.dump(
                {"models": model_keys, "weights": weights},
                "src/saved_models/ensemble_config.joblib",
            )
            notify_telegram(
                f"Ensemble - val_f1_macro: {metrics['val_f1_macro']:.4f}, val_f1_weighted: {metrics['val_f1_weighted']:.4f}"
            )


if __name__ == "__main__":
    TrainerEnsemble().train()
