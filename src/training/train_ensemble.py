import joblib
import mlflow

from src.config import DATA, ENSEMBLE
from src.ensemble import EnsembleModel
from src.training.trainer import TrainerBase
from src.utils import compute_and_log_metrics, notify_telegram

MODEL_FILES = {
    "xgb": "modelXGB.joblib",
    "extratrees": "modelExtraTrees.joblib",
    "lgbm": "modelLGBM.joblib",
    "catboost": "modelCatBoost.joblib",
}


class TrainerEnsemble(TrainerBase):
    def __init__(self, mlflow_experiment_name="ECG_Ensemble"):
        super().__init__("modelEnsemble", mlflow_experiment_name)

    def train(self):
        model_keys = ENSEMBLE["models"]
        weights = [ENSEMBLE["weights"][k] for k in model_keys]

        models = [joblib.load(f"src/saved_models/{MODEL_FILES[k]}") for k in model_keys]
        ensemble = EnsembleModel(models, weights)

        X_train, y_train, X_cv, y_cv, _ = self.load_data(DATA["feat_train"], DATA["feat_cv"])

        with mlflow.start_run():
            mlflow.log_param("models", model_keys)
            mlflow.log_param("weights", dict(zip(model_keys, weights, strict=True)))
            metrics = compute_and_log_metrics(ensemble, X_train, y_train, X_cv, y_cv)
            self.save_model(ensemble, self.model_name)
            notify_telegram(
                f"Ensemble - val_f1_macro: {metrics['val_f1_macro']:.4f}, val_f1_weighted: {metrics['val_f1_weighted']:.4f}"
            )


if __name__ == "__main__":
    TrainerEnsemble().train()
