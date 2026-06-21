import mlflow
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.config import DATA
from src.tuning.tuner import TunerBase
from src.utils import notify_telegram


class TunerExtraTrees(TunerBase):
    def __init__(self):
        super().__init__("ECG_ExtraTrees_tuning", n_trials=100)
        self.X_train, self.y_train, self.X_cv, self.y_cv = self.load_data(
            DATA["feat_train"], DATA["feat_cv"]
        )
        self.sample_weights = compute_sample_weight("balanced", self.y_train)

    def objective(self, trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": 42,
        }
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            model = ExtraTreesClassifier(**params)
            model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)
            metrics = self.log_metrics(model, self.X_train, self.y_train, self.X_cv, self.y_cv)
            notify_telegram(
                f"ExtraTrees trial - f1: {metrics['val_f1_macro']:.4f}, f1_w: {metrics['val_f1_weighted']:.4f}"
            )
            return metrics["val_f1_macro"]


if __name__ == "__main__":
    TunerExtraTrees().run()
