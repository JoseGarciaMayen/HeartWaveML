import mlflow
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.config import DATA
from src.tuning.tuner import TunerBase
from src.utils import notify_telegram


class TunerCatBoost(TunerBase):
    def __init__(self):
        super().__init__("ECG_CatBoost_tuning", n_trials=100)
        self.X_train, self.y_train, self.X_cv, self.y_cv = self.load_data(
            DATA["feat_train"], DATA["feat_cv"]
        )
        self.sample_weights = compute_sample_weight("balanced", self.y_train)

    def objective(self, trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "verbose": 0,
            "random_state": 42,
        }
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            model = CatBoostClassifier(**params)
            model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)
            metrics = self.log_metrics(model, self.X_train, self.y_train, self.X_cv, self.y_cv)
            notify_telegram(f"CatBoost trial - f1: {metrics['val_f1_macro']:.4f}, f1_w: {metrics['val_f1_weighted']:.4f}")
            return metrics["val_f1_macro"]


if __name__ == "__main__":
    TunerCatBoost().run()
