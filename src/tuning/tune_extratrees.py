from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.config import DATA, TUNING
from src.tracking import clearml_log_params
from src.tuning.tuner import TunerBase
from src.utils import notify_telegram


class TunerExtraTrees(TunerBase):
    def __init__(self):
        cfg = TUNING["extratrees"]
        super().__init__("ECG_ExtraTrees_tuning", n_trials=cfg["n_trials"])
        self.X_train, self.y_train, self.X_cv, self.y_cv = self.load_data(
            DATA["feat_train"], DATA["feat_cv"]
        )
        self.sample_weights = compute_sample_weight("balanced", self.y_train)
        self.search_space = cfg["search_space"]

    def objective(self, trial) -> float:
        search_space = self.search_space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *search_space["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *search_space["max_depth"]),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", *search_space["min_samples_split"]
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", *search_space["min_samples_leaf"]
            ),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": 42,
        }
        clearml_log_params(params)
        model = ExtraTreesClassifier(**params)
        model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)
        metrics = self.log_metrics(model, self.X_train, self.y_train, self.X_cv, self.y_cv)
        notify_telegram(
            f"ExtraTrees trial - f1: {metrics['val_f1_macro']:.4f}, f1_w: {metrics['val_f1_weighted']:.4f}"
        )
        return metrics["val_f1_macro"]


if __name__ == "__main__":
    TunerExtraTrees().run()
