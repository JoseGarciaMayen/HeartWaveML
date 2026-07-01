from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.config import DATA, TUNING
from src.tracking import clearml_log_params
from src.tuning.tuner import TunerBase
from src.utils import notify_telegram


class TunerCatBoost(TunerBase):
    def __init__(self):
        cfg = TUNING["catboost"]
        super().__init__("ECG_CatBoost_tuning", n_trials=cfg["n_trials"])
        self.X_train, self.y_train, self.X_cv, self.y_cv = self.load_data(
            DATA["feat_train"], DATA["feat_cv"]
        )
        self.sample_weights = compute_sample_weight("balanced", self.y_train)
        self.search_space = cfg["search_space"]

    def objective(self, trial) -> float:
        search_space = self.search_space
        params = {
            "iterations": trial.suggest_int("iterations", *search_space["iterations"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", *search_space["learning_rate"], log=True
            ),
            "depth": trial.suggest_int("depth", *search_space["depth"]),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *search_space["l2_leaf_reg"]),
            "border_count": trial.suggest_int("border_count", *search_space["border_count"]),
            "verbose": 0,
            "random_state": 42,
        }
        clearml_log_params(params)
        model = CatBoostClassifier(**params)
        model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)
        metrics = self.log_metrics(model, self.X_train, self.y_train, self.X_cv, self.y_cv)
        notify_telegram(
            f"CatBoost trial - f1: {metrics['val_f1_macro']:.4f}, f1_w: {metrics['val_f1_weighted']:.4f}"
        )
        return metrics["val_f1_macro"]


if __name__ == "__main__":
    TunerCatBoost().run()
