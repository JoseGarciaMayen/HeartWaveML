import mlflow
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.config import DATA, TUNING
from src.tuning.tuner import TunerBase
from src.utils import notify_telegram


class TunerLGBM(TunerBase):
    def __init__(self):
        cfg = TUNING["lgbm"]
        super().__init__("ECG_LGBM_tuning", n_trials=cfg["n_trials"])
        self.X_train, self.y_train, self.X_cv, self.y_cv = self.load_data(
            DATA["feat_train"], DATA["feat_cv"]
        )
        self.sample_weights = compute_sample_weight("balanced", self.y_train)
        self.search_space = cfg["search_space"]

    def objective(self, trial) -> float:
        search_space = self.search_space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *search_space["n_estimators"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", *search_space["learning_rate"], log=True
            ),
            "max_depth": trial.suggest_int("max_depth", *search_space["max_depth"]),
            "num_leaves": trial.suggest_int("num_leaves", *search_space["num_leaves"]),
            "subsample": trial.suggest_float("subsample", *search_space["subsample"]),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *search_space["colsample_bytree"]
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", *search_space["reg_alpha"], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *search_space["reg_lambda"], log=True),
            "verbosity": -1,
            "random_state": 42,
        }
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            model = LGBMClassifier(**params)
            model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)
            metrics = self.log_metrics(model, self.X_train, self.y_train, self.X_cv, self.y_cv)
            notify_telegram(
                f"LGBM trial - f1: {metrics['val_f1_macro']:.4f}, f1_w: {metrics['val_f1_weighted']:.4f}"
            )
            return metrics["val_f1_macro"]


if __name__ == "__main__":
    TunerLGBM().run()
