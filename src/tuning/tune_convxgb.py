from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import CNN_ARCH, DATA, TUNING
from src.tracking import clearml_log_params
from src.tuning.tuner import TunerBase
from src.utils import notify_telegram


class TunerCONVXGB(TunerBase):
    def __init__(self):
        cfg = TUNING["convxgb"]
        super().__init__("ECG_CONVX_tuning", n_trials=cfg["n_trials"])
        self.X_train, self.y_train, self.X_cv, self.y_cv = self.load_data(
            DATA["cnn_train"], DATA["cnn_cv"]
        )
        self.sample_weights = compute_sample_weight("balanced", self.y_train)
        self.search_space = cfg["search_space"]

    def objective(self, trial) -> float:
        search_space = self.search_space
        params = {
            "max_depth": trial.suggest_int("max_depth", *search_space["max_depth"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", *search_space["learning_rate"], log=True
            ),
            "n_estimators": trial.suggest_int("n_estimators", *search_space["n_estimators"]),
            "subsample": trial.suggest_float("subsample", *search_space["subsample"]),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *search_space["colsample_bytree"]
            ),
            "colsample_bynode": trial.suggest_float(
                "colsample_bynode", *search_space["colsample_bynode"]
            ),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", *search_space["colsample_bylevel"]
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", *search_space["reg_alpha"], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *search_space["reg_lambda"], log=True),
            "gamma": trial.suggest_float("gamma", *search_space["gamma"], log=True),
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
        }
        clearml_log_params(params)
        clearml_log_params({f"cnn_{k}": v for k, v in CNN_ARCH.items()})
        model = XGBClassifier(**params)
        model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)
        metrics = self.log_metrics(model, self.X_train, self.y_train, self.X_cv, self.y_cv)
        notify_telegram(
            f"CONVXGB trial - f1: {metrics['val_f1_macro']:.4f}, f1_w: {metrics['val_f1_weighted']:.4f}"
        )
        return metrics["val_f1_macro"]


if __name__ == "__main__":
    TunerCONVXGB().run()
