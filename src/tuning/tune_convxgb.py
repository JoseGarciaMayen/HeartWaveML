import mlflow
from xgboost import XGBClassifier

from src.config import CNN_ARCH, DATA
from src.tuning.tuner import TunerBase
from src.utils import notify_telegram


class TunerCONVXGB(TunerBase):
    def __init__(self):
        super().__init__("ECG_CONVX_tuning", n_trials=200)
        self.X_train, self.y_train, self.X_cv, self.y_cv = self.load_data(
            DATA["cnn_train"], DATA["cnn_cv"]
        )

    def objective(self, trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.01, 1.0, log=True),
            "objective": "multi:softprob",
            "num_class": 4,
            "random_state": 42,
        }
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_params(CNN_ARCH)
            model = XGBClassifier(**params)
            model.fit(self.X_train, self.y_train)
            metrics = self.log_metrics(model, self.X_train, self.y_train, self.X_cv, self.y_cv)
            notify_telegram(
                f"CONVXGB trial - f1: {metrics['val_f1_macro']:.4f}, f1_w: {metrics['val_f1_weighted']:.4f}"
            )
            return metrics["val_f1_macro"]


if __name__ == "__main__":
    TunerCONVXGB().run()
