if __name__ == "__main__":
    from src.runtime import configure_runtime

    configure_runtime()

import optuna
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.config import DATA, TUNING, XGB_DEVICE
from src.tracking import clearml_log_metrics, clearml_log_params
from src.tuning.tuner import TunerBase
from src.utils import get_class_weights, notify_telegram


class TunerXGB(TunerBase):
    def __init__(self):
        cfg = TUNING["xgb"]
        super().__init__("ECG_XGB_tuning", n_trials=cfg["n_trials"])
        self.search_space = cfg["search_space"]

        self.X_train, self.y_train, self.X_cv, self.y_cv = self.load_data(
            DATA["feat_train"], DATA["feat_cv"]
        )
        class_weights = get_class_weights(self.y_train)
        self.sw_train = self.y_train.map(class_weights).values

    def objective(self, trial, task=None) -> float:
        ss = self.search_space
        params = {
            "max_depth": trial.suggest_int("max_depth", *ss["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *ss["learning_rate"], log=True),
            "n_estimators": trial.suggest_int("n_estimators", *ss["n_estimators"]),
            "subsample": trial.suggest_float("subsample", *ss["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *ss["colsample_bytree"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *ss["reg_alpha"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *ss["reg_lambda"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *ss["min_child_weight"]),
        }
        clearml_log_params(params, task=task)

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
            device=XGB_DEVICE,
            **params,
        )
        model.fit(self.X_train, self.y_train, sample_weight=self.sw_train, verbose=False)

        y_pred = model.predict(self.X_cv)
        val_f1 = f1_score(self.y_cv, y_pred, average="macro", zero_division=0)
        val_f1_per = f1_score(self.y_cv, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

        clearml_log_metrics(
            {
                "val_f1_macro": val_f1,
                "val_f1_N": float(val_f1_per[0]),
                "val_f1_S": float(val_f1_per[1]),
                "val_f1_V": float(val_f1_per[2]),
            },
            task=task,
        )
        notify_telegram(
            f"XGB trial - macro:{val_f1:.4f} N:{val_f1_per[0]:.3f} "
            f"S:{val_f1_per[1]:.3f} V:{val_f1_per[2]:.3f}"
        )
        return val_f1


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    TunerXGB().run(pruner=pruner)
