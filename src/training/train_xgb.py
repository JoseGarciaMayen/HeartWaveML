from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.config import DATA
from src.tracking import clearml_log_metrics, clearml_log_params
from src.training.trainer import TrainerBase
from src.utils import notify_telegram


class TrainerXGB(TrainerBase):
    def __init__(self, model_name="modelXGB", experiment_name="ECG_XGB"):
        super().__init__(model_name, experiment_name)

    def get_typed_params(self, best_params):
        types = {
            "max_depth": int,
            "n_estimators": int,
            "min_child_weight": int,
            "learning_rate": float,
            "subsample": float,
            "colsample_bytree": float,
            "reg_alpha": float,
            "reg_lambda": float,
        }
        return {k: types.get(k, str)(v) for k, v in best_params.items()}

    def create_model(self, p):
        clearml_log_params(p)
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
            **p,
        )

    def run_training(self, model, X_train, y_train, X_cv, y_cv, sample_weight):
        model.fit(
            X_train, y_train, sample_weight=sample_weight, eval_set=[(X_cv, y_cv)], verbose=False
        )

        y_pred = model.predict(X_cv)
        val_f1 = f1_score(y_cv, y_pred, average="macro", zero_division=0)
        val_f1_per = f1_score(y_cv, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

        clearml_log_metrics(
            {
                "val_f1_macro": val_f1,
                "val_f1_N": float(val_f1_per[0]),
                "val_f1_S": float(val_f1_per[1]),
                "val_f1_V": float(val_f1_per[2]),
            }
        )
        print(
            f"\nXGB CV - F1-N:{val_f1_per[0]:.4f}  F1-S:{val_f1_per[1]:.4f}  "
            f"F1-V:{val_f1_per[2]:.4f}  macro:{val_f1:.4f}"
        )

        self.save_model(model, self.model_name)
        return val_f1

    def train(self):
        X_train, y_train, X_cv, y_cv, class_weights = self.load_data(
            DATA["feat_train"], DATA["feat_cv"]
        )
        sample_weight = y_train.map(class_weights).values
        p = self.get_typed_params(self.get_params())
        model = self.create_model(p)
        val_f1 = self.run_training(model, X_train, y_train, X_cv, y_cv, sample_weight)
        notify_telegram(f"XGB trained - val_f1_macro: {val_f1:.4f}")
        return val_f1


if __name__ == "__main__":
    TrainerXGB().train()
