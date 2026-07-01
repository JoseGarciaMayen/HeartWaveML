import atexit
import os
import sys

import joblib
import pandas as pd
from dotenv import load_dotenv

from src.tracking import clearml_get_best_params
from src.utils import get_class_weights

load_dotenv()
CANDIDATES_DIR = "src/saved_models/candidates"


class TrainerBase:
    def __init__(self, model_name, experiment_name):
        self.model_name = model_name
        self.experiment_name = experiment_name
        from src.tracking import init_clearml

        self.clearml_task = init_clearml(f"{self.experiment_name}_training")
        if self.clearml_task is not None:
            atexit.register(self._finalize_clearml)

    def _finalize_clearml(self):
        if sys.exc_info()[0] is not None:
            self.clearml_task.mark_stopped()
        else:
            self.clearml_task.mark_completed()
        self.clearml_task.close()

    def create_model(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_params(self, run_id=None):
        return clearml_get_best_params(f"{self.experiment_name}_tuning", run_id)

    def get_typed_params(self, best_params):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_data(self, train_path, cv_path):
        train_df = pd.read_csv(train_path)
        cv_df = pd.read_csv(cv_path)

        X_train = train_df.drop("class", axis=1)
        y_train = train_df["class"]

        X_cv = cv_df.drop("class", axis=1)
        y_cv = cv_df["class"]

        class_weights = get_class_weights(y_train)

        return X_train, y_train, X_cv, y_cv, class_weights

    def save_model(self, model, model_name):
        os.makedirs(CANDIDATES_DIR, exist_ok=True)
        joblib.dump(model, f"{CANDIDATES_DIR}/{model_name}.joblib")

    def run_training(self, model, X_train, y_train, X_cv, y_cv, class_weights, params):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def train(self):
        raise NotImplementedError("This method should be implemented in subclasses.")


XGB_PARAM_TYPES = {
    "max_depth": int,
    "n_estimators": int,
    "num_class": int,
    "random_state": int,
    "min_child_weight": int,
    "learning_rate": float,
    "subsample": float,
    "colsample_bytree": float,
    "colsample_bynode": float,
    "colsample_bylevel": float,
    "gamma": float,
    "reg_alpha": float,
    "reg_lambda": float,
}


class TrainerTreeBased(TrainerBase):
    """Base for tree-based tabular models (XGB, ExtraTrees, LightGBM, CatBoost).
    Subclasses must set PARAM_TYPES and implement create_model()."""

    PARAM_TYPES: dict = {}
    SUPPORTS_EVAL_SET: bool = True  # set to False for sklearn models (ExtraTrees, etc.)

    def get_typed_params(self, best_params: dict) -> dict:
        return {k: self.PARAM_TYPES.get(k, str)(v) for k, v in best_params.items()}

    def run_training(
        self, model, X_train, y_train, X_cv, y_cv, params, sample_weights=None, extra_params=None
    ):
        from src.tracking import clearml_log_params
        from src.utils import compute_and_log_metrics, notify_telegram

        clearml_log_params(params)
        if extra_params:
            clearml_log_params(extra_params)

        fit_kwargs = {}
        if self.SUPPORTS_EVAL_SET:
            fit_kwargs["eval_set"] = [(X_cv, y_cv)]
            fit_kwargs["verbose"] = False
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights
        model.fit(X_train, y_train, **fit_kwargs)

        metrics = compute_and_log_metrics(model, X_train, y_train, X_cv, y_cv)

        print(
            f"\n{self.model_name} CV -> val_f1_macro:{metrics['val_f1_macro']:.4f} "
            f"val_f1_weighted:{metrics['val_f1_weighted']:.4f}"
        )

        self.save_model(model, self.model_name)
        notify_telegram(
            f"{self.model_name} - val_f1_macro: {metrics['val_f1_macro']:.4f}, val_f1_weighted: {metrics['val_f1_weighted']:.4f}"
        )
        return metrics["val_f1_macro"]


if __name__ == "__main__":
    trainer = TrainerBase("modelXGB", "ECG_XGB")
    print(trainer.experiment_name)
    print(trainer.model_name)
    params = trainer.get_params(run_id="7fc96f05bf8f413688448e6adc9399d7")
    print(params)
