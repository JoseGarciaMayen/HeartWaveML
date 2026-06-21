import mlflow
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.config import DATA
from src.training.trainer import TrainerTreeBased

LGBM_PARAM_TYPES = {
    "n_estimators": int,
    "max_depth": int,
    "num_leaves": int,
    "random_state": int,
    "verbosity": int,
    "learning_rate": float,
    "subsample": float,
    "colsample_bytree": float,
    "reg_alpha": float,
    "reg_lambda": float,
}


class TrainerLGBM(TrainerTreeBased):
    PARAM_TYPES = LGBM_PARAM_TYPES
    SUPPORTS_EVAL_SET = False

    def __init__(self, model_name="modelLGBM", mlflow_experiment_name="ECG_LGBM"):
        super().__init__(model_name, mlflow_experiment_name)

    def _log_mlflow_model(self, model, input_example, signature):
        mlflow.lightgbm.log_model(
            model, self.model_name, registered_model_name=self.model_name, signature=signature
        )

    def create_model(self):
        params = self.get_typed_params(self.get_params())
        return LGBMClassifier(**params), params

    def train(self):
        X_train, y_train, X_cv, y_cv, _ = self.load_data(DATA["feat_train"], DATA["feat_cv"])
        sample_weights = compute_sample_weight("balanced", y_train)
        model, params = self.create_model()
        self.mlflow_start(
            model, X_train, y_train, X_cv, y_cv, params, sample_weights=sample_weights
        )


if __name__ == "__main__":
    TrainerLGBM().train()
