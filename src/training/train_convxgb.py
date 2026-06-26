from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import CNN_ARCH, DATA
from src.training.trainer import XGB_PARAM_TYPES, TrainerTreeBased


class TrainerCONVXGB(TrainerTreeBased):
    PARAM_TYPES = XGB_PARAM_TYPES

    def __init__(self, model_name="modelCONVXGB", mlflow_experiment_name="ECG_CONVX"):
        super().__init__(model_name, mlflow_experiment_name)

    def create_model(self):
        params = self.get_typed_params(self.get_params())
        return XGBClassifier(**params), params

    def train(self):
        X_train, y_train, X_cv, y_cv, _ = self.load_data(DATA["cnn_train"], DATA["cnn_cv"])
        sample_weights = compute_sample_weight("balanced", y_train)
        model, params = self.create_model()
        self.mlflow_start(
            model,
            X_train,
            y_train,
            X_cv,
            y_cv,
            params,
            sample_weights=sample_weights,
            extra_params=CNN_ARCH,
        )


if __name__ == "__main__":
    TrainerCONVXGB().train()
