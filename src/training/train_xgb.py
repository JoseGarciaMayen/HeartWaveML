from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import DATA
from src.training.trainer import XGB_PARAM_TYPES, TrainerTreeBased


class TrainerXGB(TrainerTreeBased):
    PARAM_TYPES = XGB_PARAM_TYPES

    def __init__(self, model_name="modelXGB", experiment_name="ECG_XGB"):
        super().__init__(model_name, experiment_name)

    def create_model(self):
        params = self.get_typed_params(self.get_params())
        return XGBClassifier(**params), params

    def train(self):
        X_train, y_train, X_cv, y_cv, _ = self.load_data(DATA["feat_train"], DATA["feat_cv"])
        sample_weights = compute_sample_weight("balanced", y_train)
        model, params = self.create_model()
        self.run_training(
            model, X_train, y_train, X_cv, y_cv, params, sample_weights=sample_weights
        )


if __name__ == "__main__":
    TrainerXGB().train()
