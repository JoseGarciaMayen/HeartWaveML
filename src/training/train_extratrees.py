from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.config import DATA
from src.training.trainer import TrainerTreeBased

ET_PARAM_TYPES = {
    "n_estimators": int,
    "max_depth": int,
    "min_samples_split": int,
    "min_samples_leaf": int,
    "random_state": int,
}


class TrainerExtraTrees(TrainerTreeBased):
    PARAM_TYPES = ET_PARAM_TYPES
    SUPPORTS_EVAL_SET = False

    def __init__(self, model_name="modelExtraTrees", experiment_name="ECG_ExtraTrees"):
        super().__init__(model_name, experiment_name)

    def create_model(self):
        params = self.get_typed_params(self.get_params())
        return ExtraTreesClassifier(**params), params

    def train(self):
        X_train, y_train, X_cv, y_cv, _ = self.load_data(DATA["feat_train"], DATA["feat_cv"])
        sample_weights = compute_sample_weight("balanced", y_train)
        model, params = self.create_model()
        self.run_training(
            model, X_train, y_train, X_cv, y_cv, params, sample_weights=sample_weights
        )


if __name__ == "__main__":
    TrainerExtraTrees().train()
