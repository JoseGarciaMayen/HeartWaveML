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
