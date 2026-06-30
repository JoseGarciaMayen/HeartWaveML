import os
from abc import ABC, abstractmethod

import mlflow
import optuna
from dotenv import load_dotenv

load_dotenv()
IP = os.getenv("IP", "127.0.0.1")


class TunerBase(ABC):
    """Base class for all Optuna hyperparameter tuners.
    Subclasses must implement objective() with the model-specific search space."""

    def __init__(self, experiment_name: str, n_trials: int, direction: str = "maximize"):
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.direction = direction
        mlflow.set_tracking_uri(f"http://{IP}:5000")
        mlflow.set_experiment(experiment_name)

    def load_data(self, train_path: str, cv_path: str):
        """Load train/cv CSVs. Return X_train, y_train, X_cv, y_cv."""
        from src.utils import load_splits

        return load_splits(train_path, cv_path)

    def log_metrics(self, model, X_train, y_train, X_cv, y_cv) -> dict:
        """Compute and log metrics to the active MLflow run. Return metrics dict."""
        from src.utils import compute_and_log_metrics

        return compute_and_log_metrics(model, X_train, y_train, X_cv, y_cv)

    @abstractmethod
    def objective(self, trial) -> float:
        """Define the Optuna search space. Must return the metric to optimize."""
        ...

    def run(self, pruner=None, enqueue_params: list[dict] | None = None):
        study = optuna.create_study(direction=self.direction, pruner=pruner)
        for params in enqueue_params or []:
            study.enqueue_trial(params)
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        return study
