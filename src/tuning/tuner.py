import uuid
from abc import ABC, abstractmethod

import optuna
from dotenv import load_dotenv

load_dotenv()


class TunerBase(ABC):
    """Base class for all Optuna hyperparameter tuners.
    Subclasses must implement objective() with the model-specific search space."""

    def __init__(self, experiment_name: str, n_trials: int, direction: str = "maximize"):
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.direction = direction
        self.run_id = uuid.uuid4().hex[:8]

    def load_data(self, train_path: str, cv_path: str):
        """Load train/cv CSVs. Return X_train, y_train, X_cv, y_cv."""
        from src.utils import load_splits

        return load_splits(train_path, cv_path)

    @abstractmethod
    def objective(self, trial, task=None) -> float:
        """Define the Optuna search space. Must return the metric to optimize."""
        ...

    def _run_trial(self, trial):
        """Wrap objective() with one ClearML Task per trial."""
        from src.tracking import create_child_task

        clearml_task = create_child_task(
            f"{self.experiment_name}_{self.run_id}_trial{trial.number}",
            task_type="optimizer",
            tags=[self.experiment_name, self.run_id],
        )
        try:
            result = self.objective(trial, task=clearml_task)
        except BaseException:
            if clearml_task is not None:
                clearml_task.mark_stopped()
            raise
        else:
            if clearml_task is not None:
                clearml_task.mark_completed()
            return result
        finally:
            if clearml_task is not None:
                clearml_task.close()

    def run(self, pruner=None, enqueue_params: list[dict] | None = None):
        study = optuna.create_study(direction=self.direction, pruner=pruner)
        for params in enqueue_params or []:
            study.enqueue_trial(params)
        study.optimize(self._run_trial, n_trials=self.n_trials, show_progress_bar=True)
        return study
