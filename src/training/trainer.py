import mlflow
import joblib
from mlflow.tracking import MlflowClient
import pandas as pd
from src.utils import get_class_weights
from dotenv import load_dotenv
import os

load_dotenv()
IP = os.getenv("IP")

class TrainerBase:
    def __init__(self, model_name, mlflow_experiment_name,  mlflow_tracking_uri=f"http://{IP}:5000"):
        self.model_name = model_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(f"{ self.mlflow_experiment_name}_training")

    def create_model(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def get_params(self, run_id=None):
        client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)
        experiment = client.get_experiment_by_name(f"{ self.mlflow_experiment_name}_tuning")
        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["metrics.val_f1_macro DESC"],
            max_results=1,
        )
        if runs and run_id is None:
            best_run = runs[0]
            best_params = best_run.data.params
            return best_params
        elif run_id is not None:
            run = client.get_run(run_id)
            return run.data.params
        else:
            raise ValueError("No runs found in the experiment.")

    def get_typed_params(self, best_params):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_data(self, train_path, cv_path):
        train_df = pd.read_csv(train_path)
        cv_df = pd.read_csv(cv_path)

        X_train = train_df.drop('class', axis=1)
        y_train = train_df['class']

        X_cv = cv_df.drop('class', axis=1)
        y_cv = cv_df['class']

        class_weights = get_class_weights()

        return X_train, y_train, X_cv, y_cv, class_weights
    
    def save_model(self, model, model_name):
        joblib.dump(model, f"src/saved_models/{model_name}.joblib")
    
    def mlflow_start(self, model, X_train, y_train, X_cv, y_cv, class_weights, params):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def train(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
if __name__ == "__main__":
    trainer = TrainerBase("modelXGB", "ECG_XGB")
    print(trainer.mlflow_tracking_uri)
    print(trainer.mlflow_experiment_name)
    print(trainer.model_name)
    params = trainer.get_params(run_id="7fc96f05bf8f413688448e6adc9399d7")
    print(params)