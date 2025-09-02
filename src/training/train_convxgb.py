from xgboost import XGBClassifier
from src.training.trainer import TrainerBase
import mlflow
import gc
from sklearn.metrics import f1_score, log_loss
from src.utils import notify_telegram
from mlflow.models import infer_signature

class TrainerCONVXGB(TrainerBase):
    def __init__(self,
                 model_name = "modelCONVXGB",
                 mlflow_experiment_name ="ECG_CONVX"):
        super().__init__(model_name, mlflow_experiment_name)

    def get_typed_params(self, best_params):
        param_types = {
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
            "reg_lambda": float
        }
        
        typed_best_params = {}
        for k, v in best_params.items():
            type = param_types.get(k, str)
            typed_best_params[k] = type(v)
        return typed_best_params
    
    def create_model(self):
        best_params = self.get_params()
        typed_best_params = self.get_typed_params(best_params)
        return XGBClassifier(**typed_best_params), typed_best_params
    
    def mlflow_start(self, model, X_train, y_train, X_cv, y_cv, params, params_cnn):

        with mlflow.start_run(nested=True):
            for k, v in params.items():
                mlflow.log_param(k, v)     

            for k, v in params_cnn.items():
                mlflow.log_param(k, v)       

            model.fit(X_train, y_train, verbose=True, eval_set=[(X_cv, y_cv)])

            acc = model.score(X_train, y_train)
            y_train_pred_prob = model.predict_proba(X_train)
            loss = log_loss(y_train, y_train_pred_prob)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("loss", loss)

            val_acc = model.score(X_cv, y_cv)
            y_cv_pred_prob = model.predict_proba(X_cv)
            val_loss = log_loss(y_cv, y_cv_pred_prob)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_loss", val_loss)

            y_cv_pred = model.predict(X_cv)
            val_f1 = f1_score(y_cv, y_cv_pred, average='macro')
            val_f1_weighted = f1_score(y_cv, y_cv_pred, average='weighted')
            mlflow.log_metric("val_f1_macro", val_f1)
            mlflow.log_metric("val_f1_weighted", val_f1_weighted)

            input_example = X_train.iloc[:5]
            signature = infer_signature(input_example, model.predict(input_example))
            mlflow.xgboost.log_model(model, self.model_name, registered_model_name=self.model_name, signature=signature)
            self.save_model(model, self.model_name)
            
            del model
            gc.collect()
            notify_telegram(f"Model logged with val_f1_macro: {val_f1:.4f}, val_f1_weighted: {val_f1_weighted:.4f}")
            return val_f1

    def train(self):
        """
        Function to train the XGBoost model with the best hyperparameters found by Optuna.
        It logs the model and metrics to MLflow.
        """
        X_train , y_train, X_cv, y_cv, class_weights = self.load_data('data/processed/cnn/mitbih_train_cnn.csv', 'data/processed/cnn/mitbih_cv_cnn.csv')

        params_cnn = {
        "l2": 0,
        "dropout": 0,
        "learning_rate_cnn": 0.01,
        "filters1": 16,
        "filters2": 32,
        "filters3": 64,
        }

        model, typed_best_params = self.create_model()

        self.mlflow_start(model, X_train, y_train, X_cv, y_cv, typed_best_params, params_cnn)

if __name__ == "__main__":
    trainer = TrainerCONVXGB()
    trainer.train()