import tensorflow as tf
import mlflow
import gc
from sklearn.metrics import f1_score
from src.utils import notify_telegram
import pandas as pd
from sklearn.metrics import f1_score
import mlflow
from xgboost import XGBClassifier
import optuna
import gc
from sklearn.metrics import log_loss
from dotenv import load_dotenv
import os

load_dotenv()
IP = os.getenv("IP")

mlflow.set_tracking_uri(f'http://{IP}:5000')
experiment_name = "ECG_CONVX_tuning"
mlflow.set_experiment(experiment_name)

train_features = pd.read_csv('data/processed/cnn/mitbih_train_cnn.csv')
X_train = train_features.drop('class', axis=1)
y_train = train_features['class']
cv_features = pd.read_csv('data/processed/cnn/mitbih_cv_cnn.csv')
X_cv = cv_features.drop('class', axis=1)
y_cv = cv_features['class']


def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
            'objective': 'multi:softprob',
            'num_class': 5,
            'random_state': 42
        }
    
    params_cnn = {
        "l2": 0,
        "dropout": 0,
        "learning_rate_cnn": 0.01,
        "filters1": 16,
        "filters2": 32,
        "filters3": 64,
    }

    with mlflow.start_run(nested=True):
        for k, v in params.items():
            mlflow.log_param(k, v)

        for k, v in params_cnn.items():
            mlflow.log_param(k, v)
            
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

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

        del model
        tf.keras.backend.clear_session()
        gc.collect()
        notify_telegram(f"New modelCONVX trained with f1Score: {val_f1}, f1Weighted: {val_f1_weighted}")
        return val_f1


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5000, show_progress_bar=True)

    
