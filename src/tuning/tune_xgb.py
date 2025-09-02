import pandas as pd
from sklearn.metrics import f1_score
import mlflow
from xgboost import XGBClassifier
import optuna
import gc
from sklearn.metrics import log_loss
from src.utils import notify_telegram
from sklearn.utils.class_weight import compute_sample_weight
from dotenv import load_dotenv
import os

load_dotenv()
IP = os.getenv("IP")

mlflow.set_tracking_uri(f'http://{IP}:5000')
experiment_name = "ECG_XGB_tuning"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

train_df = pd.read_csv('data/processed/feat/mitbih_train_features.csv')
cv_df = pd.read_csv('data/processed/feat/mitbih_cv_features.csv')

X_train = train_df.drop('class', axis=1)
y_train = train_df['class']

X_cv = cv_df.drop('class', axis=1)
y_cv = cv_df['class']

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'objective': 'multi:softprob',
        'num_class': 5,
        'random_state': 42
    }

    
    with mlflow.start_run(nested=True):
        try:
            for k, v in params.items():
                mlflow.log_param(k, v)

            mlflow.log_param("data", "mitbih_train_features.csv")
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=True, eval_set=[(X_cv, y_cv)])
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
            mlflow.log_metric("val_f1_macro", val_f1)
            val_f1_weighted = f1_score(y_cv, y_cv_pred, average='weighted')
            mlflow.log_metric("val_f1_weighted", val_f1_weighted)

            del model
            gc.collect()
            notify_telegram(f"Trial completed with GPU. \n f1: {val_f1:.4f}, f1_w: {val_f1_weighted:.4f}")
        finally:
            mlflow.end_run()
        return val_f1

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=500, show_progress_bar=True)
    
