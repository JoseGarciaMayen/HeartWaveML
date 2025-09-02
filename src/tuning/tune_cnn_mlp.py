import pandas as pd
import numpy as np
from src.utils import get_class_weights
from sklearn.metrics import f1_score
import mlflow
import optuna
import gc
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, BatchNormalization, LSTM, Flatten, concatenate # type: ignore
from tensorflow.keras import regularizers # type: ignore
from src.utils import notify_telegram
from dotenv import load_dotenv
import os

load_dotenv()
IP = os.getenv("IP")

mlflow.set_tracking_uri(f'http://{IP}:5000')
experiment_name = "ECG_CNNMLP_tuning"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

train_df = pd.read_csv('data/processed/feat/mitbih_train_features.csv')
cv_df = pd.read_csv('data/processed/feat/mitbih_cv_features.csv')

X_train = train_df.drop('class', axis=1)
y_train = train_df['class']

X_cv = cv_df.drop('class', axis=1)
y_cv = cv_df['class']

X_train_cnn = X_train.iloc[:, :187].values.reshape(-1,187,1)
X_train_mlp = X_train.iloc[:, 187:].values
X_cv_cnn = X_cv.iloc[:, :187].values.reshape(-1,187,1)
X_cv_mlp = X_cv.iloc[:, 187:].values

class_weights = get_class_weights()


def create_model(trial, input_shape_cnn=(187, 1), input_shape_mlp=(36,), num_classes=5):
    """
    Function to create the CNNMLP model with dynamic hyperparameters.
    """
    mlflow.log_param('input_shape_cnn', input_shape_cnn)
    mlflow.log_param('input_shape_mlp', input_shape_mlp)
    mlflow.log_param('num_classes', num_classes)

    l2 = trial.suggest_float('l2', 0.0001, 0.0002, log=True)
    mlflow.log_param('l2', l2)

    dropout = trial.suggest_float('dropout', 0, 0.4)
    mlflow.log_param('dropout', dropout)

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    mlflow.log_param('learning_rate', learning_rate)

    filters = trial.suggest_categorical('filters', [64])
    filter_multiplier = trial.suggest_categorical('filter_multiplier', [2])
    mlflow.log_param('filters', filters)
    mlflow.log_param('filter_multiplier', filter_multiplier)

    lstm_units = trial.suggest_categorical('lstm_units', [0])
    mlflow.log_param('lstm_units', lstm_units)
    
    units_mlp1 = trial.suggest_categorical('units_mlp1', [512])
    units_mlp2 = trial.suggest_categorical('units_mlp2', [256])
    units_mlp3 = trial.suggest_categorical('units_mlp3', [128])
    mlflow.log_param('units_mlp1', units_mlp1)
    mlflow.log_param('units_mlp2', units_mlp2)
    mlflow.log_param('units_mlp3', units_mlp3)
    mlflow.log_param('conv_layers', 2)

    input_cnn = Input(input_shape_cnn)
    x = Conv1D(filters=filters, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(l2), padding='same')(input_cnn)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=filters*filter_multiplier, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(l2), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    input_mlp = Input(input_shape_mlp)
    y = Dense(units_mlp1, activation='relu', kernel_regularizer=regularizers.l2(l2))(input_mlp)
    y = Dense(units_mlp2, activation='relu', kernel_regularizer=regularizers.l2(l2))(y)
    y = Dense(units_mlp3, activation='relu', kernel_regularizer=regularizers.l2(l2))(y)
    y = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2))(y)
    y = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2))(y)

    combined = concatenate([x, y])

    z = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2))(combined)
    z = Dropout(dropout)(z)
    z = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2))(z)
    z = Dropout(dropout)(z)
    
    output = Dense(num_classes, activation='linear')(z)

    model = Model(inputs=[input_cnn, input_mlp], outputs=output)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=['accuracy']
    )

    return model
    
def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    """

    mlflow.tensorflow.autolog(
        log_models=False,
        log_datasets=False,
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=True
    )
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        epochs = trial.suggest_categorical('epochs', [25, 50, 75])
        mlflow.log_param('epochs', epochs)
        model = create_model(trial)
        model.fit(
                [X_train_cnn, X_train_mlp], 
                y_train, 
                class_weight=class_weights, 
                validation_data=([X_cv_cnn, X_cv_mlp], y_cv), 
                epochs=epochs,
                callbacks=[early_stopping]
        )

        loss, acc = model.evaluate([X_train_cnn, X_train_mlp], y_train, verbose=0)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("loss", loss)

        val_loss, val_acc = model.evaluate([X_cv_cnn, X_cv_mlp], y_cv, verbose=0)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_loss", val_loss)

        y_cv_logits = model.predict([X_cv_cnn, X_cv_mlp])
        y_cv_pred = np.argmax(y_cv_logits, axis=1)
        val_f1 = f1_score(y_cv, y_cv_pred, average='macro')
        val_f1_weighted = f1_score(y_cv, y_cv_pred, average='weighted')
        mlflow.log_metric("val_f1_macro", val_f1)
        mlflow.log_metric("val_f1_weighted", val_f1_weighted)

        notify_telegram(f"New modelCNNMLP trained with f1Score: {val_f1}, f1Weighted: {val_f1_weighted}")

        del model
        gc.collect()
        tf.keras.backend.clear_session()
        return val_f1

if __name__ == "__main__":
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=50, show_progress_bar=True)     
