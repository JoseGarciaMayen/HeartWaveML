from src.training.trainer import TrainerBase
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, concatenate # type: ignore
from tensorflow.keras import regularizers # type: ignore
import mlflow
import gc
import numpy as np
from sklearn.metrics import f1_score
from src.utils import notify_telegram

class TrainerCNNMLP(TrainerBase):
    def __init__(self,
                 model_name = "modelCNNMLP",
                 mlflow_experiment_name ="ECG_CNNMLP"):
        super().__init__(model_name, mlflow_experiment_name)

    def get_typed_params(self, best_params):
        param_types = {
            "l2": float,
            "dropout": float,
            "learning_rate": float,
            "filters": int,
            "filter_multiplier": int,
            "units_mlp1": int,
            "units_mlp2": int,
            "units_mlp3": int
        }
        
        typed_best_params = {}
        for k, v in best_params.items():
            type = param_types.get(k, str)
            typed_best_params[k] = type(v)
        return typed_best_params

    def create_model(self, input_shape_cnn=(187, 1), input_shape_mlp=(36,), num_classes=5):
        """
        Function to create the CNNMLP model with dynamic hyperparameters.
        """
        best_params = self.get_typed_params(self.get_params())

        mlflow.log_param('input_shape_cnn', input_shape_cnn)
        mlflow.log_param('input_shape_mlp', input_shape_mlp)
        mlflow.log_param('num_classes', num_classes)
        mlflow.log_param('l2', best_params['l2'])
        mlflow.log_param('dropout', best_params['dropout'])
        mlflow.log_param('learning_rate', best_params['learning_rate'])
        mlflow.log_param('filters', best_params['filters'])
        mlflow.log_param('filter_multiplier', best_params['filter_multiplier'])
        mlflow.log_param('units_mlp1', best_params['units_mlp1'])
        mlflow.log_param('units_mlp2', best_params['units_mlp2'])
        mlflow.log_param('units_mlp3', best_params['units_mlp3'])
        mlflow.log_param('conv_layers', 4)

        input_cnn = Input(input_shape_cnn)
        x = Conv1D(filters=best_params['filters'], kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(best_params['l2']), padding='same')(input_cnn)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(best_params['dropout'])(x)
        x = Conv1D(filters=best_params['filters']*best_params['filter_multiplier'], kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(best_params['l2']), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(best_params['dropout'])(x)
        x = Conv1D(filters=best_params['filters']*best_params['filter_multiplier']*best_params['filter_multiplier'], kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(best_params['l2']), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(best_params['dropout'])(x)
        x = Conv1D(filters=best_params['filters']*best_params['filter_multiplier']*best_params['filter_multiplier'], kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(best_params['l2']), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(best_params['dropout'])(x)

        x = Flatten()(x)
        input_mlp = Input(input_shape_mlp)
        y = Dense(best_params['units_mlp1'], activation='relu', kernel_regularizer=regularizers.l2(best_params['l2']))(input_mlp)
        y = Dense(best_params['units_mlp2'], activation='relu', kernel_regularizer=regularizers.l2(best_params['l2']))(y)
        y = Dense(best_params['units_mlp3'], activation='relu', kernel_regularizer=regularizers.l2(best_params['l2']))(y)
        y = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(best_params['l2']))(y)
        y = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(best_params['l2']))(y)

        combined = concatenate([x, y])

        z = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(best_params['l2']))(combined)
        z = Dropout(best_params['dropout'])(z)
        z = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(best_params['l2']))(z)
        z = Dropout(best_params['dropout'])(z)
        
        output = Dense(num_classes, activation='linear')(z)

        model = Model(inputs=[input_cnn, input_mlp], outputs=output)

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(best_params['learning_rate']),
            metrics=['accuracy']
        )

        return model
    
    def save_model(self, model, model_name):
        model.save("src/saved_models/" + model_name + ".keras")

    def mlflow_start(self, model, X_train, y_train, X_cv, y_cv, class_weights):

        mlflow.tensorflow.autolog(
            log_models=False,
            log_datasets=False,
            disable=False,
            exclusive=False,
            disable_for_unsupported_versions=False,
            silent=True
        )
        
        with mlflow.start_run(nested=True):
            model.fit([X_train.iloc[:, :187].values.reshape(-1,187,1), X_train.iloc[:, 187:].values], y_train, class_weight=class_weights, validation_data=([X_cv.iloc[:, :187].values.reshape(-1,187,1), X_cv.iloc[:, 187:].values], y_cv), epochs=50)

            loss, acc = model.evaluate([X_train.iloc[:, :187].values.reshape(-1,187,1), X_train.iloc[:, 187:].values], y_train)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("loss", loss)

            val_loss, val_acc = model.evaluate([X_cv.iloc[:, :187].values.reshape(-1,187,1), X_cv.iloc[:, 187:].values], y_cv)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_loss", val_loss)

            y_cv_logits = model.predict([X_cv.iloc[:, :187].values.reshape(-1,187,1), X_cv.iloc[:, 187:].values])
            y_cv_pred = np.argmax(y_cv_logits, axis=1)
            val_f1 = f1_score(y_cv, y_cv_pred, average='macro')
            mlflow.log_metric("val_f1_macro", val_f1)

            mlflow.keras.log_model(model, self.model_name, registered_model_name=self.model_name)
            self.save_model(model, self.model_name)

            del model
            gc.collect()
            return val_f1

    def train(self):
        """
        Function to train the CNN+MLP model with the best hyperparameters found by Optuna.
        It logs the model and metrics to MLflow.
        """
        X_train , y_train, X_cv, y_cv, class_weights = self.load_data('data/processed/feat/mitbih_train_features.csv', 'data/processed/feat/mitbih_cv_features.csv')

        model = self.create_model()

        val_f1 = self.mlflow_start(model, X_train, y_train, X_cv, y_cv, class_weights)

        return val_f1
        

if __name__ == "__main__":
    trainer = TrainerCNNMLP()
    val_f1 = trainer.train()
    notify_telegram(f"Nuevo modelo entrenado con f1Score de: {val_f1}")