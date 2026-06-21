import os

from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = f"http://{os.getenv('IP', '127.0.0.1')}:5000"

DATA = {
    "feat_train": "data/processed/feat/mitbih_train_features.csv",
    "feat_cv": "data/processed/feat/mitbih_cv_features.csv",
    "feat_test": "data/processed/feat/mitbih_test_features.csv",
    "cnn_train": "data/processed/cnn/mitbih_train_cnn.csv",
    "cnn_cv": "data/processed/cnn/mitbih_cv_cnn.csv",
    "cnn_test": "data/processed/cnn/mitbih_test_cnn.csv",
    "base_train": "data/processed/base/mitbih_train.csv",
    "base_cv": "data/processed/base/mitbih_cv.csv",
    "base_test": "data/processed/base/mitbih_test.csv",
}

CNN_ARCH = {
    "l2": 0,
    "dropout": 0,
    "learning_rate": 0.01,
    "filters1": 16,
    "filters2": 32,
    "filters3": 64,
}
