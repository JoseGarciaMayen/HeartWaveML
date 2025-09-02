import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import tensorflow as tf
import json
import matplotlib.pyplot as plt

def load_test_data(path=None):
    X_test = pd.read_csv(path).drop('class', axis=1)
    y_test = pd.read_csv(path)['class']
    return X_test, y_test

def evaluate_XGB(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f2 = fbeta_score(y_test, y_pred, beta=2, average='macro')
    f2_weighted = fbeta_score(y_test, y_pred, beta=2, average='weighted')
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    return accuracy, precision, recall, f1, f1_weighted, f2, f2_weighted, cm

def evaluate_CNNMLP(model, X_test, y_test):
    X_test = X_test.iloc[:, :187].values.reshape(-1,187,1), X_test.iloc[:, 187:].values
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f2 = fbeta_score(y_test, y_pred, beta=2, average='macro')
    f2_weighted = fbeta_score(y_test, y_pred, beta=2, average='weighted')
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    return accuracy, precision, recall, f1, f1_weighted, f2, f2_weighted, cm

def generate_comparison_report(XGB_metrics, CNNMLP_metrics, CONVXGB_metrics):
    print("Comparación de métricas:")
    print("Modelo XGB:")
    print(f"Accuracy: {XGB_metrics[0]:.4f}")
    print(f"Precision: {XGB_metrics[1]:.4f}")
    print(f"Recall: {XGB_metrics[2]:.4f}")
    print(f"F1-score: {XGB_metrics[3]:.4f}")
    print(f"F1-score weighted: {XGB_metrics[4]:.4f}")
    print(f"F2-score: {XGB_metrics[5]:.4f}")
    print(f"F2-score weighted: {XGB_metrics[6]:.4f}")
    disp = ConfusionMatrixDisplay(confusion_matrix=XGB_metrics[7], display_labels=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix XGB")
    plt.savefig("src/saved_models/metrics/confusion_matrix_xgb.png")
    print("\nModelo CNNMLP:")
    print(f"Accuracy: {CNNMLP_metrics[0]:.4f}")
    print(f"Precision: {CNNMLP_metrics[1]:.4f}")
    print(f"Recall: {CNNMLP_metrics[2]:.4f}")
    print(f"F1-score: {CNNMLP_metrics[3]:.4f}")
    print(f"F1-score weighted: {CNNMLP_metrics[4]:.4f}")
    print(f"F2-score: {CNNMLP_metrics[5]:.4f}")
    print(f"F2-score weighted: {CNNMLP_metrics[6]:.4f}")
    disp = ConfusionMatrixDisplay(confusion_matrix=CNNMLP_metrics[7], display_labels=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix CNNMLP")
    plt.savefig("src/saved_models/metrics/confusion_matrix_cnnmlp.png")
    print("\nModelo CONVXGB:")
    print(f"Accuracy: {CONVXGB_metrics[0]:.4f}")
    print(f"Precision: {CONVXGB_metrics[1]:.4f}")
    print(f"Recall: {CONVXGB_metrics[2]:.4f}")
    print(f"F1-score: {CONVXGB_metrics[3]:.4f}")
    print(f"F1-score weighted: {CONVXGB_metrics[4]:.4f}")
    print(f"F2-score: {CONVXGB_metrics[5]:.4f}")
    print(f"F2-score weighted: {CONVXGB_metrics[6]:.4f}")
    disp = ConfusionMatrixDisplay(confusion_matrix=CONVXGB_metrics[7], display_labels=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix CONVXGB")
    plt.savefig("src/saved_models/metrics/confusion_matrix_convxgb.png")

    metrics = {
        "XGB": "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, F1-score weighted: {:.4f}, F2-score: {:.4f}, F2-score weighted: {:.4f}".format(*XGB_metrics),
        "CNNMLP": "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, F1-score weighted: {:.4f}, F2-score: {:.4f}, F2-score weighted: {:.4f}".format(*CNNMLP_metrics),
        "CONVXGB": "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-score: {:.4f}, F1-score weighted: {:.4f}, F2-score: {:.4f}, F2-score weighted: {:.4f}".format(*CONVXGB_metrics)
    }

    with open('src/saved_models/metrics/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

def evaluate_models():
    X_test, y_test = load_test_data('data/processed/feat/mitbih_test_features.csv')
    
    modelXGB = joblib.load("src/saved_models/modelXGB.joblib")
    XGB_metrics = evaluate_XGB(modelXGB, X_test, y_test)
    
    modelCNNMLP = tf.keras.models.load_model("src/saved_models/modelCNNMLP.keras")
    CNNMLP_metrics = evaluate_CNNMLP(modelCNNMLP, X_test, y_test)

    X_test, y_test = load_test_data('data/processed/cnn/mitbih_test_cnn.csv')

    modelCONVXGB = joblib.load("src/saved_models/modelCONVXGB.joblib")
    CONVXGB_metrics = evaluate_XGB(modelCONVXGB, X_test, y_test)
    
    generate_comparison_report(XGB_metrics, CNNMLP_metrics, CONVXGB_metrics)

if __name__ == "__main__":
    evaluate_models()