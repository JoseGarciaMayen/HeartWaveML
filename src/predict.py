import numpy as np
from src.preprocessing import preprocess_xgb, preprocess_convxgb
import joblib

def predict(X_test, model="src/saved_models/modelCONVXGB.joblib"):
    if model =="src/saved_models/modelXGB.joblib":
        X_test = preprocess_xgb(X_test)
    elif model =="src/saved_models/modelCONVXGB.joblib":
        X_test = preprocess_convxgb(X_test)
    else:
        raise ValueError("Model not found")
    
    model = joblib.load(model)

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred)
    return y_pred
    

if __name__ == '__main__':
    X_test = np.random.uniform(-0.6, 0.6, 187)
    print(X_test)
    y_pred = predict(X_test, model="src/saved_models/modelCONVXGB.joblib")
    print(y_pred)
