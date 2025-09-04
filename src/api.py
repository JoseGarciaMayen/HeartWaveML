from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import time
from datetime import datetime
from typing import List, Dict
import psutil
import os

try:
    from src.predict import predict
    PREDICT_AVAILABLE = True
except ImportError as e:
    PREDICT_AVAILABLE = False
    print("Warning: predict function not available. Error: {e}")

app = FastAPI(
    title="HeartWaveML API",
    description="API for ECG signal classification using ML models",
    version="1.0.0",
    contact={
        "name": "Jose García Mayén",
        "email": "josegarciamayen@gmail.com",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ECGSample(BaseModel):
    signal: List[float] = Field(..., description="1D array representing the ECG signal (187 samples)")

class PredictionResponse(BaseModel):
    prediction: float
    timestamp: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    predict_available: bool
    system_info: Dict

class InfoResponse(BaseModel):
    api_name: str
    version: str
    description: str
    endpoints: List[str]
    model_info: Dict

start_time = time.time()
prediction_count = 0
error_count = 0


@app.get("/", response_model=InfoResponse, tags=["General"])
def api_info():
    """Provides general information about the API"""
    endpoints = [
        "/",
        "/predict",
        "/health",
        "/metrics",
        "/docs",
        "/redoc"
    ]
    
    model_info = {
        "model_type": "CONVXGBoost",
        "input_shape": "Variable (ECG signal)",
        "output": "Classification",
        "available": PREDICT_AVAILABLE
    }
    
    return InfoResponse(
        api_name="HeartWaveML API",
        version="1.0.0",
        description="API for ECG signal classification using ML models",
        endpoints=endpoints,
        model_info=model_info
    )

@app.post("/predict", tags=["Prediction"])
def classify_ecg_batch(samples: List[ECGSample]):
    """Classifies a batch of ECG signals"""
    if len(samples) > 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size too large. Maximum: 200 samples, received: {len(samples)}"
        )
    
    results = []
    for i, sample in enumerate(samples):
        try:
            result = classify_ecg(sample)
            results.append({
                "index": i,
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total_samples": len(samples),
        "successful_predictions": sum(1 for r in results if r["success"]),
        "failed_predictions": sum(1 for r in results if not r["success"]),
        "results": results
    }
def classify_ecg(sample: ECGSample):
    """Classifies a single ECG signal"""
    global prediction_count, error_count
    
    start_processing = time.time()
    
    try:
        if not PREDICT_AVAILABLE:
            error_count += 1
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available"
            )
        
        if len(sample.signal) != 187:
            error_count += 1
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Signal must have length 187 samples, received length: {len(sample.signal)}"
            )
        
        beat_signal = np.array(sample.signal)
        
        prediction = predict(beat_signal)
        
        if isinstance(prediction, np.generic):
            prediction = prediction.item()
        
        prediction_count += 1
        processing_time = (time.time() - start_processing) * 1000
        
        return PredictionResponse(
            prediction=float(prediction),
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        error_count += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Endpoint for health check"""
    uptime = time.time() - start_time
    
    system_info = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(uptime, 2),
        predict_available=PREDICT_AVAILABLE,
        system_info=system_info
    )

@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """Provides basic metrics about the API usage"""
    uptime = time.time() - start_time
    
    return {
        "uptime_seconds": round(uptime, 2),
        "total_predictions": prediction_count,
        "total_errors": error_count,
        "predictions_per_minute": round((prediction_count / uptime) * 60, 2) if uptime > 0 else 0,
        "error_rate": round((error_count / max(prediction_count, 1)) * 100, 2),
        "last_updated": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)