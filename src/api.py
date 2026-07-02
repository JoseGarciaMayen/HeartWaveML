import sys
import time
from datetime import datetime

import psutil
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.config import SEQUENCES

WINDOW = SEQUENCES.get("window", 45)
BEAT_LENGTH = 187

MAX_BEATS = 5000

try:
    from src.predict import predict_record

    PREDICT_AVAILABLE = True
except ImportError as e:
    PREDICT_AVAILABLE = False
    print(f"Warning: predict function not available. Error: {e}")

app = FastAPI(
    title="HeartWaveML API",
    description="API for ECG signal classification using ML models",
    version="1.0.0",
    contact={
        "name": "Jose García Mayén",
        "email": "josegarciamayen@gmail.com",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Beat(BaseModel):
    signal: list[float] = Field(
        ..., description=f"1D array representing a single ECG beat ({BEAT_LENGTH} samples)"
    )
    r_peak_sample: int = Field(
        ..., description="Absolute sample index of this beat's R-peak in the recording"
    )

    @field_validator("signal")
    @classmethod
    def _check_signal_length(cls, signal: list[float]) -> list[float]:
        if len(signal) != BEAT_LENGTH:
            raise ValueError(f"Expected exactly {BEAT_LENGTH} samples, got {len(signal)}")
        return signal


class ECGRecording(BaseModel):
    beats: list[Beat] = Field(
        ...,
        description=(
            "All ordered, already-segmented beats of one ECG recording. "
            "The API returns the class of every beat."
        ),
    )

    @field_validator("beats")
    @classmethod
    def _check_length(cls, beats: list[Beat]) -> list[Beat]:
        if len(beats) == 0:
            raise ValueError("beats must not be empty")
        if len(beats) > MAX_BEATS:
            raise ValueError(f"Too many beats. Maximum: {MAX_BEATS}, received: {len(beats)}")
        return beats


class BeatPrediction(BaseModel):
    index: int
    r_peak_sample: int
    prediction: int


class PredictionResponse(BaseModel):
    predictions: list[BeatPrediction]
    n_beats: int
    timestamp: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    predict_available: bool
    system_info: dict


class InfoResponse(BaseModel):
    api_name: str
    version: str
    description: str
    endpoints: list[str]
    model_info: dict


start_time = time.time()
prediction_count = 0
error_count = 0


@app.get("/", response_model=InfoResponse, tags=["General"])
def api_info():
    """Provides general information about the API"""
    endpoints = ["/", "/predict", "/health", "/metrics", "/docs", "/redoc"]

    model_info = {
        "model_type": "Transformer",
        "input_shape": f"(n_beats, {WINDOW}, 46) sliding windows over a full ECG recording",
        "output": "Classification of every beat in the recording",
        "available": PREDICT_AVAILABLE,
    }

    return InfoResponse(
        api_name="HeartWaveML API",
        version="1.0.0",
        description="API for ECG signal classification using ML models",
        endpoints=endpoints,
        model_info=model_info,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def classify_ecg(recording: ECGRecording):
    """Classifies every beat of one ECG recording"""
    global prediction_count, error_count

    start_processing = time.time()

    try:
        if not PREDICT_AVAILABLE:
            error_count += 1
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available",
            )

        beats = [
            {"signal": beat.signal, "r_peak_sample": beat.r_peak_sample} for beat in recording.beats
        ]

        predictions = predict_record(beats)

        prediction_count += 1
        processing_time = (time.time() - start_processing) * 1000

        return PredictionResponse(
            predictions=[
                BeatPrediction(index=i, r_peak_sample=beat.r_peak_sample, prediction=pred)
                for i, (beat, pred) in enumerate(zip(recording.beats, predictions, strict=True))
            ],
            n_beats=len(predictions),
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        error_count += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}"
        ) from e


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Endpoint for health check"""
    uptime = time.time() - start_time

    system_info = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(uptime, 2),
        predict_available=PREDICT_AVAILABLE,
        system_info=system_info,
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
        "last_updated": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
