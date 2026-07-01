import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import src.api as api

client = TestClient(api.app)

WINDOW = api.WINDOW


def _window_payload(n=WINDOW):
    return {"beats": [{"signal": [0.0] * 187, "r_peak_sample": i * 300} for i in range(n)]}


@pytest.fixture
def mock_predict(monkeypatch):
    monkeypatch.setattr(api, "PREDICT_AVAILABLE", True)
    monkeypatch.setattr(api, "predict", lambda beats: 2, raising=False)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_info_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["api_name"] == "HeartWaveML API"


def test_predict_valid_window(mock_predict):
    resp = client.post("/predict", json=[_window_payload()])
    assert resp.status_code == 200
    body = resp.json()
    assert body["successful_predictions"] == 1
    assert body["results"][0]["result"]["prediction"] == 2.0


def test_predict_wrong_window_length_is_rejected(mock_predict):
    resp = client.post("/predict", json=[_window_payload(n=10)])
    assert resp.status_code == 422


def test_batch_too_large():
    windows = [_window_payload() for _ in range(201)]
    resp = client.post("/predict", json=windows)
    assert resp.status_code == 400
