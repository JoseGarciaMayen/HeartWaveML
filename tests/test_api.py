import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import src.api as api

client = TestClient(api.app)


@pytest.fixture
def mock_predict(monkeypatch):
    monkeypatch.setattr(api, "PREDICT_AVAILABLE", True)
    monkeypatch.setattr(api, "predict", lambda signal: 2)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_info_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["api_name"] == "HeartWaveML API"


def test_predict_valid_signal(mock_predict):
    signal = [0.0] * 187
    resp = client.post("/predict", json=[{"signal": signal}])
    assert resp.status_code == 200
    body = resp.json()
    assert body["successful_predictions"] == 1
    assert body["results"][0]["result"]["prediction"] == 2.0


def test_predict_wrong_length_is_rejected(mock_predict):
    resp = client.post("/predict", json=[{"signal": [0.0] * 100}])
    assert resp.status_code == 200
    body = resp.json()
    assert body["failed_predictions"] == 1
    assert "187" in body["results"][0]["error"]


def test_batch_too_large():
    samples = [{"signal": [0.0] * 187} for _ in range(51)]
    resp = client.post("/predict", json=samples)
    assert resp.status_code == 400
