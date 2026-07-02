import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import src.api as api

client = TestClient(api.app)


def _recording_payload(n=50):
    return {"beats": [{"signal": [0.0] * 187, "r_peak_sample": i * 300} for i in range(n)]}


@pytest.fixture
def mock_predict(monkeypatch):
    monkeypatch.setattr(api, "PREDICT_AVAILABLE", True)
    monkeypatch.setattr(api, "predict_record", lambda beats: [2] * len(beats), raising=False)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_info_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["api_name"] == "HeartWaveML API"


def test_predict_valid_recording(mock_predict):
    resp = client.post("/predict", json=_recording_payload(n=50))
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_beats"] == 50
    assert len(body["predictions"]) == 50
    assert body["predictions"][0]["prediction"] == 2


def test_predict_empty_beats_is_rejected(mock_predict):
    resp = client.post("/predict", json={"beats": []})
    assert resp.status_code == 422


def test_predict_wrong_signal_length_is_rejected(mock_predict):
    payload = _recording_payload()
    payload["beats"][0]["signal"] = [0.0] * 100
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_too_many_beats_is_rejected(mock_predict, monkeypatch):
    monkeypatch.setattr(api, "MAX_BEATS", 10)
    resp = client.post("/predict", json=_recording_payload(n=11))
    assert resp.status_code == 422
