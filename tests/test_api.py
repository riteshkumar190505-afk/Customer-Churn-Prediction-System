import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.main import app, evaluate_risk_profile

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "calibrated_threshold" in data


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "numeric" in data["features"]
    assert "categorical" in data["features"]


def test_risk_profile_logic():
    level_high, _ = evaluate_risk_profile(0.85, 0.40)
    assert level_high == "High"

    level_med, _ = evaluate_risk_profile(0.55, 0.40)
    assert level_med == "Medium"

    level_low, _ = evaluate_risk_profile(0.25, 0.40)
    assert level_low == "Low"


def test_prediction_schema_validation():
    # Incomplete customer payload should trigger 422 Unprocessable Entity
    invalid_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No"
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
