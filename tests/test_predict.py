from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "model_version" in response.json()

def test_predict_with_feature_id():
    response = client.post("/predict", json={
        "feature_id": 1
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "model_version" in data
