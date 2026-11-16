import pytest
from fastapi.testclient import TestClient

import src.app as app_module
from src.train_and_save_model import download_data, preprocess_data, train_model


# Train a real model once and inject it, so the tests don't need a model.joblib file
@pytest.fixture(scope="module")
def client():
    X, y = download_data()
    X_train, _, y_train, _ = preprocess_data(X, y)
    app_module._model = train_model(X_train, y_train)
    yield TestClient(app_module.app)
    app_module._model = None

# ----------------- Test Health ----------------- #
def test_health(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_version" in response.json()

# ----------------- Test Predict ----------------- #
def test_predict_setosa(client):
    # A typical setosa flower: short, narrow petals
    response = client.post("/predict", json={
        "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    assert response.status_code == 200
    body = response.json()
    assert body["class_id"] == 0
    assert body["class_name"] == "setosa"
    assert set(body["probabilities"]) == {"setosa", "versicolor", "virginica"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)

def test_predict_virginica(client):
    # A typical virginica flower: long, wide petals
    response = client.post("/predict", json={
        "sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2})
    assert response.status_code == 200
    assert response.json()["class_name"] == "virginica"

# ----------------- Test Input validation ----------------- #
def test_predict_rejects_bad_input(client):
    # Missing field
    assert client.post("/predict", json={"sepal_length": 5.1}).status_code == 422
    # Negative measurement (fields must be > 0)
    assert client.post("/predict", json={
        "sepal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}).status_code == 422
