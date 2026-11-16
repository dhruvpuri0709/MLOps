import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# The trained model is copied into the Docker image at build time; MODEL_VERSION is set from the build
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")

# Same column names the model was trained with (sklearn's load_iris feature names)
FEATURE_NAMES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

_model = None

def get_model():
    """Loads the model once and reuses it for every request."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

@asynccontextmanager
async def lifespan(app):
    # Load at startup so the container fails immediately if the model file is missing
    get_model()
    yield

app = FastAPI(title="Iris Model API", version=MODEL_VERSION, lifespan=lifespan)

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])

@app.get("/")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
def predict(features: IrisFeatures):
    model = get_model()
    row = pd.DataFrame([[features.sepal_length, features.sepal_width,
                         features.petal_length, features.petal_width]], columns=FEATURE_NAMES)
    class_id = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probabilities)},
        "model_version": MODEL_VERSION,
    }
