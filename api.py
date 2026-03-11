"""
SAT Dengue — API REST para predicción de exceso epidémico de dengue.

Ejecutar con:
    uvicorn api:app --reload
"""

from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Modelo global
# ---------------------------------------------------------------------------
_data: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    raw = joblib.load("logistic_regression.joblib")
    _data["model"] = raw["model"]
    _data["scaler"] = raw["scaler"]
    _data["features"] = raw["features"]
    yield
    _data.clear()


app = FastAPI(
    title="SAT Dengue API",
    description="API de predicción de exceso epidémico de dengue a nivel municipal.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    features: dict[str, float]


class PredictResponse(BaseModel):
    probabilidad_exceso: float
    nivel_alerta: str


class HealthResponse(BaseModel):
    model_name: str
    feature_count: int


class FeaturesResponse(BaseModel):
    features: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health():
    model = _data["model"]
    return HealthResponse(
        model_name=type(model).__name__,
        feature_count=len(_data["features"]),
    )


@app.get("/features", response_model=FeaturesResponse)
def features():
    return FeaturesResponse(features=_data["features"])


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    feature_names = _data["features"]
    values = [req.features[f] for f in feature_names]

    X = np.array(values).reshape(1, -1)
    X_scaled = _data["scaler"].transform(X)
    prob = float(_data["model"].predict_proba(X_scaled)[0, 1])

    if prob >= 0.6:
        nivel = "Alerta"
    elif prob >= 0.3:
        nivel = "Riesgo"
    else:
        nivel = "Normal"

    return PredictResponse(probabilidad_exceso=round(prob, 6), nivel_alerta=nivel)
