"""
main.py  —  FastAPI Anomaly Detection Service
----------------------------------------------
Provides real-time inference for telecom network KPI anomaly detection.

Endpoints:
  GET  /health          — liveness check
  GET  /model/info      — model metadata (threshold, metrics, features)
  POST /predict         — single snapshot prediction
  POST /predict/batch   — batch of snapshots (e.g. last hour of readings)

Usage:
  uvicorn api.main:app --reload --port 8000

Example request (single prediction):
  curl -X POST http://localhost:8000/predict \\
    -H "Content-Type: application/json" \\
    -d '{
      "cell_id": "CELL_001",
      "latency_ms": 145.5,
      "throughput_mbps": 12.3,
      "packet_loss_pct": 8.7,
      "sinr_db": 4.2,
      "connected_users": 87,
      "hour": 14,
      "day_of_week": 2,
      "is_weekend": 0,
      "features": {}
    }'
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Load model artifacts ───────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent.parent / "models"

try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "model.json")

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    with open(MODEL_DIR / "metadata.json") as f:
        metadata = json.load(f)

    FEATURE_COLS = metadata["feature_cols"]
    THRESHOLD    = metadata["threshold"]

except FileNotFoundError:
    raise RuntimeError(
        "Model artifacts not found. Train the model first:\n"
        "  python models/train.py"
    )

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Telecom Network Anomaly Detection API",
    description=(
        "Real-time ML inference for detecting anomalies in telecom network KPIs. "
        "Powered by XGBoost trained on latency, throughput, packet loss, SINR, "
        "and connected users time-series data."
    ),
    version="1.0.0",
)


# ── Schemas ────────────────────────────────────────────────────────────────

class KPISnapshot(BaseModel):
    """
    A single reading from a cell tower at one point in time.
    Include both the raw KPI values AND the pre-computed time-series
    features (lags, rolling stats, z-scores, rate-of-change).

    In production, the feature engineering pipeline (features.py) would
    compute these automatically from the last 4h of readings per cell.
    For direct API testing, pass features as a flat dict.
    """
    cell_id: str = Field(..., example="CELL_001")

    # Raw KPIs (for logging / display)
    latency_ms:       float = Field(..., ge=0,   example=145.5)
    throughput_mbps:  float = Field(..., ge=0,   example=12.3)
    packet_loss_pct:  float = Field(..., ge=0, le=100, example=8.7)
    sinr_db:          float = Field(...,         example=4.2)
    connected_users:  int   = Field(..., ge=0,   example=87)

    # Time features
    hour:        int = Field(..., ge=0, le=23, example=14)
    day_of_week: int = Field(..., ge=0, le=6,  example=2)
    is_weekend:  int = Field(..., ge=0, le=1,  example=0)

    # Pre-computed engineered features (flat dict, keys = feature_cols)
    features: dict[str, float] = Field(
        default={},
        description="Pre-computed feature dict from features.py. If empty, "
                    "raw KPIs are used for the known base features only."
    )


class PredictionResult(BaseModel):
    cell_id:        str
    anomaly:        bool
    anomaly_score:  float = Field(..., description="Probability of anomaly (0–1)")
    threshold:      float
    severity:       str   = Field(..., description="normal / warning / critical")
    raw_kpis: dict[str, Any]


class BatchRequest(BaseModel):
    snapshots: list[KPISnapshot]


class BatchResult(BaseModel):
    results:         list[PredictionResult]
    anomaly_count:   int
    total:           int


# ── Helpers ────────────────────────────────────────────────────────────────

def build_feature_vector(snapshot: KPISnapshot) -> np.ndarray:
    """
    Build the feature vector expected by the model.
    If pre-computed features are provided, use them directly.
    Otherwise fall back to raw KPIs for the base features.
    """
    if snapshot.features:
        try:
            vec = [snapshot.features[col] for col in FEATURE_COLS]
            return np.array(vec, dtype=float).reshape(1, -1)
        except KeyError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Missing feature: {e}. Provide all {len(FEATURE_COLS)} features."
            )

    # Fallback: use raw KPIs for the base columns only
    # (accuracy will be lower without time-series features)
    base_map = {
        "latency_ms":      snapshot.latency_ms,
        "throughput_mbps": snapshot.throughput_mbps,
        "packet_loss_pct": snapshot.packet_loss_pct,
        "sinr_db":         snapshot.sinr_db,
        "connected_users": float(snapshot.connected_users),
        "hour":            float(snapshot.hour),
        "day_of_week":     float(snapshot.day_of_week),
        "is_weekend":      float(snapshot.is_weekend),
    }
    vec = [base_map.get(col, 0.0) for col in FEATURE_COLS]
    return np.array(vec, dtype=float).reshape(1, -1)


def score_to_severity(score: float, threshold: float) -> str:
    if score < threshold:
        return "normal"
    elif score < threshold + 0.15:
        return "warning"
    else:
        return "critical"


def run_inference(snapshot: KPISnapshot) -> PredictionResult:
    X = build_feature_vector(snapshot)
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    is_anomaly = prob >= THRESHOLD

    return PredictionResult(
        cell_id=snapshot.cell_id,
        anomaly=is_anomaly,
        anomaly_score=round(prob, 4),
        threshold=THRESHOLD,
        severity=score_to_severity(prob, THRESHOLD),
        raw_kpis={
            "latency_ms":      snapshot.latency_ms,
            "throughput_mbps": snapshot.throughput_mbps,
            "packet_loss_pct": snapshot.packet_loss_pct,
            "sinr_db":         snapshot.sinr_db,
            "connected_users": snapshot.connected_users,
        },
    )


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost", "version": "1.0.0"}


@app.get("/model/info")
def model_info():
    return {
        "threshold":    THRESHOLD,
        "metrics":      metadata["metrics"],
        "feature_count": len(FEATURE_COLS),
        "train_rows":   metadata["train_rows"],
        "test_rows":    metadata["test_rows"],
    }


@app.post("/predict", response_model=PredictionResult)
def predict(snapshot: KPISnapshot):
    """
    Predict whether a single KPI snapshot represents a network anomaly.
    Returns anomaly probability, boolean flag, and severity level.
    """
    return run_inference(snapshot)


@app.post("/predict/batch", response_model=BatchResult)
def predict_batch(request: BatchRequest):
    """
    Predict anomalies for a batch of KPI snapshots (e.g. one cell's last hour).
    Useful for bulk ingestion from a monitoring pipeline.
    """
    if len(request.snapshots) > 500:
        raise HTTPException(status_code=400, detail="Max batch size is 500.")

    results = [run_inference(s) for s in request.snapshots]
    anomaly_count = sum(1 for r in results if r.anomaly)

    return BatchResult(
        results=results,
        anomaly_count=anomaly_count,
        total=len(results),
    )
