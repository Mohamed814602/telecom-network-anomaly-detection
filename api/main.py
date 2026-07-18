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
from typing import Any, Optional

import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.live_features import CellHistoryStore, make_raw_row, now_iso, MIN_HISTORY_ROWS

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

# Per-cell rolling history used to compute real-time features when the
# caller sends raw KPIs instead of pre-computed features. See
# api/live_features.py for why this replaced the old zero-fill fallback.
history_store = CellHistoryStore()

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

    # Ordering for rolling/lag features. If omitted, server receive-time is
    # used — fine for a live single-cell stream, but pass real timestamps
    # if you're backfilling historical readings out of real-time order.
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp of this reading. Defaults to now().",
        example="2026-07-18T14:00:00Z",
    )

    # Raw KPIs (for logging / display)
    latency_ms:       float = Field(..., ge=0,   example=145.5)
    throughput_mbps:  float = Field(..., ge=0,   example=12.3)
    packet_loss_pct:  float = Field(..., ge=0, le=100, example=8.7)
    sinr_db:          float = Field(...,         example=4.2)
    connected_users:  int   = Field(..., ge=0,   example=87)

    # Time features
    # These are now derived from `timestamp` during feature computation.
    # Kept optional here for display/logging convenience only — they are
    # NOT used to build the model's feature vector anymore.
    hour:        Optional[int] = Field(default=None, ge=0, le=23, example=14)
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6,  example=2)
    is_weekend:  Optional[int] = Field(default=None, ge=0, le=1,  example=0)

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
    feature_source: str   = Field(
        ..., description="'precomputed' if caller supplied features, "
                          "'live' if computed from this cell's buffered history"
    )
    raw_kpis: dict[str, Any]


class BatchRequest(BaseModel):
    snapshots: list[KPISnapshot]


class BatchItemResult(BaseModel):
    cell_id: str
    ok:      bool
    result:  Optional[PredictionResult] = None
    error:   Optional[str] = None


class BatchResult(BaseModel):
    results:         list[BatchItemResult]
    anomaly_count:   int
    scored_count:    int
    error_count:     int
    total:           int


# ── Helpers ────────────────────────────────────────────────────────────────

def build_feature_vector(snapshot: KPISnapshot) -> tuple[np.ndarray, str]:
    """
    Build the feature vector expected by the model.

    Two supported paths:
      1. Caller supplies pre-computed `features` (e.g. a backfill/offline
         job that already ran data/features.py) — used directly.
      2. Caller supplies raw KPIs only — the server maintains a rolling
         per-cell history and computes the SAME 68 features training used,
         via data/features.engineer_features(). This requires enough prior
         readings for that cell; if there aren't enough yet, we raise a
         425 rather than silently scoring on a mostly-zeroed vector.

    Returns (feature_vector, source) where source is "precomputed" or
    "live" — surfaced on the response so callers can tell which path
    scored their request.
    """
    if snapshot.features:
        try:
            vec = [snapshot.features[col] for col in FEATURE_COLS]
            return np.array(vec, dtype=float).reshape(1, -1), "precomputed"
        except KeyError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Missing feature: {e}. Provide all {len(FEATURE_COLS)} features."
            )

    raw_row = make_raw_row(
        timestamp=snapshot.timestamp or now_iso(),
        cell_id=snapshot.cell_id,
        kpis={
            "latency_ms":      snapshot.latency_ms,
            "throughput_mbps": snapshot.throughput_mbps,
            "packet_loss_pct": snapshot.packet_loss_pct,
            "sinr_db":         snapshot.sinr_db,
            "connected_users": snapshot.connected_users,
        },
    )
    computed = history_store.add_and_compute(snapshot.cell_id, raw_row)

    if computed is None:
        rows_so_far = history_store.rows_available(snapshot.cell_id)
        raise HTTPException(
            status_code=425,  # Too Early
            detail=(
                f"Not enough history for cell '{snapshot.cell_id}' yet to compute "
                f"time-series features ({rows_so_far}/{MIN_HISTORY_ROWS} readings "
                "buffered). Keep sending readings for this cell, in chronological "
                "order, or pass pre-computed `features` directly."
            ),
        )

    try:
        vec = [computed[col] for col in FEATURE_COLS]
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Live feature computation is missing expected column: {e}"
        )
    return np.array(vec, dtype=float).reshape(1, -1), "live"


def score_to_severity(score: float, threshold: float) -> str:
    if score < threshold:
        return "normal"
    elif score < threshold + 0.15:
        return "warning"
    else:
        return "critical"


def run_inference(snapshot: KPISnapshot) -> PredictionResult:
    X, feature_source = build_feature_vector(snapshot)
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    is_anomaly = prob >= THRESHOLD

    return PredictionResult(
        cell_id=snapshot.cell_id,
        anomaly=is_anomaly,
        anomaly_score=round(prob, 4),
        threshold=THRESHOLD,
        severity=score_to_severity(prob, THRESHOLD),
        feature_source=feature_source,
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
    Useful for bulk ingestion from a monitoring pipeline, and for backfilling
    a cell's history in one call (snapshots are processed in list order, so
    send them chronologically per cell).

    Each item is scored independently — if an early snapshot in the batch
    is a cold start (not enough history yet), that item reports its own
    error rather than failing the entire batch.
    """
    if len(request.snapshots) > 500:
        raise HTTPException(status_code=400, detail="Max batch size is 500.")

    items: list[BatchItemResult] = []
    for snapshot in request.snapshots:
        try:
            result = run_inference(snapshot)
            items.append(BatchItemResult(cell_id=snapshot.cell_id, ok=True, result=result))
        except HTTPException as e:
            items.append(BatchItemResult(cell_id=snapshot.cell_id, ok=False, error=e.detail))

    anomaly_count = sum(1 for it in items if it.ok and it.result.anomaly)
    error_count   = sum(1 for it in items if not it.ok)

    return BatchResult(
        results=items,
        anomaly_count=anomaly_count,
        scored_count=len(items) - error_count,
        error_count=error_count,
        total=len(items),
    )
