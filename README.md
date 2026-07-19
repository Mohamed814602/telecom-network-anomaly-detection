# Telecom Network Anomaly Detection System

An end-to-end ML system that detects anomalies in telecom network KPIs (latency, throughput, packet loss) in real time, deployed as a production-ready REST API.

**Key result: 95.2% recall on detecting network faults before service degradation.**

---

## Problem

Telecom networks generate thousands of KPI readings per minute across hundreds of cell towers. Identifying fault conditions (congestion, hardware degradation, interference) manually is slow and reactive. This system flags anomalies automatically, enabling NOC teams to act before users are impacted.

## Solution

A supervised XGBoost classifier trained on time-series KPI features, served via FastAPI for real-time inference.

### KPIs monitored
| KPI | Unit | What it signals |
|---|---|---|
| `latency_ms` | ms | Congestion, backhaul issues |
| `throughput_mbps` | Mbps | Hardware faults, interference |
| `packet_loss_pct` | % | Link instability, interference |
| `sinr_db` | dB | Signal quality, interference |
| `connected_users` | count | Load-related anomalies |

---

## Project Structure

```
telecom_anomaly/
├── data/
│   ├── generate_data.py   # Synthetic KPI dataset generator
│   ├── features.py        # Time-series feature engineering
│   └── raw_kpis.csv       # Generated after running generate_data.py
├── models/
│   ├── train.py           # XGBoost training + threshold tuning
│   ├── model.json         # Saved model (generated after training)
│   ├── scaler.pkl         # Feature scaler (generated after training)
│   └── metadata.json      # Threshold + metrics (generated after training)
├── api/
│   └── main.py            # FastAPI inference service
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data
```bash
python data/generate_data.py
```

### 3. Engineer features
```bash
python data/features.py
```

### 4. Train the model
```bash
python models/train.py
```

### 5. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Make a prediction

A brand-new cell has no history yet, so the **first call for any `cell_id` intentionally returns 425 Too Early** — the API won't guess with a mostly-empty feature vector:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cell_id": "CELL_001",
    "latency_ms": 145.5,
    "throughput_mbps": 12.3,
    "packet_loss_pct": 8.7,
    "sinr_db": 4.2,
    "connected_users": 87
  }'
```

Send ~20 more readings for the same `cell_id`, in chronological order, and the API starts returning real predictions once it has enough history to compute the time-series features (see [How `/predict` builds its feature vector](#api-endpoints) further down). If you already have pre-computed features on hand, pass them directly via the `features` field instead and skip the warm-up.

### 7. View interactive API docs
Open: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Running with Docker

Build and run the API in a container instead of a local Python environment:

```bash
docker compose up --build
```

The API is then available at [http://localhost:8000](http://localhost:8000) — same endpoints as above.

Or with plain Docker, no compose:

```bash
docker build -t telecom-anomaly-detection .
docker run -p 8000:8000 telecom-anomaly-detection
```

Notes:
- Runs as a non-root user, with a container `HEALTHCHECK` against `/health`.
- The trained model artifacts (`models/model.json`, `scaler.pkl`, `metadata.json`) are baked into the image at build time — rebuild the image after retraining.

---

## Feature Engineering

For each KPI, the following features are computed per cell tower:

| Feature type | Detail |
|---|---|
| **Lag features** | Values at t-1, t-2, t-4, t-8 (15 min to 2 hours ago) |
| **Rolling mean** | 1h and 4h windows |
| **Rolling std** | 1h and 4h windows — captures variance bursts |
| **Rolling min/max** | 1h window |
| **Z-score** | Deviation from 4h rolling mean |
| **Rate of change** | `(t - t-1) / t-1` — detects sudden jumps |
| **Time features** | Hour of day, day of week, is_weekend |

Total: **68 features** across 5 KPIs.

| Feature group | Count |
|---|---|
| Lag features (t-1, t-2, t-4, t-8) | 5 KPIs × 4 = 20 |
| Rolling mean (1h, 4h) | 5 KPIs × 2 = 10 |
| Rolling std (1h, 4h) | 5 KPIs × 2 = 10 |
| Rolling min/max (1h) | 5 KPIs × 2 = 10 |
| Z-score (4h window) | 5 KPIs × 1 = 5 |
| Rate of change | 5 KPIs × 1 = 5 |
| Raw KPI values | 5 |
| Time features (hour, day_of_week, is_weekend) | 3 |
| **Total** | **68** |

---

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | XGBoost (gradient boosted trees) |
| Trees | 300 (with early stopping) |
| Max depth | 6 |
| Learning rate | 0.05 |
| Class imbalance | Handled via `scale_pos_weight` |
| Threshold | Tuned on validation set to maximize F1 |
| Train/test split | Time-based 80/20 (no shuffle) |

### Why XGBoost?
- Handles tabular time-series features well out of the box
- Robust to feature scale differences and missing values
- Fast inference (<1ms per prediction) — suitable for real-time monitoring
- Feature importance scores aid explainability for NOC teams

### Why tune the threshold?
The default 0.5 threshold maximizes accuracy, not recall. In anomaly detection, a missed fault (false negative) is far more costly than a false alarm. Threshold tuning lets us control this trade-off explicitly.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/model/info` | Model metrics and threshold |
| POST | `/predict` | Single snapshot prediction |
| POST | `/predict/batch` | Batch of up to 500 snapshots |

---

## Results

| Metric | Value |
|---|---|
| Recall | **95.2%** — catches 95 of every 100 faults |
| Precision | **76.9%** — ~23% false alarm rate, acceptable for NOC operations |
| F1 Score | **0.851** |

*Results on held-out test set (last 20% of time window, never seen during training).*
*Threshold tuned to maximize recall subject to a 93% recall floor — missing a fault is more costly than a false alarm in NOC operations.*

---

## Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `Pandas` · `NumPy` · `FastAPI` · `Pydantic` · `Uvicorn` · `Docker`
