"""
features.py
-----------
Time-series feature engineering for telecom KPI anomaly detection.

For each raw KPI we generate:
  - Lag features      : value at t-1, t-2, t-4, t-8 (15min, 30min, 1h, 2h ago)
  - Rolling mean      : 4-slot (1h) and 16-slot (4h) windows
  - Rolling std       : same windows — captures variance bursts
  - Rolling min/max   : 4-slot window — captures extreme values
  - Z-score           : deviation from 4h rolling mean in std units
  - Rate of change    : (t - t-1) / t-1 — detects sudden jumps

All windows are computed PER CELL to avoid leaking signal across cells.
"""

import pandas as pd
import numpy as np
from pathlib import Path


KPI_COLS = [
    "latency_ms",
    "throughput_mbps",
    "packet_loss_pct",
    "sinr_db",
    "connected_users",
]

LAG_STEPS    = [1, 2, 4, 8]        # in 15-min slots
SHORT_WINDOW = 4                    # 1 hour
LONG_WINDOW  = 16                   # 4 hours


def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Input : raw KPI DataFrame (columns: timestamp, cell_id, kpis..., anomaly)
    Output: feature-engineered DataFrame, NaN rows dropped
    """
    df = df.copy()
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)

    feature_frames = []

    for cell_id, group in df.groupby("cell_id"):
        group = group.copy().reset_index(drop=True)

        for kpi in KPI_COLS:
            series = group[kpi]

            # ── Lag features ──────────────────────────────────────────────
            for lag in LAG_STEPS:
                group[f"{kpi}_lag{lag}"] = series.shift(lag)

            # ── Rolling statistics ────────────────────────────────────────
            for w, label in [(SHORT_WINDOW, "1h"), (LONG_WINDOW, "4h")]:
                roll = series.shift(1).rolling(w, min_periods=w // 2)
                group[f"{kpi}_roll_mean_{label}"] = roll.mean()
                group[f"{kpi}_roll_std_{label}"]  = roll.std()

            roll_short = series.shift(1).rolling(SHORT_WINDOW, min_periods=2)
            group[f"{kpi}_roll_min_1h"] = roll_short.min()
            group[f"{kpi}_roll_max_1h"] = roll_short.max()

            # ── Z-score (deviation from 4h rolling mean) ──────────────────
            mean_4h = group[f"{kpi}_roll_mean_4h"]
            std_4h  = group[f"{kpi}_roll_std_4h"].replace(0, np.nan)
            group[f"{kpi}_zscore"] = (series - mean_4h) / std_4h

            # ── Rate of change ─────────────────────────────────────────────
            prev = series.shift(1).replace(0, np.nan)
            group[f"{kpi}_roc"] = (series - series.shift(1)) / prev

        # ── Time features ─────────────────────────────────────────────────
        group["hour"]       = pd.to_datetime(group["timestamp"]).dt.hour
        group["day_of_week"] = pd.to_datetime(group["timestamp"]).dt.dayofweek
        group["is_weekend"] = (group["day_of_week"] >= 5).astype(int)

        feature_frames.append(group)

    result = pd.concat(feature_frames, ignore_index=True)

    # Drop rows where rolling features couldn't be computed
    n_before = len(result)
    result = result.dropna().reset_index(drop=True)
    n_dropped = n_before - len(result)
    if verbose:
        print(f"Feature engineering complete. Dropped {n_dropped} NaN rows, "
              f"{len(result):,} rows remaining.")

    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature column names (excludes metadata and target)."""
    exclude = {"timestamp", "cell_id", "anomaly"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    raw_path = Path(__file__).parent / "raw_kpis.csv"
    df_raw = pd.read_csv(raw_path, parse_dates=["timestamp"])
    df_feat = engineer_features(df_raw)
    out = Path(__file__).parent / "features.csv"
    df_feat.to_csv(out, index=False)
    print(f"Saved → {out}")
    print(f"Feature columns: {len(get_feature_columns(df_feat))}")
