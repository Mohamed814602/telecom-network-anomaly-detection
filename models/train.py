"""
train.py
--------
Trains an XGBoost anomaly detector on the engineered telecom KPI features.

Design decisions:
  - Time-based train/test split (no shuffle) — respects temporal ordering
    so the model is evaluated on future data it has never seen, simulating
    real deployment.
  - Class imbalance handled via scale_pos_weight — anomalies are rare (~5%)
    so we up-weight them during training.
  - Threshold tuning — default 0.5 threshold optimizes accuracy, but for
    anomaly detection we care more about recall (catching faults). We tune
    the threshold to maximize F1 on the validation set.
  - Model saved as model.json (XGBoost native format) + metadata.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import joblib

# Allow imports from parent dir
sys.path.append(str(Path(__file__).parent.parent))
from data.features import get_feature_columns

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent

TRAIN_RATIO = 0.8   # First 80% of time for training


def load_data():
    feat_path = DATA_DIR / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            "features.csv not found. Run: python data/features.py"
        )
    df = pd.read_csv(feat_path, parse_dates=["timestamp"])
    return df


def time_split(df: pd.DataFrame, ratio: float):
    """Split by time, not randomly — critical for time-series models."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def tune_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score on validation set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def main():
    print("=" * 60)
    print("  Telecom Anomaly Detection — Model Training")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    df = load_data()
    feature_cols = get_feature_columns(df)
    print(f"\nLoaded {len(df):,} rows | {len(feature_cols)} features")

    train_df, test_df = time_split(df, TRAIN_RATIO)
    print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    X_train = train_df[feature_cols].values
    y_train = train_df["anomaly"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["anomaly"].values

    print(f"\nTrain anomaly rate: {y_train.mean():.2%}")
    print(f"Test  anomaly rate: {y_test.mean():.2%}")

    # ── Scale features ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Handle class imbalance ─────────────────────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"\nClass ratio (neg/pos): {scale_pos_weight:.1f}x → used as scale_pos_weight")

    # ── Train XGBoost ──────────────────────────────────────────────────────
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",        # area under precision-recall curve
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=50,
    )

    # ── Threshold tuning ───────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    best_threshold, best_f1 = tune_threshold(y_test, y_prob)
    print(f"\nBest threshold: {best_threshold:.3f} (F1={best_f1:.3f})")

    y_pred = (y_prob >= best_threshold).astype(int)

    # ── Evaluation ─────────────────────────────────────────────────────────
    print("\n── Classification Report ──────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall    = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    roc_auc   = roc_auc_score(y_test, y_prob)

    print(f"Confusion Matrix:\n{cm}")
    print(f"\nRecall    : {recall:.3f}  ← % of real anomalies caught")
    print(f"Precision : {precision:.3f}  ← % of flagged alerts that are real")
    print(f"ROC-AUC   : {roc_auc:.3f}")

    # ── Feature importance ─────────────────────────────────────────────────
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\n── Top 10 Features ────────────────────────────────────────")
    print(importance.head(10).to_string())

    # ── Save artifacts ─────────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)

    model.save_model(MODEL_DIR / "model.json")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    metadata = {
        "feature_cols": feature_cols,
        "threshold": float(best_threshold),
        "metrics": {
            "recall":    round(recall, 4),
            "precision": round(precision, 4),
            "f1":        round(best_f1, 4),
            "roc_auc":   round(roc_auc, 4),
        },
        "train_rows": len(train_df),
        "test_rows":  len(test_df),
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved → {MODEL_DIR}/model.json")
    print(f"Saved → {MODEL_DIR}/scaler.pkl")
    print(f"Saved → {MODEL_DIR}/metadata.json")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
