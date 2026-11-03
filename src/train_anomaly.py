"""Train and evaluate an Isolation Forest on MetroPT-3."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

INPUT_PATH = Path("data/processed/metropt_features.csv.gz")
MODEL_PATH = Path("models/metropt_isolation_forest.joblib")

TARGET = "failure_within_24h"

EXCLUDED_COLUMNS = {
    "Unnamed: 0",
    "timestamp",
    TARGET,
    "is_active_failure",
    "has_full_24h_future",
    "has_gap_in_previous_24h",
    "has_full_24h_history",
    "eligible_for_training",
}


def main():
    print("Loading MetroPT-3 features...")

    df = pd.read_csv(
        INPUT_PATH,
        compression="gzip",
        parse_dates=["timestamp"],
    )

    df = df[df["eligible_for_training"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Identify the E3 pre-failure window from positive labels.
    positive = df[df[TARGET] == 1]

    if positive.empty:
        raise ValueError("No positive failure labels found.")

    event_start = positive["timestamp"].min()

    # Train only on observations before the positive event window.
    train = df[df["timestamp"] < event_start].copy()

    # Test on the positive event window and all later eligible data.
    test = df[df["timestamp"] >= event_start].copy()

    feature_cols = [
        col for col in df.columns
        if col not in EXCLUDED_COLUMNS
    ]

    X_train = train[feature_cols].astype("float32")
    X_test = test[feature_cols].astype("float32")

    y_test = test[TARGET].astype(int)

    print("\n========== DATA SUMMARY ==========")
    print(f"Features: {len(feature_cols)}")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Test positive labels: {int(y_test.sum()):,}")
    print(f"Test negative labels: {int((y_test == 0).sum()):,}")

    print("\nTraining Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train)

    # Higher scores mean more anomalous.
    anomaly_scores = -model.decision_function(X_test)

    # Set threshold using the 99th percentile of training anomaly scores.
    train_scores = -model.decision_function(X_train)
    threshold = np.percentile(train_scores, 99)

    predictions = (anomaly_scores >= threshold).astype(int)

    print("\n========== EVALUATION ==========")

    print(f"ROC-AUC: {roc_auc_score(y_test, anomaly_scores):.4f}")
    print(
        "PR-AUC (Average Precision): "
        f"{average_precision_score(y_test, anomaly_scores):.4f}"
    )
    print(f"Training-derived anomaly threshold: {threshold:.6f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "threshold": threshold,
        },
        MODEL_PATH,
    )

    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
