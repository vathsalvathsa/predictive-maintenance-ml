"""Inspect Isolation Forest scores for normal and pre-failure observations."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

DATA_PATH = Path("data/processed/metropt_features.csv.gz")
MODEL_PATH = Path("models/metropt_isolation_forest.joblib")


def main():
    df = pd.read_csv(
        DATA_PATH,
        compression="gzip",
        parse_dates=["timestamp"],
    )

    df = df[df["eligible_for_training"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    positive = df[df["failure_within_24h"] == 1]
    event_start = positive["timestamp"].min()

    test = df[df["timestamp"] >= event_start].copy()

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["feature_columns"]

    X_test = test[features].astype("float32")
    y = test["failure_within_24h"].astype(int)

    scores = -model.decision_function(X_test)

    results = pd.DataFrame({
        "label": y.to_numpy(),
        "anomaly_score": scores,
    })

    print("ANOMALY SCORE DISTRIBUTION")
    print(
        results.groupby("label")["anomaly_score"]
        .describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
    )

    print("\nROC-AUC using anomaly score:")
    print(round(roc_auc_score(y, scores), 4))

    print("\nROC-AUC using reversed score:")
    print(round(roc_auc_score(y, -scores), 4))

    print("\nAverage precision using anomaly score:")
    print(round(average_precision_score(y, scores), 4))

    print("\nAverage precision using reversed score:")
    print(round(average_precision_score(y, -scores), 4))


if __name__ == "__main__":
    main()
