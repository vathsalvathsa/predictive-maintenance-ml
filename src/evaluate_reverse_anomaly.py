"""Exploratory evaluation using the reversed Isolation Forest ranking."""

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

DATA_PATH = "data/processed/metropt_features.csv.gz"
MODEL_PATH = "models/metropt_isolation_forest.joblib"


def main():
    df = pd.read_csv(
        DATA_PATH,
        compression="gzip",
        parse_dates=["timestamp"],
    )

    df = df[df["eligible_for_training"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    event_start = df.loc[
        df["failure_within_24h"] == 1, "timestamp"
    ].min()

    train = df[df["timestamp"] < event_start]
    test = df[df["timestamp"] >= event_start]

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["feature_columns"]

    X_train = train[features].astype("float32")
    X_test = test[features].astype("float32")
    y_test = test["failure_within_24h"].astype(int)

    # Correct reversed ranking: higher decision scores are flagged.
    train_scores = model.decision_function(X_train)
    test_scores = model.decision_function(X_test)

    threshold = pd.Series(train_scores).quantile(0.99)
    predictions = (test_scores >= threshold).astype(int)

    print("\n========== REVERSED RANKING EXPERIMENT ==========")
    print(f"Threshold: {threshold:.6f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, test_scores):.4f}")
    print(
        "Average precision: "
        f"{average_precision_score(y_test, test_scores):.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print(f"Predicted positives: {predictions.sum():,}")
    print(f"Actual positives: {y_test.sum():,}")


if __name__ == "__main__":
    main()
