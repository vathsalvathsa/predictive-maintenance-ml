"""Create chronological train, validation, and test datasets."""

from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/metropt_features.csv.gz")
OUTPUT_DIR = Path("data/processed")

TARGET = "failure_within_24h"
GAP = pd.Timedelta(hours=24)


def main():
    print("Loading MetroPT-3 feature dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        compression="gzip",
        parse_dates=["timestamp"]
    )

    # Keep only observations approved for training.
    df = df[df["eligible_for_training"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Identify model features and remove rows with incomplete features.
    excluded = {
        "Unnamed: 0",
        "timestamp",
        TARGET,
        "is_active_failure",
        "has_full_24h_future",
        "has_gap_in_previous_24h",
        "has_full_24h_history",
        "eligible_for_training",
    }

    feature_cols = [c for c in df.columns if c not in excluded]

    before = len(df)
    df = df.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)

    print(f"Eligible rows before feature cleanup: {before:,}")
    print(f"Rows after removing incomplete features: {len(df):,}")

    # Find chronological boundaries.
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_boundary = df.loc[train_end, "timestamp"]
    val_boundary = df.loc[val_end, "timestamp"]

    # Purge observations whose 24-hour target horizon crosses a boundary.
    train = df.iloc[:train_end].copy()
    train = train[train["timestamp"] < train_boundary - GAP]

    validation = df.iloc[train_end:val_end].copy()
    validation = validation[
        validation["timestamp"] >= train_boundary + GAP
    ]
    validation = validation[
        validation["timestamp"] < val_boundary - GAP
    ]

    test = df.iloc[val_end:].copy()
    test = test[test["timestamp"] >= val_boundary + GAP]

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    # Save only timestamp, target, and model features.
    save_cols = ["timestamp"] + feature_cols + [TARGET]

    print("\n========== CHRONOLOGICAL SPLIT SUMMARY ==========")

    for name, split in splits.items():
        output_path = OUTPUT_DIR / f"metropt_{name}.csv.gz"
        split[save_cols].to_csv(
            output_path,
            index=False,
            compression="gzip"
        )

        positives = int(split[TARGET].sum())
        negatives = len(split) - positives

        print(f"\n{name.upper()}")
        print(f"Rows: {len(split):,}")
        print(f"Positive labels: {positives:,}")
        print(f"Negative labels: {negatives:,}")

        if len(split):
            print(f"Start: {split['timestamp'].min()}")
            print(f"End:   {split['timestamp'].max()}")

        print(f"Saved: {output_path}")

    print("\nChronological splitting completed.")


if __name__ == "__main__":
    main()
