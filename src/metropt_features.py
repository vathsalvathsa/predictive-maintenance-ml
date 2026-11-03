"""Feature engineering for the MetroPT-3 predictive maintenance dataset."""

from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/metropt_labeled.csv.gz")
OUTPUT_PATH = Path("data/processed/metropt_features.csv.gz")

CONTINUOUS_COLS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
    "Oil_temperature", "Motor_current", "MPG", "Caudal_impulses",
]

STATUS_COLS = ["COMP", "DV_eletric", "Towers", "Pressure_switch"]
ROLLING_WINDOWS = [6, 36]


def load_data():
    print("Loading prepared MetroPT-3 data...")
    df = pd.read_csv(
        INPUT_PATH,
        compression="gzip",
        parse_dates=["timestamp"]
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def inspect_constant_columns(df):
    sensor_cols = CONTINUOUS_COLS + STATUS_COLS + ["LPS", "Oil_level"]

    print("\nChecking sensor variability...")
    for col in sensor_cols:
        print(f"{col}: {df[col].nunique()} unique values")


def add_features(df):
    print("\nEngineering sensor features...")
    df = df.copy()

    for col in CONTINUOUS_COLS:
        for window in ROLLING_WINDOWS:
            rolling = df[col].rolling(window=window, min_periods=window)
            df[f"{col}_mean_{window}"] = rolling.mean()
            df[f"{col}_std_{window}"] = rolling.std()

        df[f"{col}_change"] = df[col].diff()

    df["TP2_TP3_difference"] = df["TP2"] - df["TP3"]
    df["TP3_reservoir_difference"] = df["TP3"] - df["Reservoirs"]

    for col in STATUS_COLS:
        df[f"{col}_change"] = df[col].diff()

    return df


def get_feature_columns(df):
    excluded = {
        "Unnamed: 0",
        "timestamp",
        "failure_within_24h",
        "is_active_failure",
        "has_full_24h_future",
        "has_gap_in_previous_24h",
        "has_full_24h_history",
        "eligible_for_training",
    }

    return [col for col in df.columns if col not in excluded]


def main():
    df = load_data()
    inspect_constant_columns(df)

    df = add_features(df)
    feature_cols = get_feature_columns(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, compression="gzip")

    print("\n========== FEATURE ENGINEERING SUMMARY ==========")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
