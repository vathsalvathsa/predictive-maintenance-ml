"""Feature engineering for predictive maintenance."""

import pandas as pd

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "current"]
WINDOW = 6


def add_features(df):
    df = df.sort_values(["machine_id", "timestamp_index"]).copy()
    for col in SENSOR_COLS:
        df[f"{col}_roll_mean"] = (
            df.groupby("machine_id")[col]
              .transform(lambda s: s.rolling(WINDOW, min_periods=1).mean())
        )
        df[f"{col}_roc"] = (
            df.groupby("machine_id")[col].diff().fillna(0)
        )
    df["temp_vib_ratio"] = df["temperature"] / (df["vibration"] + 1e-6)
    df["press_temp_ratio"] = df["pressure"] / (df["temperature"] + 1e-6)
    df["current_rpm_ratio"] = df["current"] / (df["rpm"] + 1e-6)
    return df


def get_feature_columns(df):
    exclude = {"machine_id", "timestamp_index", "failure_within_24h"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    df = pd.read_csv("data/sensor_data.csv")
    out = add_features(df)
    out.to_csv("data/sensor_data_features.csv", index=False)
    print(f"Saved featured dataset -> data/sensor_data_features.csv")
    print(f"Total columns: {len(out.columns)}")
    print(f"Feature columns: {len(get_feature_columns(out))}")
