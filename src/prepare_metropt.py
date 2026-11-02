
"""Prepare MetroPT-3 data and construct a 24-hour failure target.

Raw sensor data is never modified.
Long historical timestamp gaps are flagged, not interpolated.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. File paths and configuration
# --------------------------------------------------

RAW_PATH = Path("data/raw/MetroPT3(AirCompressor).csv")
EVENTS_PATH = Path("docs/failure_events.csv")
OUTPUT_PATH = Path("data/processed/metropt_labeled.csv.gz")

WARNING_HOURS = 24
MAX_HISTORICAL_GAP_SECONDS = 60


# --------------------------------------------------
# 2. Load and validate the data
# --------------------------------------------------

def load_data():
    print("Loading MetroPT-3 dataset...")

    df = pd.read_csv(RAW_PATH)
    events = pd.read_csv(EVENTS_PATH)

    required_event_columns = {
        "event_id", "start_time", "end_time", "failure", "severity"
    }

    if not required_event_columns.issubset(events.columns):
        raise ValueError("Failure event CSV is missing required columns.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="raise"
    )

    events["start_time"] = pd.to_datetime(
        events["start_time"], errors="raise"
    )

    events["end_time"] = pd.to_datetime(
        events["end_time"], errors="raise"
    )

    if df["timestamp"].isna().any():
        raise ValueError("Dataset contains missing timestamps.")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Dataset timestamps are not chronological.")

    if df["timestamp"].duplicated().any():
        raise ValueError("Dataset contains duplicate timestamps.")

    if (events["end_time"] < events["start_time"]).any():
        raise ValueError("A failure event has an invalid time interval.")

    events = events.sort_values("start_time").reset_index(drop=True)

    if (events["start_time"].iloc[1:].reset_index(drop=True)
            < events["end_time"].iloc[:-1].reset_index(drop=True)).any():
        raise ValueError("Failure event periods overlap.")

    print(f"Sensor rows: {len(df):,}")
    print(f"Failure events: {len(events)}")

    return df, events


# --------------------------------------------------
# 3. Create the 24-hour prediction target
# --------------------------------------------------

def create_target(df, events):
    print("\nCreating 24-hour failure target...")

    timestamps = df["timestamp"].to_numpy(dtype="datetime64[ns]")
    failure_starts = events["start_time"].to_numpy(
        dtype="datetime64[ns]"
    )
    failure_ends = events["end_time"].to_numpy(
        dtype="datetime64[ns]"
    )

    horizon = np.timedelta64(WARNING_HOURS, "h")

    # Find the next reported failure strictly after each observation.
    next_event_index = np.searchsorted(
        failure_starts, timestamps, side="right"
    )

    target = np.zeros(len(df), dtype=np.int8)

    has_future_event = next_event_index < len(failure_starts)

    target[has_future_event] = (
        failure_starts[next_event_index[has_future_event]]
        <= timestamps[has_future_event] + horizon
    ).astype(np.int8)

    # Mark observations inside reported failure periods.
    active_failure = np.zeros(len(df), dtype=bool)

    for start, end in zip(failure_starts, failure_ends):
        active_failure |= (
            (timestamps >= start) & (timestamps <= end)
        )

    # The target is undefined during an active reported failure.
    target = target.astype(float)
    target[active_failure] = np.nan

    # The target is also undefined when a complete future
    # 24-hour observation window is unavailable.
    has_full_future = (
        timestamps + horizon <= timestamps[-1]
    )

    target[~has_full_future] = np.nan

    df["has_full_24h_future"] = has_full_future
    df["failure_within_24h"] = target
    df["is_active_failure"] = active_failure

    return df


# --------------------------------------------------
# 4. Check historical data quality
# --------------------------------------------------

def check_historical_gaps(df):
    print("\nChecking historical timestamp gaps...")

    timestamps = df["timestamp"].to_numpy(dtype="datetime64[ns]")

    gap_seconds = (
        np.diff(timestamps) / np.timedelta64(1, "s")
    )

    gap_end_times = timestamps[1:][
        gap_seconds > MAX_HISTORICAL_GAP_SECONDS
    ]

    horizon = np.timedelta64(WARNING_HOURS, "h")

    # Find gaps ending inside each observation's previous 24 hours.
    left = np.searchsorted(
        gap_end_times, timestamps - horizon, side="right"
    )

    right = np.searchsorted(
        gap_end_times, timestamps, side="right"
    )

    has_historical_gap = right > left

    has_full_history = (
        timestamps - timestamps[0] >= horizon
    )

    df["has_gap_in_previous_24h"] = has_historical_gap
    df["has_full_24h_history"] = has_full_history

    df["eligible_for_training"] = (
        has_full_history
        & ~has_historical_gap
        & ~df["is_active_failure"].to_numpy()
    )

    return df


# --------------------------------------------------
# 5. Save the prepared dataset
# --------------------------------------------------

def main():
    df, events = load_data()

    df = create_target(df, events)
    df = check_historical_gaps(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False, compression="gzip")

    print("\n========== PREPARATION SUMMARY ==========")
    print(f"Total rows: {len(df):,}")
    print(f"Positive labels: {(df['failure_within_24h'] == 1).sum():,}")
    print(f"Negative labels: {(df['failure_within_24h'] == 0).sum():,}")
    print(f"Active failure rows: {df['is_active_failure'].sum():,}")
    print(
        "Rows with historical gaps: "
        f"{df['has_gap_in_previous_24h'].sum():,}"
    )
    print(
        "Rows with full 24-hour history: "
        f"{df['has_full_24h_history'].sum():,}"
    )
    print(
        "Training-eligible rows: "
        f"{df['eligible_for_training'].sum():,}"
    )

    print(f"\nSaved prepared dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()