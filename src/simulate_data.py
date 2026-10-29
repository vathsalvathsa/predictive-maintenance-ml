"""Generates 40,000 simulated sensor records for predictive maintenance."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_RECORDS = 40_000
N_MACHINES = 20
FAILURE_RATE = 0.08
OUTPUT_PATH = os.path.join("data", "sensor_data.csv")


def generate_data(n_records=N_RECORDS, n_machines=N_MACHINES,
                  failure_rate=FAILURE_RATE):
    records = []
    for machine_id in range(1, n_machines + 1):
        base_temp = np.random.uniform(60, 75)
        base_vib = np.random.uniform(0.2, 0.5)
        base_press = np.random.uniform(90, 110)
        base_rpm = np.random.uniform(1400, 1600)
        base_current = np.random.uniform(8, 12)
        machine_records = n_records // n_machines
        n_failures = max(1, int(machine_records * failure_rate / 24))
        failure_starts = np.random.choice(
            range(72, machine_records - 24), size=n_failures, replace=False)
        for i in range(machine_records):
            in_pre_failure = any(fs - 72 <= i < fs for fs in failure_starts)
            in_failure_window = any(fs - 24 <= i < fs for fs in failure_starts)
            drift = 1.15 if in_pre_failure else 1.0
            temp = base_temp * drift + np.random.normal(0, 2)
            vib = base_vib * drift + np.random.normal(0, 0.05)
            press = base_press * drift + np.random.normal(0, 3)
            rpm = base_rpm + (np.random.normal(0, 200) if in_pre_failure
                              else np.random.normal(0, 80))
            current = base_current * drift + np.random.normal(0, 0.4)
            records.append({
                "machine_id": machine_id,
                "timestamp_index": i,
                "temperature": round(temp, 2),
                "vibration": round(vib, 3),
                "pressure": round(press, 2),
                "rpm": round(rpm, 1),
                "current": round(current, 2),
                "failure_within_24h": int(in_failure_window),
            })
    df = pd.DataFrame(records)
    for col in ["temperature", "vibration", "pressure", "rpm", "current"]:
        mask = np.random.rand(len(df)) < 0.02
        df.loc[mask, col] = np.nan
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} records -> {OUTPUT_PATH}")
    print(f"Failure rate: {df['failure_within_24h'].mean():.2%}")
    print(df.head())
