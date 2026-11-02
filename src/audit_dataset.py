"""
MetroPT-3 Dataset Audit
Purpose: Perform an initial quality and statistical audit
of the original industrial sensor dataset.
"""

from pathlib import Path
import pandas as pd


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "MetroPT3(AirCompressor).csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"


def main():

    # 2. Verify that the dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # 3. Load the original dataset
    print("\nLoading MetroPT-3 dataset...")

    df = pd.read_csv(DATA_PATH)

    # 4. Create report directory
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== DATASET OVERVIEW ==========")

    # 5. Dataset dimensions
    print(f"Number of records: {df.shape[0]:,}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    # 6. Data types
    print("\n========== DATA TYPES ==========")
    print(df.dtypes.to_string())

    # 7. Missing-value analysis
    print("\n========== MISSING VALUES ==========")

    missing = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percentage": (
            df.isnull().mean() * 100
        )
    })

    print(missing.to_string())

    missing.to_csv(
        REPORT_DIR / "missing_values.csv"
    )

    # 8. Duplicate analysis
    print("\n========== DUPLICATE ANALYSIS ==========")

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_rows:,}")

    # 9. Numerical descriptive statistics
    print("\n========== SENSOR STATISTICS ==========")

    numeric_df = df.select_dtypes(
        include="number"
    )

    statistics = numeric_df.describe().T

    print(statistics.to_string())

    statistics.to_csv(
        REPORT_DIR / "sensor_statistics.csv"
    )

    # 10. Timestamp analysis
    print("\n========== TIMESTAMP ANALYSIS ==========")

    timestamp_columns = [
        column for column in df.columns
        if column.strip().lower() in {
            "timestamp", "datetime", "date"
        }
    ]

    if timestamp_columns:

        timestamp_column = timestamp_columns[0]

        timestamps = pd.to_datetime(
            df[timestamp_column],
            errors="coerce"
        )

        print(
            f"Timestamp column: {timestamp_column}"
        )

        print(
            f"Invalid timestamps: "
            f"{timestamps.isna().sum():,}"
        )

        valid_timestamps = timestamps.dropna()

        if not valid_timestamps.empty:

            print(
                f"First timestamp: "
                f"{valid_timestamps.min()}"
            )

            print(
                f"Last timestamp: "
                f"{valid_timestamps.max()}"
            )

            intervals = (
                valid_timestamps.sort_values()
                .diff()
                .dt.total_seconds()
                .dropna()
            )

            if not intervals.empty:

                print(
                    "\nSampling interval statistics "
                    "(seconds):"
                )

                print(
                    intervals.describe().to_string()
                )

                print(
                    "\nMost frequent intervals:"
                )

                print(
                    intervals.value_counts()
                    .head(10)
                    .to_string()
                )

    else:
        print("No recognized timestamp column found.")

    # 11. IQR-based outlier screening
    print("\n========== OUTLIER SCREENING ==========")

    outlier_results = []

    for column in numeric_df.columns:

        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = (
            (numeric_df[column] < lower_bound)
            | (numeric_df[column] > upper_bound)
        ).sum()

        outlier_results.append({
            "feature": column,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_percentage": (
                outlier_count / len(df) * 100
            )
        })

    outliers = pd.DataFrame(outlier_results)

    print(outliers.to_string(index=False))

    outliers.to_csv(
        REPORT_DIR / "outlier_screening.csv",
        index=False
    )

    # 12. Save a basic audit summary
    summary = [
        "METROPT-3 DATASET AUDIT",
        "=======================",
        f"Records: {df.shape[0]:,}",
        f"Columns: {df.shape[1]}",
        f"Duplicate rows: {duplicate_rows:,}",
        f"Missing values: {df.isnull().sum().sum():,}",
        f"Numeric columns: {len(numeric_df.columns)}",
    ]

    summary_path = REPORT_DIR / "audit_summary.txt"

    summary_path.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print("\n========== AUDIT COMPLETED ==========")
    print(f"Reports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
