from pathlib import Path
import pandas as pd


DATASET_DIR = Path(__file__).parent / "bidmc_csv"
OUTPUT_FILE = Path(__file__).parent / "bidmc_clean.csv"


print("=" * 70)
print("BIDMC DATASET PREPROCESSING")
print("=" * 70)


# Find all Numerics files
files = sorted(DATASET_DIR.glob("*_Numerics.csv"))

if not files:
    raise FileNotFoundError(
        f"No Numerics CSV files found in {DATASET_DIR}"
    )


print(f"\nFound {len(files)} recording files.")


all_data = []


for file in files:

    print(f"Processing: {file.name}")

    # Read CSV
    df = pd.read_csv(file)

    # Remove spaces from column names
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # Rename BIDMC columns
    df = df.rename(columns={
        "Time [s]": "time",
        "HR": "heart_rate",
        "PULSE": "pulse_rate",
        "RESP": "respiratory_rate",
        "SpO2": "spo2"
    })

    # Get recording number
    recording_id = file.stem.replace(
        "_Numerics",
        ""
    )

    df["recording_id"] = recording_id

    # Add patient ID
    number = recording_id.split("_")[-1]

    df["patient_id"] = (
        "BIDMC_" + number
    )

    # Keep only the columns we need
    df = df[
        [
            "patient_id",
            "recording_id",
            "time",
            "heart_rate",
            "pulse_rate",
            "spo2",
            "respiratory_rate"
        ]
    ]

    # Convert measurements to numeric
    numeric_columns = [
        "time",
        "heart_rate",
        "pulse_rate",
        "spo2",
        "respiratory_rate"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Interpolate missing physiological measurements
    measurement_columns = [
        "heart_rate",
        "pulse_rate",
        "spo2",
        "respiratory_rate"
    ]

    df[measurement_columns] = (
        df[measurement_columns]
        .interpolate(method="linear")
        .ffill()
        .bfill()
    )

    all_data.append(df)


# Combine all recordings
final_df = pd.concat(
    all_data,
    ignore_index=True
)


# Sort data
final_df = final_df.sort_values(
    ["recording_id", "time"]
).reset_index(drop=True)


# Final column order
final_df = final_df[
    [
        "patient_id",
        "recording_id",
        "time",
        "heart_rate",
        "pulse_rate",
        "spo2",
        "respiratory_rate"
    ]
]


# Save
final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print(f"\nTotal rows: {len(final_df):,}")

print(
    f"Total recordings: "
    f"{final_df['recording_id'].nunique()}"
)

print("\nColumns:")
print(final_df.columns.tolist())

print("\nMissing values after preprocessing:")
print(final_df.isnull().sum())

print("\nRows per recording:")
print(
    final_df.groupby("recording_id")
    .size()
    .describe()
)

print("\nFirst 10 rows:")
print(final_df.head(10))

print("\nBasic statistics:")
print(
    final_df[
        [
            "heart_rate",
            "pulse_rate",
            "spo2",
            "respiratory_rate"
        ]
    ].describe()
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)