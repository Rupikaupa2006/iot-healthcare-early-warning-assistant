from pathlib import Path
import pandas as pd


DATASET_DIR = Path(__file__).parent / "bidmc_csv"


print("=" * 70)
print("BIDMC DATASET INSPECTION")
print("=" * 70)

if not DATASET_DIR.exists():
    print(f"Dataset folder not found: {DATASET_DIR}")
    raise SystemExit(1)


numerics_files = sorted(DATASET_DIR.glob("*_Numerics.csv"))

print(f"\nNumerics files found: {len(numerics_files)}")

if not numerics_files:
    print("No Numerics CSV files found.")
    raise SystemExit(1)


for file in numerics_files[:3]:

    print("\n" + "-" * 70)
    print(f"FILE: {file.name}")
    print("-" * 70)

    df = pd.read_csv(file)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nShape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)