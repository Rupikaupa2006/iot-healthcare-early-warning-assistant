from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_DIR
    / "dataset"
    / "bidmc_clean.csv"
)

MODEL_FILE = (
    PROJECT_DIR
    / "anomaly"
    / "bidmc_isolation_forest.joblib"
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

FEATURES = [
    "heart_rate",
    "pulse_rate",
    "spo2",
    "respiratory_rate"
]

TRAIN_RATIO = 0.70


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 70)
print("BIDMC REAL-DATA ISOLATION FOREST TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_FILE)

print(f"Total rows: {len(df):,}")
print(f"Total recordings: {df['recording_id'].nunique()}")


# --------------------------------------------------
# CREATE TEMPORAL TRAINING SET
# --------------------------------------------------

training_parts = []
testing_parts = []


for recording_id, group in df.groupby("recording_id"):

    group = group.sort_values("time").reset_index(drop=True)

    split_index = int(len(group) * TRAIN_RATIO)

    train_part = group.iloc[:split_index]
    test_part = group.iloc[split_index:]

    training_parts.append(train_part)
    testing_parts.append(test_part)


training_df = pd.concat(
    training_parts,
    ignore_index=True
)

testing_df = pd.concat(
    testing_parts,
    ignore_index=True
)


print("\nTemporal split:")
print(f"Training rows: {len(training_df):,}")
print(f"Testing rows:  {len(testing_df):,}")


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

X_train = training_df[FEATURES]

X_test = testing_df[FEATURES]


print("\nFeatures used by the model:")

for feature in FEATURES:
    print(f"  - {feature}")


# --------------------------------------------------
# MODEL
# --------------------------------------------------

print("\nTraining Isolation Forest...")


model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "isolation_forest",
        IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1
        )
    )
])


model.fit(X_train)


# --------------------------------------------------
# TEST ON UNSEEN TEMPORAL DATA
# --------------------------------------------------

predictions = model.predict(X_test)

scores = model.decision_function(X_test)


testing_df = testing_df.copy()

testing_df["ml_prediction"] = predictions

testing_df["anomaly_score"] = scores


# Isolation Forest:
# +1 = normal
# -1 = anomaly

anomaly_count = (
    testing_df["ml_prediction"] == -1
).sum()

normal_count = (
    testing_df["ml_prediction"] == 1
).sum()


anomaly_percentage = (
    anomaly_count / len(testing_df)
) * 100


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)


# --------------------------------------------------
# REPORT
# --------------------------------------------------

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print(f"\nModel saved to:")
print(MODEL_FILE)

print("\nTraining observations:")
print(f"{len(training_df):,}")

print("\nTesting observations:")
print(f"{len(testing_df):,}")

print("\nTest-set ML results:")
print(f"Normal observations:  {normal_count:,}")
print(f"Anomalous observations: {anomaly_count:,}")
print(f"Anomaly percentage: {anomaly_percentage:.2f}%")

print("\nAnomaly score statistics:")
print(
    testing_df["anomaly_score"]
    .describe()
    .round(4)
)

print("\nSample predictions:")
print(
    testing_df[
        [
            "patient_id",
            "recording_id",
            "time",
            "heart_rate",
            "pulse_rate",
            "spo2",
            "respiratory_rate",
            "ml_prediction",
            "anomaly_score"
        ]
    ].head(10)
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)