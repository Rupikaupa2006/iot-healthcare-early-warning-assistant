"""
Knowledge-Driven Personalized IoT Healthcare Monitoring
Early-Warning Analysis Engine

Dataset schema:
    patient_id
    recording_id
    time
    heart_rate
    pulse_rate
    spo2
    respiratory_rate

This module provides:
    - Personalized baseline calculation
    - Z-score deviation analysis
    - Multisensor pattern detection
    - Temporal pattern analysis
    - Pre-trained Isolation Forest anomaly detection
    - Combined monitoring status
    - BIDMC DataFrame/list compatibility

IMPORTANT:
This is an academic monitoring prototype.
It is NOT a clinical diagnostic or treatment system.
"""

import statistics
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "heart_rate",
    "pulse_rate",
    "spo2",
    "respiratory_rate",
]

FEATURE_LABELS = {
    "heart_rate": "Heart Rate",
    "pulse_rate": "Pulse Rate",
    "spo2": "SpO2",
    "respiratory_rate": "Respiratory Rate",
}

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "bidmc_isolation_forest.joblib"
)

Z_SCORE_THRESHOLD = 2.0


# ============================================================
# LOAD TRAINED ML MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_ml_model():
    """
    Load the Isolation Forest model trained on the
    real BIDMC dataset.

    The model is loaded once and reused instead of
    being loaded repeatedly.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}\n"
            "Run anomaly/train_bidmc_model.py first."
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value):
    """
    Convert a value to float safely.

    Returns None if conversion is not possible.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def z_score(value, mean, std):
    """
    Calculate z-score.

    If standard deviation is zero or unavailable,
    return 0 because no measurable variation exists.
    """

    value = safe_float(value)
    mean = safe_float(mean)
    std = safe_float(std)

    if value is None or mean is None or std is None:
        return 0.0

    if std == 0:
        return 0.0

    return (value - mean) / std


# ============================================================
# PERSONALIZED BASELINE
# ============================================================

def calculate_baseline(rows):
    """
    Calculate an individual's personalized baseline
    from historical observations.

    Expected row keys:
        heart_rate
        pulse_rate
        spo2
        respiratory_rate

    Returns:
        {
            feature: {
                "mean": ...,
                "std": ...,
                "count": ...
            }
        }
    """

    if rows is None:
        rows = []

    # Allow pandas DataFrame input
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict(orient="records")

    baseline = {}

    for feature in FEATURES:

        values = []

        for row in rows:

            if not isinstance(row, dict):
                continue

            value = safe_float(
                row.get(feature)
            )

            if value is not None:
                values.append(value)

        if not values:

            baseline[feature] = {
                "mean": 0.0,
                "std": 0.0,
                "count": 0,
            }

            continue

        mean_value = statistics.mean(values)

        if len(values) > 1:
            std_value = statistics.stdev(values)
        else:
            std_value = 0.0

        baseline[feature] = {
            "mean": round(mean_value, 4),
            "std": round(std_value, 4),
            "count": len(values),
        }

    return baseline


# ============================================================
# CURRENT READING ANALYSIS
# ============================================================

def analyze_current_reading(reading, baseline):
    """
    Compare the current physiological observation
    against the personalized historical baseline.
    """

    if reading is None:
        raise ValueError(
            "Current reading is required."
        )

    if baseline is None:
        baseline = {}

    z_scores = {}
    deviations = []

    for feature in FEATURES:

        value = safe_float(
            reading.get(feature)
        )

        if feature not in baseline:

            z_scores[feature] = 0.0
            continue

        mean_value = baseline[feature]["mean"]
        std_value = baseline[feature]["std"]

        score = z_score(
            value,
            mean_value,
            std_value
        )

        z_scores[feature] = round(
            score,
            4
        )

        if abs(score) >= Z_SCORE_THRESHOLD:

            direction = (
                "above baseline"
                if score > 0
                else "below baseline"
            )

            deviations.append({
                "sensor": feature,
                "label": FEATURE_LABELS[feature],
                "value": value,
                "baseline_mean": mean_value,
                "baseline_std": std_value,
                "z_score": round(score, 4),
                "direction": direction,
            })

    return {
        "z_scores": z_scores,
        "deviations": deviations,
    }


# ============================================================
# MULTISENSOR / TEMPORAL ANALYSIS
# ============================================================

def analyze_temporal_pattern(
    recent_rows,
    baseline,
    min_deviating_sensors=2,
):
    """
    Examine recent observations for repeated
    multisensor deviations from the personalized baseline.
    """

    if recent_rows is None:
        recent_rows = []

    # Convert DataFrame if necessary
    if isinstance(recent_rows, pd.DataFrame):
        recent_rows = recent_rows.to_dict(
            orient="records"
        )

    if len(recent_rows) == 0:

        return {
            "pattern": "insufficient_data",
            "events": [],
            "deviating_sensor_counts": {
                feature: 0
                for feature in FEATURES
            },
        }

    events = []

    sensor_counts = {
        feature: 0
        for feature in FEATURES
    }

    for index, row in enumerate(recent_rows):

        row_result = analyze_current_reading(
            row,
            baseline
        )

        deviations = row_result["deviations"]

        if len(deviations) >= min_deviating_sensors:

            event = {
                "index": index,
                "timestamp": row.get("timestamp"),
                "time": row.get("time"),
                "deviating_sensors": [
                    deviation["sensor"]
                    for deviation in deviations
                ],
                "deviation_count": len(deviations),
            }

            events.append(event)

        for deviation in deviations:

            sensor = deviation["sensor"]

            if sensor in sensor_counts:
                sensor_counts[sensor] += 1

    if not events:

        pattern = "no_multisensor_pattern"

    elif len(events) >= 2:

        pattern = "persistent_multisensor_pattern"

    else:

        pattern = "intermittent_multisensor_pattern"

    return {
        "pattern": pattern,
        "events": events,
        "deviating_sensor_counts": sensor_counts,
    }


# ============================================================
# ISOLATION FOREST
# ============================================================

def run_isolation_forest(current_reading):
    """
    Use the pre-trained Isolation Forest model.

    Feature order MUST match the training script:

        heart_rate
        pulse_rate
        spo2
        respiratory_rate
    """

    if current_reading is None:
        raise ValueError(
            "Current reading is required for ML analysis."
        )

    model = load_ml_model()

    feature_vector = {}

    for feature in FEATURES:

        value = safe_float(
            current_reading.get(feature)
        )

        if value is None:

            raise ValueError(
                f"Missing required feature: {feature}"
            )

        feature_vector[feature] = value

    # Use DataFrame so feature names match
    # those used during model training.
    feature_df = pd.DataFrame(
        [feature_vector],
        columns=FEATURES
    )

    prediction = model.predict(
        feature_df
    )[0]

    score = model.decision_function(
        feature_df
    )[0]

    if prediction == -1:
        result = "ANOMALY"
    else:
        result = "NORMAL"

    return {
        "prediction": int(prediction),
        "result": result,
        "score": round(
            float(score),
            6
        ),
    }


# ============================================================
# FINAL MONITORING STATUS
# ============================================================

def calculate_final_status(
    deviations,
    ml_prediction,
    temporal_pattern=None,
):
    """
    Combine multiple monitoring signals.

    NORMAL:
        No baseline deviations and no ML anomaly.

    WATCH:
        One monitoring signal is present.

    ATTENTION:
        Multiple independent monitoring signals
        are present.

    These labels represent prototype monitoring states,
    not medical diagnoses.
    """

    if deviations is None:
        deviations = []

    signal_count = 0

    # --------------------------------------------------------
    # Signal 1: baseline deviation
    # --------------------------------------------------------

    if len(deviations) >= 1:
        signal_count += 1

    # --------------------------------------------------------
    # Signal 2: multisensor deviation
    # --------------------------------------------------------

    if len(deviations) >= 2:
        signal_count += 1

    # --------------------------------------------------------
    # Signal 3: ML anomaly
    # --------------------------------------------------------

    if ml_prediction == -1:
        signal_count += 1

    # --------------------------------------------------------
    # Signal 4: persistent temporal pattern
    # --------------------------------------------------------

    if temporal_pattern == "persistent_multisensor_pattern":
        signal_count += 1

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if signal_count == 0:

        status = "NORMAL"

    elif signal_count == 1:

        status = "WATCH"

    else:

        status = "ATTENTION"

    return {
        "status": status,
        "signal_count": signal_count,
    }


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_reading(
    current_reading,
    historical_rows,
    recent_rows=None,
):
    """
    Run the complete monitoring pipeline.

    Pipeline:

        Historical observations
                ↓
        Personalized baseline
                ↓
        Current deviation analysis
                ↓
        Temporal / multisensor analysis
                ↓
        Isolation Forest
                ↓
        Combined monitoring status
    """

    if current_reading is None:
        raise ValueError(
            "Current reading is required."
        )

    # Convert DataFrame inputs to lists
    if isinstance(historical_rows, pd.DataFrame):

        historical_rows = historical_rows.to_dict(
            orient="records"
        )

    if isinstance(recent_rows, pd.DataFrame):

        recent_rows = recent_rows.to_dict(
            orient="records"
        )

    if historical_rows is None or len(historical_rows) == 0:

        raise ValueError(
            "Historical rows are required to calculate "
            "a personalized baseline."
        )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline = calculate_baseline(
        historical_rows
    )

    # --------------------------------------------------------
    # CURRENT READING
    # --------------------------------------------------------

    current_analysis = analyze_current_reading(
        current_reading,
        baseline
    )

    z_scores = current_analysis["z_scores"]

    deviations = current_analysis["deviations"]

    # --------------------------------------------------------
    # RECENT HISTORY
    # --------------------------------------------------------

    if recent_rows is None:

        recent_rows = historical_rows[-10:]

    temporal_analysis = analyze_temporal_pattern(
        recent_rows,
        baseline
    )

    temporal_pattern = temporal_analysis["pattern"]

    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    ml_result = run_isolation_forest(
        current_reading
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    final_status = calculate_final_status(
        deviations=deviations,
        ml_prediction=ml_result["prediction"],
        temporal_pattern=temporal_pattern,
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {

        "device_id":
            current_reading.get("device_id"),

        "patient_id":
            current_reading.get("patient_id"),

        "recording_id":
            current_reading.get("recording_id"),

        "timestamp":
            current_reading.get("timestamp"),

        "time":
            current_reading.get("time"),

        "heart_rate":
            safe_float(
                current_reading.get(
                    "heart_rate"
                )
            ),

        "pulse_rate":
            safe_float(
                current_reading.get(
                    "pulse_rate"
                )
            ),

        "spo2":
            safe_float(
                current_reading.get(
                    "spo2"
                )
            ),

        "respiratory_rate":
            safe_float(
                current_reading.get(
                    "respiratory_rate"
                )
            ),

        "baseline":
            baseline,

        "z_scores":
            z_scores,

        "deviations":
            deviations,

        "multisensor_events":
            temporal_analysis["events"],

        "deviating_sensor_counts":
            temporal_analysis[
                "deviating_sensor_counts"
            ],

        "temporal_pattern":
            temporal_pattern,

        "ml_prediction":
            ml_result["prediction"],

        "ml_result":
            ml_result["result"],

        "ml_score":
            ml_result["score"],

        "signal_count":
            final_status["signal_count"],

        "status":
            final_status["status"],
    }


# ============================================================
# BIDMC CSV HELPER
# ============================================================

def analyze_bidmc_rows(
    rows,
    current_index=None,
    recent_window=10,
):
    """
    Analyze a sequence of BIDMC observations.

    Accepts either:

        - pandas DataFrame
        - list of dictionaries

    The observations should belong to one recording/patient
    and should already be sorted chronologically.

    Important:
        The DataFrame must contain the physiological columns:

            heart_rate
            pulse_rate
            spo2
            respiratory_rate
    """

    # --------------------------------------------------------
    # Convert pandas DataFrame into records
    # --------------------------------------------------------

    if isinstance(rows, pd.DataFrame):

        if rows.empty:

            raise ValueError(
                "The provided BIDMC DataFrame is empty."
            )

        rows = rows.to_dict(
            orient="records"
        )

    # --------------------------------------------------------
    # Validate rows
    # --------------------------------------------------------

    if rows is None or len(rows) == 0:

        raise ValueError(
            "No BIDMC observations were provided."
        )

    # --------------------------------------------------------
    # Validate required columns/keys
    # --------------------------------------------------------

    first_row = rows[0]

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in first_row
    ]

    if missing_features:

        raise ValueError(
            "Missing required BIDMC features: "
            + ", ".join(missing_features)
        )

    # --------------------------------------------------------
    # Select current observation
    # --------------------------------------------------------

    if current_index is None:

        current_index = len(rows) - 1

    if current_index < 0 or current_index >= len(rows):

        raise IndexError(
            "current_index is outside the available rows."
        )

    current_reading = rows[current_index]

    # --------------------------------------------------------
    # Historical observations
    # --------------------------------------------------------

    historical_rows = rows[:current_index]

    if len(historical_rows) == 0:

        raise ValueError(
            "At least one historical observation is "
            "required before the current observation."
        )

    # --------------------------------------------------------
    # Recent observations
    # --------------------------------------------------------

    if recent_window is None or recent_window <= 0:

        recent_window = 10

    recent_rows = historical_rows[-recent_window:]

    # --------------------------------------------------------
    # Run complete analysis
    # --------------------------------------------------------

    return analyze_reading(
        current_reading=current_reading,
        historical_rows=historical_rows,
        recent_rows=recent_rows,
    )


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BIDMC HEALTH ANALYSIS ENGINE TEST")
    print("=" * 60)

    sample_history = [

        {
            "heart_rate": 80,
            "pulse_rate": 79,
            "spo2": 98,
            "respiratory_rate": 16,
        },

        {
            "heart_rate": 82,
            "pulse_rate": 81,
            "spo2": 97,
            "respiratory_rate": 17,
        },

        {
            "heart_rate": 81,
            "pulse_rate": 80,
            "spo2": 98,
            "respiratory_rate": 16,
        },

        {
            "heart_rate": 83,
            "pulse_rate": 82,
            "spo2": 97,
            "respiratory_rate": 17,
        },

    ]

    current = {

        "patient_id": "BIDMC_01",

        "recording_id": "BIDMC_01",

        "time": 300,

        "heart_rate": 90,

        "pulse_rate": 89,

        "spo2": 96,

        "respiratory_rate": 20,

    }

    result = analyze_reading(
        current_reading=current,
        historical_rows=sample_history,
        recent_rows=sample_history,
    )

    print("\nPatient:")
    print(
        result["patient_id"]
    )

    print("\nCurrent reading:")

    print(
        "HR:",
        result["heart_rate"]
    )

    print(
        "Pulse:",
        result["pulse_rate"]
    )

    print(
        "SpO2:",
        result["spo2"]
    )

    print(
        "Respiratory Rate:",
        result["respiratory_rate"]
    )

    print("\nZ-scores:")

    for sensor, score in result["z_scores"].items():

        print(
            f"{FEATURE_LABELS[sensor]}: {score}"
        )

    print("\nDeviations:")

    for deviation in result["deviations"]:

        print(
            f"- {deviation['label']}: "
            f"{deviation['direction']} "
            f"(z={deviation['z_score']})"
        )

    print("\nTemporal pattern:")

    print(
        result["temporal_pattern"]
    )

    print("\nML result:")

    print(
        result["ml_result"]
    )

    print("\nML score:")

    print(
        result["ml_score"]
    )

    print("\nFinal monitoring status:")

    print(
        result["status"]
    )

    print("\nSignal count:")

    print(
        result["signal_count"]
    )

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)