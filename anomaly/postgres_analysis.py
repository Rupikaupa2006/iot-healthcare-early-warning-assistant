import os
import getpass
import pandas as pd
import psycopg2
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from anomaly.health_analysis import analyze_bidmc_rows


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "healthcare_iot")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# ============================================================
# BIDMC CONFIGURATION
# ============================================================

PATIENT_ID = "BIDMC_01"
RECORDING_ID = "bidmc_01"

# Number of recent observations used for analysis.
# The latest observation is compared with the previous ones
# to calculate the personalized baseline.
HISTORY_SIZE = 20


# ============================================================
# CONNECT TO POSTGRESQL
# ============================================================

def get_connection():
    """Create a connection to the healthcare_iot database."""

    password = DB_PASSWORD

    if not password:
        password = getpass.getpass(
            "Enter PostgreSQL password: "
        )

    print("\nConnecting to PostgreSQL...")

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=password
    )

    print("Connected to PostgreSQL successfully.")

    return connection


# ============================================================
# LOAD BIDMC DATA FROM POSTGRESQL
# ============================================================

def load_recent_bidmc_rows(
    connection,
    patient_id=PATIENT_ID,
    recording_id=RECORDING_ID,
    history_size=HISTORY_SIZE
):
    """
    Load recent BIDMC observations from PostgreSQL.

    The rows are returned in chronological order because
    the analysis engine expects the latest observation
    to be the final row.
    """

    query = """
        SELECT
            patient_id,
            recording_id,
            time_seconds,
            heart_rate,
            pulse_rate,
            spo2,
            respiratory_rate
        FROM bidmc_telemetry
        WHERE patient_id = %s
          AND recording_id = %s
        ORDER BY time_seconds DESC
        LIMIT %s;
    """

    print("\nLoading data from bidmc_telemetry...")

    df = pd.read_sql_query(
        query,
        connection,
        params=[
            patient_id,
            recording_id,
            history_size
        ]
    )

    if df.empty:
        raise ValueError(
            f"No telemetry found for "
            f"{patient_id} / {recording_id}"
        )

    # Convert database column name to the name
    # expected by health_analysis.py
    df = df.rename(
        columns={
            "time_seconds": "time"
        }
    )

    # Latest row should be the final row.
    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    return df


# ============================================================
# DISPLAY DATA
# ============================================================

def display_recent_data(df):
    """Display the observations loaded from PostgreSQL."""

    print("\n----------------------------------------")
    print("POSTGRESQL TELEMETRY")
    print("----------------------------------------")

    print(
        f"Patient: {df['patient_id'].iloc[-1]}"
    )

    print(
        f"Recording: {df['recording_id'].iloc[-1]}"
    )

    print(
        f"Observations loaded: {len(df)}"
    )

    print("\nLatest observation:")

    latest = df.iloc[-1]

    print(
        f"Time: {latest['time']}s"
    )

    print(
        f"Heart Rate: {latest['heart_rate']}"
    )

    print(
        f"Pulse Rate: {latest['pulse_rate']}"
    )

    print(
        f"SpO2: {latest['spo2']}"
    )

    print(
        f"Respiratory Rate: "
        f"{latest['respiratory_rate']}"
    )


# ============================================================
# RUN HEALTH ANALYSIS
# ============================================================

def run_analysis(df):
    """Send PostgreSQL data into the existing analysis engine."""

    print("\n----------------------------------------")
    print("HEALTHSENSE ANALYSIS")
    print("----------------------------------------")

    result = analyze_bidmc_rows(df)

    return result


# ============================================================
# DISPLAY ANALYSIS RESULT
# ============================================================

def display_analysis_result(result):

    print("\nPatient:")
    print(
        result.get("patient_id")
    )

    print("\nRecording:")
    print(
        result.get("recording_id")
    )

    print("\nCurrent Reading:")

    print(
        f"HR: {result.get('heart_rate')}"
    )

    print(
        f"Pulse: {result.get('pulse_rate')}"
    )

    print(
        f"SpO2: {result.get('spo2')}"
    )

    print(
        f"Respiratory Rate: "
        f"{result.get('respiratory_rate')}"
    )

    print("\nPersonalized Baseline:")

    baseline = result.get(
        "baseline",
        {}
    )

    for sensor, values in baseline.items():

        print(
            f"- {sensor}: "
            f"mean={values.get('mean')}, "
            f"std={values.get('std')}, "
            f"count={values.get('count')}"
        )

    print("\nZ-Scores:")

    for sensor, z_score in result.get(
        "z_scores",
        {}
    ).items():

        print(
            f"- {sensor}: {z_score}"
        )

    print("\nDeviations:")

    deviations = result.get(
        "deviations",
        []
    )

    if deviations:

        for deviation in deviations:

            print(
                f"- {deviation['label']}: "
                f"{deviation['value']} "
                f"({deviation['direction']}, "
                f"z={deviation['z_score']})"
            )

    else:

        print("- No significant baseline deviation detected.")

    print("\nTemporal Pattern:")

    print(
        result.get(
            "temporal_pattern"
        )
    )

    print("\nML Result:")

    print(
        result.get(
            "ml_result"
        )
    )

    print("\nML Score:")

    print(
        result.get(
            "ml_score"
        )
    )

    print("\nSignal Count:")

    print(
        result.get(
            "signal_count"
        )
    )

    print("\nFinal Monitoring Status:")

    print(
        result.get(
            "status"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("HEALTHSENSE POSTGRESQL ANALYSIS")
    print("========================================")

    connection = None

    try:

        # 1. Connect to PostgreSQL
        connection = get_connection()

        # 2. Load recent real BIDMC telemetry
        df = load_recent_bidmc_rows(
            connection
        )

        # 3. Show loaded data
        display_recent_data(
            df
        )

        # 4. Run existing HealthSense analysis
        result = run_analysis(
            df
        )

        # 5. Display result
        display_analysis_result(
            result
        )

        print("\n========================================")
        print("POSTGRESQL → HEALTH ANALYSIS SUCCESS")
        print("========================================")

    except Exception as error:

        print("\nERROR:")
        print(error)

        raise

    finally:

        if connection is not None:

            connection.close()

            print(
                "\nPostgreSQL connection closed."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()