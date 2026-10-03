import json
import time
import pandas as pd
import paho.mqtt.client as mqtt

import os
from dotenv import load_dotenv
load_dotenv()


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "bidmc_clean.csv"
)


# ============================================================
# MQTT CONFIGURATION
# ============================================================
MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
MQTT_TOPIC = "health/bidmc/BIDMC_01"
if not MQTT_BROKER or not MQTT_USERNAME or not MQTT_PASSWORD:
    raise RuntimeError(
        "MQTT credentials are missing. Check your .env file."
    )

# ============================================================
# REPLAY SETTINGS
# ============================================================

PATIENT_ID = "BIDMC_01"

# Number of observations to replay
NUMBER_OF_OBSERVATIONS = 50

# Delay between observations
DELAY_SECONDS = 1


# ============================================================
# LOAD REAL BIDMC DATA
# ============================================================

print("Loading real BIDMC dataset...")

df = pd.read_csv(DATASET_PATH)

patient_data = df[
    df["patient_id"] == PATIENT_ID
].copy()

patient_data = patient_data.sort_values(
    "time"
).reset_index(drop=True)


if patient_data.empty:
    raise ValueError(
        f"No data found for {PATIENT_ID}"
    )


print(
    f"Loaded {len(patient_data)} observations "
    f"for {PATIENT_ID}"
)


# ============================================================
# MQTT CLIENT
# ============================================================

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.username_pw_set(
    MQTT_USERNAME,
    MQTT_PASSWORD
)

client.tls_set()


print("\nConnecting to HiveMQ...")

client.connect(
    MQTT_BROKER,
    MQTT_PORT,
    60
)

client.loop_start()

print("Connected to HiveMQ successfully.\n")


# ============================================================
# REPLAY REAL DATA
# ============================================================

replay_data = patient_data.head(
    NUMBER_OF_OBSERVATIONS
)

for _, row in replay_data.iterrows():

    payload = {
        "patient_id": row["patient_id"],
        "recording_id": row["recording_id"],
        "time": int(row["time"]),
        "heart_rate": float(row["heart_rate"]),
        "pulse_rate": float(row["pulse_rate"]),
        "spo2": float(row["spo2"]),
        "respiratory_rate": float(
            row["respiratory_rate"]
        )
    }

    message = json.dumps(payload)

    result = client.publish(
        MQTT_TOPIC,
        message
    )

    if result.rc == mqtt.MQTT_ERR_SUCCESS:

        print(
            f"Published observation "
            f"{payload['time']}s → "
            f"HR={payload['heart_rate']}, "
            f"Pulse={payload['pulse_rate']}, "
            f"SpO2={payload['spo2']}, "
            f"RR={payload['respiratory_rate']}"
        )

    else:

        print(
            "MQTT publish failed. "
            f"Return code: {result.rc}"
        )

    time.sleep(DELAY_SECONDS)


# ============================================================
# CLEANUP
# ============================================================

client.loop_stop()
client.disconnect()

print("\nBIDMC replay completed.")