import random
from datetime import datetime


def generate_sensor_data():

    # Normal patient data
    data = {
        "device_id": "PATIENT_001",
        "timestamp": datetime.now().isoformat(),
        "heart_rate": random.randint(70, 90),
        "spo2": random.randint(96, 100),
        "body_temperature": round(random.uniform(36.5, 37.2), 1),
        "respiratory_rate": random.randint(14, 20),
        "activity": random.choice(["resting", "walking", "sleeping"])
    }

    # -----------------------------------
    # Synthetic anomaly for testing
    # -----------------------------------

    if random.random() < 0.20:

        print("\n⚠️ SYNTHETIC ANOMALY GENERATED FOR TESTING")

        data["heart_rate"] = random.randint(110, 125)
        data["spo2"] = random.randint(90, 93)
        data["body_temperature"] = round(random.uniform(38.0, 38.8), 1)
        data["respiratory_rate"] = random.randint(23, 28)
        data["activity"] = "resting"

    return data