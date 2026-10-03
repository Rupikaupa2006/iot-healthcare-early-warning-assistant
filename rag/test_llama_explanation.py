import sys
import os


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd

from rag.integrated_health_analysis import (
    analyze_with_evidence
)

from rag.rag_llm_explainer import (
    generate_llm_explanation
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "HEALTHSENSE — LLAMA 3.2 EXPLANATION TEST"
)

print("=" * 70)


# ============================================================
# LOAD REAL BIDMC DATA
# ============================================================

print(
    "\nLoading real BIDMC data..."
)

dataset_path = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "bidmc_clean.csv"
)

df = pd.read_csv(
    dataset_path
)


# ============================================================
# SELECT REAL PATIENT DATA
# ============================================================

patient_df = df[
    df["patient_id"] == "BIDMC_01"
].tail(20).copy()


if patient_df.empty:

    raise ValueError(
        "No BIDMC_01 observations were found."
    )


print(
    "Selected observations:",
    len(patient_df)
)


# ============================================================
# RUN INTEGRATED ANALYSIS
# ============================================================

print(
    "\nRunning integrated health analysis..."
)

result = analyze_with_evidence(
    patient_df
)


analysis = result["analysis"]

dynamic_rag = result["dynamic_rag"]


# ============================================================
# HEALTH ANALYSIS
# ============================================================

print("\n" + "-" * 70)

print(
    "HEALTH ANALYSIS"
)

print("-" * 70)


print(
    "Patient:",
    analysis.get(
        "patient_id",
        "N/A"
    )
)

print(
    "Recording:",
    analysis.get(
        "recording_id",
        "N/A"
    )
)

print(
    "Status:",
    analysis.get(
        "status",
        "N/A"
    )
)

print(
    "ML Result:",
    analysis.get(
        "ml_result",
        "N/A"
    )
)

print(
    "ML Score:",
    analysis.get(
        "ml_score",
        "N/A"
    )
)

print(
    "Temporal Pattern:",
    analysis.get(
        "temporal_pattern",
        "N/A"
    )
)


# ============================================================
# DEVIATIONS
# ============================================================

print(
    "\nDetected deviations:"
)

deviations = analysis.get(
    "deviations",
    []
)

if deviations:

    for deviation in deviations:

        print(
            "-",
            deviation
        )

else:

    print(
        "- No significant baseline deviation detected."
    )


# ============================================================
# DYNAMIC RAG
# ============================================================

print("\n" + "-" * 70)

print(
    "DYNAMIC RAG EVIDENCE"
)

print("-" * 70)


if isinstance(
    dynamic_rag,
    dict
):

    query = dynamic_rag.get(
        "query",
        ""
    )

    dynamic_evidence = (
        dynamic_rag.get(
            "evidence",
            []
        )
    )

else:

    query = ""

    dynamic_evidence = []


print("\nQuery:")

print(query)


print(
    "\nRetrieved evidence:"
)


for index, item in enumerate(
    dynamic_evidence,
    start=1
):

    print(
        f"\nEvidence {index}"
    )

    if not isinstance(
        item,
        dict
    ):

        print(
            "Invalid evidence item:",
            item
        )

        continue

    print(
        "Source:",
        item.get(
            "source",
            "Unknown"
        )
    )

    print(
        "Distance:",
        item.get(
            "distance",
            "N/A"
        )
    )

    print(
        "Text:",
        item.get(
            "text",
            ""
        )
    )


# ============================================================
# CURRENT READING
# ============================================================

current_reading = (
    patient_df
    .iloc[-1]
    .to_dict()
)


print(
    "\nCurrent reading:"
)

print(
    "Heart Rate:",
    current_reading.get(
        "heart_rate",
        "N/A"
    )
)

print(
    "Pulse Rate:",
    current_reading.get(
        "pulse_rate",
        "N/A"
    )
)

print(
    "SpO2:",
    current_reading.get(
        "spo2",
        "N/A"
    )
)

print(
    "Respiratory Rate:",
    current_reading.get(
        "respiratory_rate",
        "N/A"
    )
)


# ============================================================
# LLAMA
# ============================================================

print("\n" + "-" * 70)

print(
    "GENERATING LLAMA 3.2 EXPLANATION"
)

print("-" * 70)


llm_result = generate_llm_explanation(
    analysis,
    current_reading,
    evidence=dynamic_rag
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)

print(
    "LLAMA 3.2 EXPLANATION"
)

print("=" * 70)


print()

print(
    llm_result.get(
        "explanation",
        "No explanation generated."
    )
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)

print(
    "LLAMA 3.2 TEST COMPLETED"
)

print("=" * 70)