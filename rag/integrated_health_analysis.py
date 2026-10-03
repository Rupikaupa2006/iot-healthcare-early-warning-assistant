import os
import sys

import pandas as pd


# ==========================================
# PROJECT ROOT
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ==========================================
# IMPORTS
# ==========================================

from anomaly.health_analysis import (
    analyze_bidmc_rows
)

from rag.rag_pipeline import (
    get_evidence
)

from rag.rag_anomaly_explainer import (
    retrieve_anomaly_evidence
)


# ==========================================
# SIGNAL-SPECIFIC QUERIES
# ==========================================

QUERIES = {

    "baseline": (
        "How should physiological measurements be "
        "compared with an individual's personal baseline?"
    ),

    "deviation": (
        "How should a physiological measurement that "
        "deviates from an individual's personal baseline "
        "be interpreted?"
    ),

    "multisensor": (
        "How should simultaneous deviations in heart rate, "
        "pulse rate, SpO2 and respiratory rate be interpreted?"
    ),

    "machine_learning": (
        "How should an Isolation Forest machine learning "
        "anomaly detection result be interpreted in "
        "physiological monitoring?"
    ),

    "human_review": (
        "Why should an unusual physiological pattern "
        "be reviewed by a human rather than treated "
        "as a medical diagnosis?"
    ),

    "respiratory_rate": (
        "How should a respiratory rate measurement that "
        "differs from a person's historical baseline "
        "be interpreted?"
    ),

    "heart_rate": (
        "How should a heart rate measurement that differs "
        "from a person's historical baseline be interpreted?"
    ),

    "pulse_rate": (
        "How should a pulse rate measurement that differs "
        "from a person's historical baseline be interpreted?"
    ),

    "spo2": (
        "How should an SpO2 measurement that differs "
        "from a person's historical baseline be interpreted?"
    )
}


# ==========================================
# DETERMINE RELEVANT SIGNALS
# ==========================================

def determine_signals(result):

    signals = []

    # Baseline is always relevant
    signals.append("baseline")


    # ======================================
    # INDIVIDUAL DEVIATIONS
    # ======================================

    deviations = result.get(
        "deviations",
        []
    )

    for deviation in deviations:

        sensor = deviation.get(
            "sensor"
        )

        if sensor == "heart_rate":

            signals.append(
                "heart_rate"
            )

        elif sensor == "pulse_rate":

            signals.append(
                "pulse_rate"
            )

        elif sensor == "spo2":

            signals.append(
                "spo2"
            )

        elif sensor == "respiratory_rate":

            signals.append(
                "respiratory_rate"
            )

        signals.append(
            "deviation"
        )


    # ======================================
    # MULTISENSOR / TEMPORAL PATTERN
    # ======================================

    if result.get(
        "temporal_pattern"
    ) != "no_multisensor_pattern":

        signals.append(
            "multisensor"
        )


    # ======================================
    # MACHINE LEARNING SIGNAL
    # ======================================

    if result.get(
        "ml_result"
    ):

        signals.append(
            "machine_learning"
        )


    # ======================================
    # HUMAN REVIEW
    # ======================================

    if result.get(
        "status"
    ) in (
        "WATCH",
        "ATTENTION"
    ):

        signals.append(
            "human_review"
        )


    # ======================================
    # REMOVE DUPLICATES
    # ======================================

    return list(
        dict.fromkeys(
            signals
        )
    )


# ==========================================
# RETRIEVE SIGNAL-SPECIFIC EVIDENCE
# ==========================================

def retrieve_relevant_evidence(
    result
):

    signals = determine_signals(
        result
    )

    evidence = []


    for signal in signals:

        query = QUERIES.get(
            signal
        )

        if not query:
            continue

        matches = get_evidence(
            query,
            n_results=2
        )

        evidence.append(
            {
                "signal": signal,
                "query": query,
                "results": matches
            }
        )


    return evidence


# ==========================================
# INTEGRATED ANALYSIS
# ==========================================

def analyze_with_evidence(
    rows
):
    """
    Complete analysis pipeline:

    1. Analyze physiological readings.
    2. Retrieve signal-specific RAG evidence.
    3. Retrieve dynamic contextual RAG evidence.
    """

    # ======================================
    # HEALTH ANALYSIS
    # ======================================

    analysis = analyze_bidmc_rows(
        rows
    )


    # ======================================
    # SIGNAL-SPECIFIC RAG
    # ======================================

    evidence = retrieve_relevant_evidence(
        analysis
    )


    # ======================================
    # GET CURRENT READING
    # ======================================

    if isinstance(
        rows,
        pd.DataFrame
    ):

        if rows.empty:
            raise ValueError(
                "Cannot analyze an empty DataFrame."
            )

        current_reading = (
            rows.iloc[-1]
            .to_dict()
        )

    else:

        if not rows:
            raise ValueError(
                "Cannot analyze empty rows."
            )

        current_reading = rows[-1]


    # ======================================
    # DYNAMIC CONTEXTUAL RAG
    # ======================================

    dynamic_rag = retrieve_anomaly_evidence(
        analysis,
        current_reading,
        n_results=3
    )


    # ======================================
    # RETURN COMPLETE RESULT
    # ======================================

    return {

        "analysis": analysis,

        "evidence": evidence,

        "dynamic_rag": dynamic_rag
    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print(
        "Loading real BIDMC data..."
    )


    dataset_path = os.path.join(
        BASE_DIR,
        "dataset",
        "bidmc_clean.csv"
    )


    df = pd.read_csv(
        dataset_path
    )


    # ======================================
    # SELECT REAL BIDMC RECORDING
    # ======================================

    patient_data = df[
        df["patient_id"] == "BIDMC_01"
    ].tail(20)


    if patient_data.empty:

        raise ValueError(
            "No BIDMC_01 observations found."
        )


    print(
        "Selected observations:",
        len(patient_data)
    )


    print(
        "\nRunning integrated "
        "health analysis..."
    )


    result = analyze_with_evidence(
        patient_data
    )


    analysis = result[
        "analysis"
    ]


    # ======================================
    # HEALTH ANALYSIS
    # ======================================

    print("\n" + "=" * 60)

    print(
        "HEALTH ANALYSIS"
    )

    print("=" * 60)


    print(
        "Patient:",
        analysis.get(
            "patient_id"
        )
    )


    print(
        "Recording:",
        analysis.get(
            "recording_id"
        )
    )


    print(
        "Status:",
        analysis.get(
            "status"
        )
    )


    print(
        "ML result:",
        analysis.get(
            "ml_result"
        )
    )


    print(
        "ML score:",
        analysis.get(
            "ml_score"
        )
    )


    print(
        "Temporal pattern:",
        analysis.get(
            "temporal_pattern"
        )
    )


    # ======================================
    # DETECTED DEVIATIONS
    # ======================================

    print(
        "\nDetected deviations:"
    )


    deviations = analysis.get(
        "deviations",
        []
    )


    if not deviations:

        print(
            "- No significant deviations detected."
        )

    else:

        for deviation in deviations:

            print(
                "-",
                deviation.get(
                    "label",
                    deviation.get(
                        "sensor",
                        "measurement"
                    )
                ),
                ":",
                deviation.get(
                    "direction",
                    "changed"
                ),
                "(z=",
                deviation.get(
                    "z_score"
                ),
                ")"
            )


    # ======================================
    # SIGNAL-SPECIFIC RAG EVIDENCE
    # ======================================

    print(
        "\n" + "=" * 60
    )

    print(
        "SIGNAL-SPECIFIC RAG EVIDENCE"
    )

    print(
        "=" * 60
    )


    for item in result[
        "evidence"
    ]:

        print(
            "\nSIGNAL:",
            item["signal"]
        )


        print(
            "QUERY:",
            item["query"]
        )


        for evidence_item in item[
            "results"
        ]:

            print(
                "\nSOURCE:",
                evidence_item[
                    "source"
                ]
            )


            print(
                evidence_item[
                    "text"
                ]
            )


            print(
                "-" * 50
            )


    # ======================================
    # DYNAMIC CONTEXTUAL RAG
    # ======================================

    dynamic_rag = result[
        "dynamic_rag"
    ]


    print(
        "\n" + "=" * 60
    )

    print(
        "DYNAMIC CONTEXTUAL RAG"
    )

    print(
        "=" * 60
    )


    print(
        "\nGenerated Query:"
    )

    print(
        dynamic_rag[
            "query"
        ]
    )


    print(
        "\nRetrieved Evidence:"
    )


    for i, evidence_item in enumerate(
        dynamic_rag[
            "evidence"
        ],
        start=1
    ):

        print(
            f"\nEVIDENCE {i}"
        )


        print(
            "Source:",
            evidence_item[
                "source"
            ]
        )


        print(
            "Distance:",
            evidence_item[
                "distance"
            ]
        )


        print(
            "\nKnowledge:"
        )


        print(
            evidence_item[
                "text"
            ]
        )


        print(
            "-" * 50
        )


    print(
        "\n" + "=" * 60
    )

    print(
        "INTEGRATED ANALYSIS + "
        "DYNAMIC RAG TEST COMPLETED"
    )

    print(
        "=" * 60
    )