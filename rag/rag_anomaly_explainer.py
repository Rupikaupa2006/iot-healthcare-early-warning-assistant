import os

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ==========================================
# CONNECT TO CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_or_create_collection(
    name="healthcare_knowledge"
)

print("Connected to healthcare knowledge base.")


# ==========================================
# BUILD DYNAMIC ANOMALY QUERY
# ==========================================

def build_anomaly_query(
    analysis,
    current_reading
):
    """
    Build a contextual RAG query using
    the actual BIDMC analysis result and
    current physiological reading.
    """

    heart_rate = current_reading.get(
        "heart_rate"
    )

    pulse_rate = current_reading.get(
        "pulse_rate"
    )

    spo2 = current_reading.get(
        "spo2"
    )

    respiratory_rate = current_reading.get(
        "respiratory_rate"
    )

    status = analysis.get(
        "status",
        "UNKNOWN"
    )

    ml_result = analysis.get(
        "ml_result",
        "UNKNOWN"
    )

    ml_score = analysis.get(
        "ml_score"
    )

    temporal_pattern = analysis.get(
        "temporal_pattern",
        "UNKNOWN"
    )

    deviations = analysis.get(
        "deviations",
        []
    )


    # ======================================
    # FORMAT BASELINE DEVIATIONS
    # ======================================

    deviation_text = (
        "No significant baseline deviations detected."
    )

    if deviations:

        deviation_items = []

        for deviation in deviations:

            sensor = deviation.get(
                "sensor",
                "measurement"
            )

            label = deviation.get(
                "label",
                sensor
            )

            direction = deviation.get(
                "direction",
                "changed"
            )

            z_score = deviation.get(
                "z_score"
            )

            deviation_items.append(
                f"{label}: {direction}, "
                f"z-score={z_score}"
            )

        deviation_text = "; ".join(
            deviation_items
        )


    # ======================================
    # CREATE CONTEXTUAL QUERY
    # ======================================

    query = f"""
A patient is being monitored using
physiological measurements from the
BIDMC dataset.

Current physiological measurements:

Heart rate: {heart_rate} bpm
Pulse rate: {pulse_rate} bpm
SpO2: {spo2}%
Respiratory rate: {respiratory_rate} breaths/min

Personalized baseline analysis:

{deviation_text}

Temporal pattern:

{temporal_pattern}

Machine learning result:

{ml_result}

Machine learning score:

{ml_score}

Prototype monitoring status:

{status}

Retrieve knowledge explaining how these
physiological measurements should be
interpreted relative to:

1. The patient's personal baseline.
2. Recent temporal trends.
3. Multisensor physiological patterns.
4. Machine-learning anomaly signals.
5. The need for human review.

The explanation must remain within the
context of physiological monitoring.

Do not diagnose a disease.
Do not recommend treatment.
"""

    return query


# ==========================================
# RETRIEVE DYNAMIC EVIDENCE
# ==========================================

def retrieve_anomaly_evidence(
    analysis,
    current_reading,
    n_results=3
):
    """
    Retrieve evidence from ChromaDB based on
    the actual analysis result and current
    BIDMC physiological reading.
    """

    query = build_anomaly_query(
        analysis,
        current_reading
    )


    # ======================================
    # CREATE QUERY EMBEDDING
    # ======================================

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()


    # ======================================
    # SEARCH CHROMADB
    # ======================================

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )


    # ======================================
    # PREPARE RESULTS
    # ======================================

    evidence = []

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]


    for i, document in enumerate(
        documents
    ):

        if i < len(metadatas):

            source = metadatas[i].get(
                "source",
                "unknown"
            )

        else:

            source = "unknown"


        if i < len(distances):

            distance = distances[i]

        else:

            distance = None


        evidence.append(
            {
                "source": source,
                "text": document,
                "distance": distance
            }
        )


    return {
        "query": query,
        "evidence": evidence
    }


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("DYNAMIC RAG ANOMALY EXPLAINER")
    print("=" * 60)

    print(
        "\nModule loaded successfully."
    )

    print(
        "This module now uses real analysis "
        "results instead of hardcoded readings."
    )

    print(
        "\nNo synthetic temperature or "
        "activity values are used."
    )

    print("\nDynamic RAG module ready.")