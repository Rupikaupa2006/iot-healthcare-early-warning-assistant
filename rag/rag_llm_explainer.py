import ollama

from rag.rag_anomaly_explainer import (
    retrieve_anomaly_evidence
)


# ============================================================
# NORMALIZE EVIDENCE
# ============================================================

def normalize_evidence(evidence):
    """
    Convert all supported evidence formats into one format:

    [
        {
            "source": "...",
            "text": "...",
            "distance": ...
        }
    ]

    Supported input formats:

    1. None

    2. A list of evidence dictionaries

    3. A dynamic RAG dictionary:

       {
           "query": "...",
           "evidence": [...]
       }

    This function prevents the common error:

        AttributeError:
        'str' object has no attribute 'get'
    """

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if evidence is None:
        return []

    # --------------------------------------------------------
    # Dynamic RAG dictionary
    # --------------------------------------------------------

    if isinstance(evidence, dict):

        # If this is the complete dynamic RAG result:
        #
        # {
        #     "query": "...",
        #     "evidence": [...]
        # }

        if "evidence" in evidence:

            evidence = evidence.get(
                "evidence",
                []
            )

        else:
            # A single evidence dictionary
            evidence = [evidence]

    # --------------------------------------------------------
    # Evidence should now be a list
    # --------------------------------------------------------

    if not isinstance(evidence, list):
        return []

    # --------------------------------------------------------
    # Keep only valid evidence dictionaries
    # --------------------------------------------------------

    cleaned_evidence = []

    for item in evidence:

        if isinstance(item, dict):

            cleaned_evidence.append(item)

    return cleaned_evidence


# ============================================================
# BUILD GROUNDED LLM PROMPT
# ============================================================

def build_llm_prompt(
    analysis,
    current_reading,
    evidence
):
    """
    Build a grounded prompt for Llama 3.2.
    """

    # --------------------------------------------------------
    # Normalize evidence AGAIN here.
    #
    # This makes the function safe even if someone calls
    # build_llm_prompt() directly in the future.
    # --------------------------------------------------------

    evidence = normalize_evidence(
        evidence
    )

    # ========================================================
    # CURRENT READING
    # ========================================================

    heart_rate = current_reading.get(
        "heart_rate",
        "N/A"
    )

    pulse_rate = current_reading.get(
        "pulse_rate",
        "N/A"
    )

    spo2 = current_reading.get(
        "spo2",
        "N/A"
    )

    respiratory_rate = current_reading.get(
        "respiratory_rate",
        "N/A"
    )

    # ========================================================
    # ANALYSIS
    # ========================================================

    status = analysis.get(
        "status",
        "UNKNOWN"
    )

    ml_result = analysis.get(
        "ml_result",
        "UNKNOWN"
    )

    ml_score = analysis.get(
        "ml_score",
        "N/A"
    )

    temporal_pattern = analysis.get(
        "temporal_pattern",
        "N/A"
    )

    deviations = analysis.get(
        "deviations",
        []
    )

    # ========================================================
    # FORMAT DEVIATIONS
    # ========================================================

    deviation_text = (
        "No significant baseline deviation detected."
    )

    if isinstance(
        deviations,
        list
    ) and deviations:

        deviation_lines = []

        for deviation in deviations:

            if not isinstance(
                deviation,
                dict
            ):
                continue

            sensor = deviation.get(
                "label",
                deviation.get(
                    "sensor",
                    "Unknown sensor"
                )
            )

            value = deviation.get(
                "value",
                "N/A"
            )

            baseline_mean = deviation.get(
                "baseline_mean",
                "N/A"
            )

            baseline_std = deviation.get(
                "baseline_std",
                "N/A"
            )

            z_score = deviation.get(
                "z_score",
                "N/A"
            )

            direction = deviation.get(
                "direction",
                "N/A"
            )

            deviation_lines.append(
                f"- {sensor}: "
                f"value={value}, "
                f"baseline_mean={baseline_mean}, "
                f"baseline_std={baseline_std}, "
                f"z_score={z_score}, "
                f"direction={direction}"
            )

        if deviation_lines:

            deviation_text = "\n".join(
                deviation_lines
            )

    # ========================================================
    # FORMAT RETRIEVED EVIDENCE
    # ========================================================

    if evidence:

        evidence_blocks = []

        for index, item in enumerate(
            evidence,
            start=1
        ):

            # This check is intentionally defensive.
            if not isinstance(
                item,
                dict
            ):
                continue

            source = item.get(
                "source",
                "Unknown source"
            )

            text = item.get(
                "text",
                ""
            )

            distance = item.get(
                "distance",
                "N/A"
            )

            evidence_blocks.append(
                f"Evidence {index}\n"
                f"Source: {source}\n"
                f"Distance: {distance}\n"
                f"Content:\n{text}"
            )

        if evidence_blocks:

            evidence_text = (
                "\n\n".join(
                    evidence_blocks
                )
            )

        else:

            evidence_text = (
                "No valid knowledge-base "
                "evidence was retrieved."
            )

    else:

        evidence_text = (
            "No additional knowledge-base "
            "evidence was retrieved."
        )

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    prompt = f"""
You are the AI explanation component of the
HealthSense physiological monitoring prototype.

The system is an engineering and research prototype
for monitoring and early-warning support.

Your task is to explain the current monitoring result
using ONLY the information provided below.

IMPORTANT SAFETY RULES:

- Do NOT diagnose a disease.
- Do NOT claim that the patient has a disease.
- Do NOT recommend medication.
- Do NOT recommend treatment.
- Do NOT invent physiological values.
- Do NOT invent trends.
- Do NOT invent evidence.
- Do NOT make claims unsupported by the provided data.
- Do NOT treat an ML anomaly signal as a medical diagnosis.

Use cautious monitoring language such as:

"The system detected..."

"The current reading differs from the recent
personalized baseline..."

"This may warrant continued monitoring..."

"The observation may require human review..."

If different analysis layers disagree, explicitly
describe the disagreement instead of forcing them
to agree.

============================================================
CURRENT PHYSIOLOGICAL MEASUREMENTS
============================================================

Heart rate: {heart_rate} bpm
Pulse rate: {pulse_rate} bpm
SpO2: {spo2}%
Respiratory rate: {respiratory_rate} breaths/min

============================================================
PERSONALIZED BASELINE ANALYSIS
============================================================

{deviation_text}

============================================================
TEMPORAL / MULTISENSOR ANALYSIS
============================================================

Temporal pattern:

{temporal_pattern}

============================================================
MACHINE-LEARNING ANALYSIS
============================================================

ML result:

{ml_result}

ML score:

{ml_score}

============================================================
PROTOTYPE MONITORING STATUS
============================================================

{status}

============================================================
RETRIEVED KNOWLEDGE-BASE EVIDENCE
============================================================

{evidence_text}

============================================================
EXPLANATION TASK
============================================================

Write a concise explanation in 1–3 short paragraphs.

Explain:

1. What the system detected relative to the
   patient's personalized baseline.

2. Whether a temporal or multisensor pattern
   was detected.

3. What the machine-learning model reported.

4. Whether the analysis layers agree or disagree.

5. Why the observation may be relevant for
   continued monitoring or human review.

The explanation must remain within the context
of physiological monitoring.

Do not diagnose.

Do not recommend treatment.

Do not recommend medication.

Do not invent information.

============================================================
END OF PROVIDED INFORMATION
============================================================
"""

    return prompt.strip()


# ============================================================
# GENERATE LLAMA EXPLANATION
# ============================================================

def generate_llm_explanation(
    analysis,
    current_reading,
    evidence=None
):
    """
    Generate a grounded explanation using
    Ollama + Llama 3.2.
    """

    # --------------------------------------------------------
    # If no evidence was supplied, retrieve it automatically.
    # --------------------------------------------------------

    if evidence is None:

        evidence = retrieve_anomaly_evidence(
            analysis,
            current_reading
        )

    # --------------------------------------------------------
    # NORMALIZE HERE
    #
    # This is the important fix.
    #
    # If evidence is:
    #
    # {
    #     "query": "...",
    #     "evidence": [...]
    # }
    #
    # it becomes:
    #
    # [...]
    # --------------------------------------------------------

    evidence_items = normalize_evidence(
        evidence
    )

    # --------------------------------------------------------
    # Build prompt using ONLY the evidence list
    # --------------------------------------------------------

    prompt = build_llm_prompt(
        analysis,
        current_reading,
        evidence_items
    )

    # --------------------------------------------------------
    # Call Ollama
    # --------------------------------------------------------

    print("\nConnecting to Ollama...")

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # --------------------------------------------------------
    # Extract response safely
    # --------------------------------------------------------

    if not response:

        explanation = (
            "No explanation was returned by "
            "the Llama model."
        )

    else:

        message = response.get(
            "message",
            {}
        )

        explanation = message.get(
            "content",
            ""
        ).strip()

        if not explanation:

            explanation = (
                "The Llama model returned an "
                "empty explanation."
            )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "explanation": explanation,
        "evidence": evidence_items,
        "prompt": prompt
    }


# ============================================================
# DIRECT MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("HEALTHSENSE — RAG + LLAMA 3.2 DIRECT TEST")
    print("=" * 70)

    example_analysis = {

        "status": "WATCH",

        "ml_result": "NORMAL",

        "ml_score": 0.10424,

        "temporal_pattern":
            "no_multisensor_pattern",

        "deviations": [
            {
                "sensor":
                    "respiratory_rate",

                "label":
                    "Respiratory Rate",

                "value":
                    20.0,

                "baseline_mean":
                    20.9474,

                "baseline_std":
                    0.2294,

                "z_score":
                    -4.1299,

                "direction":
                    "below baseline"
            }
        ]
    }

    example_reading = {

        "heart_rate":
            90.0,

        "pulse_rate":
            90.0,

        "spo2":
            96.0,

        "respiratory_rate":
            20.0
    }

    print(
        "\nGenerating Llama explanation..."
    )

    result = generate_llm_explanation(
        example_analysis,
        example_reading
    )

    print("\n" + "-" * 70)

    print(
        "LLAMA 3.2 EXPLANATION"
    )

    print("-" * 70)

    print(
        result["explanation"]
    )

    print("\n" + "=" * 70)

    print(
        "DIRECT TEST COMPLETED"
    )

    print("=" * 70)