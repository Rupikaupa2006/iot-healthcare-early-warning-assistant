# Vital Signs and Personalized Physiological Monitoring

## Purpose

This knowledge document describes the physiological measurements used by the
HealthSense AI monitoring prototype.

The prototype works with four measurements available in the BIDMC PPG and
Respiration Dataset:

- Heart Rate (HR)
- Pulse Rate (PULSE)
- Oxygen Saturation (SpO2)
- Respiratory Rate (RESP)

The system uses these measurements to identify changes relative to an
individual's historical observations.

The purpose of the system is monitoring and early-warning support. It does
not provide a medical diagnosis or replace evaluation by a qualified
healthcare professional.

---

## Heart Rate

Heart rate represents the number of heartbeats per minute.

In the BIDMC dataset, heart rate is represented by the `HR` measurement.

Heart rate can vary between observations because of physiological state,
activity, measurement conditions, and other factors.

The HealthSense AI prototype does not interpret a single heart-rate value
in isolation. Instead, it can compare the current value with the patient's
historical baseline.

A value that differs substantially from the individual's recent baseline
may be identified as a monitoring signal.

---

## Pulse Rate

Pulse rate represents the measured pulse frequency.

In the BIDMC dataset, pulse rate is represented by the `PULSE` measurement.

Pulse rate and heart rate can be similar but are treated as separate
measurements in the dataset and in the monitoring pipeline.

The system can compare pulse rate with the individual's historical values
to identify deviations from their personal baseline.

---

## Oxygen Saturation (SpO2)

SpO2 represents peripheral oxygen saturation measured using pulse
oximetry.

In the BIDMC dataset, oxygen saturation is represented by the `SpO2`
measurement.

Changes in SpO2 may provide useful information for physiological
monitoring. However, the HealthSense AI prototype does not use a single
SpO2 measurement to make a clinical diagnosis.

Instead, the system considers:

- the individual's historical baseline,
- the magnitude of deviation,
- recent measurements,
- temporal trends,
- relationships with other monitored measurements, and
- the machine-learning anomaly signal.

---

## Respiratory Rate

Respiratory rate represents the number of respiratory cycles per minute.

In the BIDMC dataset, respiratory rate is represented by the `RESP`
measurement.

Respiratory rate can change over time and may vary between individuals.

The HealthSense AI prototype compares the current respiratory-rate
measurement with the individual's historical observations.

A substantial deviation from the personal baseline can become a
monitoring signal and may require further review.

---

## Importance of Personal Baselines

A population-level reference value does not necessarily represent an
individual's normal physiological state.

For this reason, HealthSense AI calculates a personalized baseline from
previous observations in the selected recording.

The baseline contains:

- mean
- standard deviation
- number of historical observations

The current measurement can then be compared with the individual's
historical baseline.

This allows the system to identify changes that are unusual for that
specific recording rather than relying only on population-level values.

---

## Statistical Deviation

The prototype uses a z-score to measure how far a current measurement is
from its historical baseline.

The calculation is based on the current value, the historical mean, and
the historical standard deviation.

A larger absolute z-score indicates a larger statistical deviation from
the historical baseline.

The current prototype uses a configurable z-score threshold when
identifying deviations.

A statistical deviation should be interpreted as a monitoring signal,
not as proof of a medical condition.

---

## Interpreting Multiple Vital Signs

Physiological measurements can provide more information when considered
together rather than independently.

HealthSense AI therefore considers:

- heart rate,
- pulse rate,
- SpO2,
- respiratory rate, and
- their recent trends together.

For example, simultaneous deviations in multiple measurements may provide
a stronger monitoring signal than an isolated deviation in a single
measurement.

The system therefore performs multisensor analysis after calculating
individual measurement deviations.

---

## Temporal Trends

A single unusual measurement may not represent a persistent change.

HealthSense AI therefore considers recent observations when analyzing
physiological measurements.

Temporal analysis can help identify:

- repeated deviations,
- persistent changes,
- changes occurring across multiple measurements,
- short-term trends, and
- multisensor patterns.

This provides additional context for interpreting the current observation.

---

## Multisensor Patterns

A multisensor pattern occurs when multiple monitored measurements show
meaningful changes within a related time period.

The prototype can examine deviations across:

- heart rate,
- pulse rate,
- SpO2, and
- respiratory rate.

The purpose of multisensor analysis is to provide additional context to
the monitoring decision.

A detected multisensor pattern is not itself a diagnosis.

The system reports the pattern as an observation that may require human
review.

---

## Machine Learning Anomaly Detection

HealthSense AI also uses an Isolation Forest model as an additional
anomaly-detection signal.

The model is trained using physiological measurements from the BIDMC
dataset.

The current model uses these four features:

- heart rate
- pulse rate
- SpO2
- respiratory rate

The machine-learning model identifies observations that differ from
patterns learned from the training data.

The ML result is treated as one signal within the overall monitoring
pipeline.

An ML anomaly does not mean that a patient has a medical condition.

Similarly, an observation that is not flagged by the ML model does not
guarantee that the observation is clinically normal.

---

## Combining Monitoring Signals

HealthSense AI combines multiple forms of evidence:

1. Personalized baseline comparison
2. Statistical deviation analysis
3. Temporal analysis
4. Multisensor pattern analysis
5. Machine-learning anomaly detection
6. Knowledge-base evidence

These signals are used together to generate a monitoring status.

The current prototype uses three monitoring statuses:

### NORMAL

The available signals do not indicate a meaningful deviation requiring
additional monitoring attention.

### WATCH

One or more signals indicate that the current observation differs from
the individual's recent pattern and may benefit from continued monitoring.

### ATTENTION

Multiple monitoring signals indicate a stronger deviation or pattern that
should receive additional human review.

These statuses are monitoring-oriented labels and are not medical diagnoses.

---

## Human Review

Automated monitoring systems can identify patterns but cannot replace
clinical judgment.

A monitoring signal should therefore be reviewed in context.

The HealthSense AI prototype is designed to support a human-in-the-loop
workflow.

The system can provide:

- the current measurements,
- personalized baseline values,
- statistical deviations,
- temporal patterns,
- multisensor observations,
- machine-learning results, and
- supporting knowledge-base evidence.

This information allows a human reviewer to understand why the system
generated a particular monitoring signal.

---

## Evidence-Based Interpretation

The prototype uses a knowledge base to provide supporting information
for detected monitoring patterns.

The Retrieval-Augmented Generation (RAG) pipeline retrieves relevant
knowledge-base sections based on the detected pattern.

Retrieved information is used to provide evidence for the monitoring
interpretation.

The system should distinguish between:

- measured physiological values,
- calculated statistical signals,
- machine-learning outputs, and
- knowledge-based explanations.

This separation helps maintain transparency and makes the monitoring
decision easier to audit.

---

## Limitations

The BIDMC dataset is a research dataset and does not represent every
possible patient population or real-world monitoring environment.

The HealthSense AI prototype also has several limitations:

- It does not diagnose diseases.
- It does not prescribe treatment.
- It does not replace healthcare professionals.
- Statistical deviations do not necessarily indicate illness.
- Machine-learning anomalies do not necessarily indicate clinical
  abnormalities.
- The interpretation depends on the quality and characteristics of the
  available data.
- Personalized baselines depend on the amount and quality of historical
  observations.
- Missing or noisy measurements can affect monitoring results.

Therefore, system outputs should be interpreted as monitoring and
early-warning signals rather than definitive clinical conclusions.

---

## Dataset Alignment

The current implementation uses the BIDMC PPG and Respiration Dataset.

The physiological measurements used by the implementation are:

| Dataset Measurement | HealthSense AI Feature |
|---|---|
| HR | heart_rate |
| PULSE | pulse_rate |
| SpO2 | spo2 |
| RESP | respiratory_rate |

The implementation does not use body temperature or activity as BIDMC
physiological input features.

---

## Prototype Monitoring Pipeline

The current monitoring workflow is:

```text
BIDMC Physiological Data
        ↓
Data Preprocessing
        ↓
Personalized Baseline
        ↓
Statistical Deviation Analysis
        ↓
Temporal Analysis
        ↓
Multisensor Pattern Analysis
        ↓
Isolation Forest Anomaly Detection
        ↓
Knowledge Retrieval (RAG)
        ↓
Monitoring Interpretation
        ↓
Human Review