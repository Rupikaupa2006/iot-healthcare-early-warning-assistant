# Physiological Monitoring Guidelines

## Purpose of the Monitoring System

The system is designed as a prototype for IoT-based physiological monitoring
and early warning.

It collects telemetry, identifies unusual patterns, retrieves relevant
knowledge, and generates an evidence-grounded explanation for human review.

It is not intended to replace a healthcare professional.

## Monitoring Instead of Diagnosis

An anomaly detected by the system means that the observed measurements differ
from the expected or learned pattern.

An anomaly does not automatically mean that a person has a particular
disease or medical condition.

The system should use terms such as:

- potential abnormal physiological pattern
- unusual sensor pattern
- deviation from personal baseline
- anomaly detected
- requires review

The system should not claim that it has diagnosed a disease.

## Importance of Context

Physiological measurements can be affected by:

- physical activity
- sleep
- stress
- environmental conditions
- sensor placement
- motion
- measurement quality
- individual physiological differences

Therefore, measurements should be interpreted using historical context,
activity information, and multiple available signals.

## Measurement Quality

Wearable and IoT sensors can produce noisy or inaccurate measurements.

Unexpected values should be considered together with other measurements and
recent trends.

The system should avoid treating one isolated sensor value as definitive.

## Personalized Monitoring

The prototype maintains a historical baseline for each monitored individual.

New observations can be compared with this baseline.

Personalized monitoring allows the system to identify changes that may be
unusual for that individual rather than relying exclusively on population-level
thresholds.

## Alert Explanation

Each alert should provide enough information for a reviewer to understand why
it was generated.

Relevant information may include:

- patient/device identifier
- timestamp
- current sensor values
- personal baseline
- deviation scores
- machine-learning anomaly result
- temporal pattern
- multisensor pattern
- retrieved knowledge
- evidence sources
- generated explanation

## Human-in-the-Loop

The final interpretation of an automatically detected pattern should remain
with an appropriately qualified human reviewer.

The system provides information and explanations to support review rather than
making autonomous clinical decisions.

## Auditability

Important events should be stored so that the system can reconstruct why an
alert was generated.

An audit record can connect the original telemetry with the anomaly-detection
result, retrieved evidence, and generated explanation.

This makes the prototype more transparent and easier to evaluate.