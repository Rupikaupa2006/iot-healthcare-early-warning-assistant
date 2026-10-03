# Multisensor Physiological Monitoring Patterns

## Purpose

This document provides general knowledge for interpreting combinations
of physiological measurements used by the HealthSense AI monitoring
prototype.

The prototype uses:

- Heart Rate
- Pulse Rate
- SpO2
- Respiratory Rate

The purpose of this knowledge base is to support monitoring explanations
and evidence retrieval. It is not intended to provide clinical diagnosis
or treatment recommendations.

---

## Multisensor Pattern

A multisensor pattern occurs when more than one physiological measurement
changes from an individual's recent historical baseline.

For example, a recent sequence may show:

- heart rate changing from the individual's recent baseline,
- pulse rate changing from the individual's recent baseline,
- SpO2 changing from the individual's recent baseline,
- respiratory rate changing from the individual's recent baseline.

The monitoring system can identify these changes and determine whether
multiple measurements are deviating at approximately the same time.

---

## Personalized Baseline

A physiological value should be interpreted relative to the individual's
historical observations when a personalized baseline is available.

The HealthSense AI prototype calculates a baseline from previous
observations in the selected recording.

The baseline contains:

- mean
- standard deviation
- number of historical observations

A current measurement can then be compared with this baseline using
a statistical z-score.

---

## Statistical Deviation

A large absolute z-score indicates that the current observation is
statistically distant from the selected historical baseline.

The prototype uses this statistical signal as one component of its
monitoring analysis.

A statistical deviation does not by itself indicate a medical condition.

---

## Temporal Pattern

A temporal pattern considers whether measurements continue to differ
across a sequence of observations rather than looking at only one
measurement.

The system therefore considers recent observations when evaluating
physiological changes.

This helps distinguish an isolated measurement from a change that
persists across multiple observations.

---

## Multisensor Interpretation

When multiple physiological measurements deviate from an individual's
baseline within a recent observation window, the system may identify
a potential multisensor physiological pattern.

This should be interpreted using:

1. The individual's historical baseline
2. The direction and magnitude of deviations
3. Recent temporal trends
4. The number of physiological measurements involved
5. The machine-learning anomaly signal
6. Measurement quality and available context

The system should describe such a finding as a potential abnormal
physiological pattern requiring review rather than as a diagnosis.

---

## Machine-Learning Signal

The HealthSense AI prototype also uses an Isolation Forest model as
a supporting anomaly-detection signal.

The machine-learning result should not be interpreted independently.

For example:

- Statistical analysis may identify a deviation while the ML model
  does not flag the observation.
- The ML model may flag an unusual observation while the personalized
  statistical analysis shows limited deviation.
- Both signals may indicate unusual behavior.

These situations should be presented transparently rather than forcing
the signals to agree.

---

## Monitoring Decision

The final monitoring status combines multiple signals:

1. Personalized statistical deviations
2. Temporal analysis
3. Multisensor pattern analysis
4. Isolation Forest anomaly detection

The result represents a monitoring and decision-support signal.

It is not a clinical diagnosis.