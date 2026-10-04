# Knowledge-Driven, Personalized IoT Healthcare Monitoring & Early-Warning Assistant

> **Engineering / Research Prototype**  
> A knowledge-driven healthcare monitoring and early-warning system that combines real physiological data, IoT communication, personalized baseline analysis, machine learning, temporal and multisensor analysis, Retrieval-Augmented Generation (RAG), and a local Large Language Model (LLM).

> **Medical Disclaimer:** This project is an academic engineering/research prototype. It is **not a medical diagnostic system**, does not provide medical diagnoses or treatment recommendations, and should not be used for clinical decision-making.

## Overview

Healthcare monitoring systems often collect multiple physiological measurements but may provide limited context about whether a particular observation differs significantly from an individual's previous measurements.

This project explores a **personalized, knowledge-driven monitoring pipeline** that processes real physiological observations and combines multiple analytical layers:

- Personalized baseline analysis
- Temporal trend analysis
- Multisensor pattern analysis
- Machine-learning-based anomaly detection
- IoT communication using MQTT
- Real-time data orchestration using Node-RED
- PostgreSQL-based telemetry storage
- Retrieval-Augmented Generation (RAG)
- Local LLM-based explanations
- Evidence retrieval from a domain knowledge base
- Dynamic Streamlit visualization
- Human-in-the-loop monitoring

The system is designed to identify **potentially unusual physiological patterns** and provide an evidence-grounded explanation for why an observation may require further review.

## Key Features

### 1. Real Healthcare Dataset

The system uses the **PhysioNet BIDMC PPG and Respiration Dataset v1.0.0** rather than relying entirely on synthetic data.

The dataset contains:

- **53 recordings**
- **25,493 physiological observations**
- **481 observations per recording**
- Heart rate
- Pulse rate
- Respiratory rate
- SpO2

Dataset source:

https://physionet.org/content/bidmc/1.0.0/

### 2. Personalized Baseline Analysis

Instead of relying only on fixed generic thresholds, the system establishes a baseline from the **available historical observations within a recording**.

The baseline is calculated for:

- Heart rate
- Pulse rate
- SpO2
- Respiratory rate

The current observation is compared against the calculated baseline using statistical deviation and z-score analysis.

This allows the system to identify observations that differ substantially from the observed baseline.

### 3. Temporal Analysis

Physiological measurements are analyzed over time rather than treating every observation as an isolated point.

The temporal analysis considers:

- Recent observations
- Changes from previous observations
- Direction of change
- Persistence of deviations
- Recent physiological patterns

This helps distinguish a temporary change from a potentially persistent monitoring signal.

### 4. Multisensor Pattern Analysis

The system analyzes multiple physiological measurements together.

The monitored variables are:

- Heart rate
- Pulse rate
- SpO2
- Respiratory rate

Instead of evaluating every sensor independently, the system also checks whether multiple measurements show related changes within the same observation window.

This provides additional context for the monitoring decision.

### 5. Machine Learning Anomaly Detection

The project uses an **Isolation Forest** model for unsupervised anomaly detection.

Model configuration:

- Algorithm: Isolation Forest
- Number of estimators: 200
- Random state: 42
- Contamination: `auto`
- Features:
  - Heart rate
  - Pulse rate
  - SpO2
  - Respiratory rate

The model is trained using a **temporal split**:

- Training observations: **17,808**
- Testing observations: **7,685**

The model is trained on the earlier portion of each recording and evaluated on later unseen observations.

> **Important:** The percentage of observations flagged by the Isolation Forest is an **ML anomaly signal**, not a clinical abnormality rate and not a percentage of patients with a medical condition.

## IoT Data Pipeline

The project integrates real physiological observations with an IoT-style data pipeline.

    PhysioNet BIDMC Dataset
              ↓
        Preprocessing
              ↓
        Real-data Replay
              ↓
             MQTT
              ↓
         HiveMQ Cloud
              ↓
           Node-RED
              ↓
          PostgreSQL
              ↓
     Monitoring Analysis

The same physiological observations are processed through an IoT communication pipeline before reaching the monitoring and explanation layers.

### MQTT Communication

The project uses **MQTT** for lightweight telemetry communication.

The BIDMC observations are replayed as MQTT messages containing:

    patient_id
    recording_id
    time
    heart_rate
    pulse_rate
    spo2
    respiratory_rate

Example topic structure:

    health/bidmc/BIDMC_01

MQTT allows the system to simulate the continuous transmission of physiological telemetry from a monitoring device.

### HiveMQ Cloud

**HiveMQ Cloud** is used as the MQTT broker.

The architecture is:

    BIDMC Replay Script
            ↓
       HiveMQ Cloud
            ↓
    Node-RED MQTT Input

The replay script publishes real BIDMC observations to the configured HiveMQ Cloud MQTT endpoint.

Credentials are loaded through environment variables and are **not stored directly in the source code**.

### Node-RED

Node-RED is used as the orchestration layer between MQTT and PostgreSQL.

Current pipeline:

    BIDMC Telemetry
          ↓
    Prepare BIDMC PostgreSQL
          ↓
       PostgreSQL
          ↓
         Debug

Node-RED receives the MQTT telemetry, prepares the incoming data, and stores the observations in PostgreSQL.

### PostgreSQL

PostgreSQL is used as the persistent telemetry database.

The main table used by the project is:

    bidmc_telemetry

Stored fields include:

    patient_id
    recording_id
    time_seconds
    heart_rate
    pulse_rate
    spo2
    respiratory_rate

This database provides the monitoring engine with a persistent source of incoming physiological observations.

## Monitoring Analysis Engine

The monitoring engine combines multiple analytical signals instead of relying on a single threshold.

The analysis pipeline includes:

    Incoming Observation
            ↓
    Personalized Baseline
            ↓
    Deviation Analysis
            ↓
    Temporal Analysis
            ↓
    Multisensor Pattern Analysis
            ↓
    Isolation Forest
            ↓
    Combined Monitoring Status

The monitoring status can indicate:

- `NORMAL`
- `WATCH`
- `ATTENTION`

These statuses represent **prototype monitoring signals** and should not be interpreted as medical diagnoses.

### Personalized Deviation Analysis

The system calculates how far the current physiological observation differs from the calculated baseline.

For each monitored variable, the analysis considers:

- Baseline mean
- Baseline standard deviation
- Current value
- Statistical deviation
- Z-score

A deviation may contribute to a monitoring signal when it is sufficiently different from the observed baseline.

### Temporal Analysis

The temporal layer examines the recent sequence of observations.

It helps identify:

- Sudden changes
- Persistent deviations
- Recent trends
- Repeated unusual observations

This is important because a single unusual observation may have different significance from a pattern that persists over multiple observations.

### Multisensor Analysis

The system also evaluates the physiological variables together.

For example, it can determine whether multiple monitored variables change in a related observation window.

The purpose is not to diagnose a condition, but to provide additional context to the monitoring signal.

## Retrieval-Augmented Generation (RAG)

The project uses Retrieval-Augmented Generation to provide evidence-based explanations.

The RAG pipeline is:

    Knowledge Base
          ↓
    Text Chunking
          ↓
    Sentence Transformer Embeddings
          ↓
    ChromaDB
          ↓
    Similarity Retrieval
          ↓
    Relevant Evidence
          ↓
    LLM Explanation

The knowledge base contains Markdown documents covering:

- Vital-sign concepts
- Multisensor monitoring patterns
- Monitoring guidelines

### Embedding Model

The project uses:

`all-MiniLM-L6-v2`

from Sentence Transformers.

The model converts knowledge-base text into embeddings that can be searched for semantically relevant information.

### ChromaDB

ChromaDB is used as the vector database for storing and retrieving knowledge-base embeddings.

The RAG pipeline retrieves the most relevant knowledge chunks based on the current monitoring context.

## Local LLM Explanation

The project uses a locally hosted LLM through **Ollama**.

Current model:

`llama3.2:3b`

The LLM receives:

- Current physiological observations
- Personalized baseline analysis
- Temporal analysis
- Multisensor analysis
- Isolation Forest result
- Retrieved knowledge-base evidence

It then generates a grounded monitoring explanation.

### Evidence-Based Explanation

The explanation layer is designed to distinguish between different analytical signals.

For example:

> The personalized baseline analysis identified a deviation from the observed baseline, while the machine-learning anomaly detector did not independently flag the observation.

This distinction is important because the statistical baseline and the machine-learning model represent different analytical perspectives.

The LLM is instructed to use the retrieved evidence rather than inventing unsupported medical conclusions.

## Dynamic Streamlit Dashboard

The project includes a dynamic dashboard built with **Streamlit**.

The dashboard is designed to continuously update as new observations are replayed through the IoT pipeline.

### Dashboard Components

- Real-time physiological values
- Heart rate
- Pulse rate
- SpO2
- Respiratory rate
- Dynamic rolling graph
- Physiological fingerprint
- Personalized baseline
- Personalized deviation
- AI monitoring signals
- RAG evidence
- Local LLM explanation
- Monitoring decision pipeline
- Human-in-the-loop section
- Dataset and architecture information

The dashboard automatically changes as new observations enter the pipeline rather than depending on manual slider interaction.

## System Architecture

    ┌─────────────────────────────┐
    │   PhysioNet BIDMC Dataset   │
    │       53 Recordings         │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │     Data Preprocessing       │
    │   Cleaning + Normalization  │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │       Real Data Replay       │
    │        MQTT Publisher        │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │         HiveMQ Cloud         │
    │         MQTT Broker          │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │          Node-RED            │
    │      IoT Orchestration       │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │         PostgreSQL           │
    │      Telemetry Storage       │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌────────────────────────────────────────┐
    │        Monitoring Analysis Engine      │
    │                                        │
    │  Personalized Baseline                 │
    │  Temporal Analysis                     │
    │  Multisensor Analysis                  │
    │  Isolation Forest                      │
    └────────────────────┬───────────────────┘
                         │
                         ▼
    ┌─────────────────────────────┐
    │      RAG Knowledge Layer     │
    │  Sentence Transformers       │
    │  ChromaDB                    │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │            Ollama             │
    │         Llama 3.2:3b         │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │      Streamlit Dashboard     │
    │       Dynamic Monitoring      │
    └─────────────────────────────┘

## Repository Structure

    CSE24_939_FAULT DIAGNOSIS ASSISTANT/
    │
    ├── anomaly/
    │   ├── bidmc_isolation_forest.joblib
    │   ├── health_analysis.py
    │   ├── postgres_analysis.py
    │   └── train_bidmc_model.py
    │
    ├── dataset/
    │   ├── bidmc_clean.csv
    │   ├── inspect_bidmc.py
    │   └── preprocess_bidmc.py
    │
    ├── knowledge_base/
    │   ├── monitoring_guidelines.md
    │   ├── multisensor_patterns.md
    │   └── vital_signs.md
    │
    ├── mqtt/
    │   └── bidmc_replay.py
    │
    ├── rag/
    │   ├── integrated_health_analysis.py
    │   ├── rag_anomaly_explainer.py
    │   ├── rag_llm_explainer.py
    │   ├── rag_pipeline.py
    │   └── test_llama_explanation.py
    │
    ├── simulator/
    │   └── sensor_simulator.py
    │
    ├── .env.example
    ├── .gitignore
    ├── dashboard.py
    ├── requirements.txt
    └── README.md

>> The raw BIDMC dataset, `.env` file, and local vector database are excluded from version control. The cleaned dataset and trained Isolation Forest model used by the deployed dashboard are included in the repository.
## Installation

### 1. Clone the Repository

    git clone https://github.com/Rupikaupa2006/iot-healthcare-early-warning-assistant.git
    cd iot-healthcare-early-warning-assistant
    
### 2. Create a Virtual Environment

On Windows:

    py -m venv .venv

Activate it:

    .venv\Scripts\activate

### 3. Install Dependencies

    py -m pip install -r requirements.txt

## Required Services

The project uses the following components:

| Component | Purpose |
|---|---|
| Python | Data processing, ML and backend analysis |
| PostgreSQL | Telemetry storage |
| Node-RED | IoT orchestration |
| HiveMQ Cloud | MQTT broker |
| Ollama | Local LLM execution |
| Streamlit | Dynamic dashboard |
| ChromaDB | Vector storage for RAG |

## Environment Configuration

Create a local `.env` file using `.env.example` as the template.

The environment file should contain:

    # HiveMQ Cloud
    MQTT_BROKER=your_hivemq_broker
    MQTT_PORT=8883
    MQTT_USERNAME=your_hivemq_username
    MQTT_PASSWORD=your_hivemq_password

    # PostgreSQL
    POSTGRES_HOST=127.0.0.1
    POSTGRES_PORT=5432
    POSTGRES_DB=healthcare_iot
    POSTGRES_USER=postgres
    POSTGRES_PASSWORD=your_postgresql_password

    # Ollama
    OLLAMA_MODEL=llama3.2:3b

> Never commit `.env` to GitHub. Credentials should always remain local.

## Dataset Setup

Download the **PhysioNet BIDMC PPG and Respiration Dataset v1.0.0** from:

https://physionet.org/content/bidmc/1.0.0/

After extraction, place the CSV directory inside:

    dataset/bidmc_csv/

The directory should contain files similar to:

    bidmc_01_Breaths.csv
    bidmc_01_Fix.txt
    bidmc_01_Numerics.csv
    bidmc_01_Signals.csv

    bidmc_02_Breaths.csv
    bidmc_02_Fix.txt
    bidmc_02_Numerics.csv
    bidmc_02_Signals.csv

    ...

The raw dataset is intentionally ignored by Git because of its size.

## Dataset Inspection

To inspect the downloaded BIDMC files:

    py dataset/inspect_bidmc.py

The preprocessing pipeline can then be executed using:

    py dataset/preprocess_bidmc.py

This produces:

    dataset/bidmc_clean.csv

The cleaned dataset contains:

    patient_id
    recording_id
    time
    heart_rate
    pulse_rate
    spo2
    respiratory_rate

The preprocessing pipeline handles missing physiological observations through interpolation and forward/backward filling so that the downstream analysis receives complete numerical inputs.

## Train the Machine Learning Model

Train the Isolation Forest model using:

    py anomaly/train_bidmc_model.py

The trained model is saved locally as:

    anomaly/bidmc_isolation_forest.joblib

The model can be regenerated at any time using:

    py anomaly/train_bidmc_model.py

The model uses a temporal split so that later observations are not used during training.

## Running the System

The complete system consists of several components.

### 1. Start PostgreSQL

Make sure the PostgreSQL service is running.

On Windows:

    Get-Service postgresql-x64-18

If required:

    Start-Service postgresql-x64-18

### 2. Start Node-RED

Start Node-RED using:

    node-red

Then open:

    http://127.0.0.1:1880/

Use the configured MQTT-to-PostgreSQL flow.

### 3. Start Ollama

Make sure Ollama is running locally and the required model is available.

Check the installed models using:

    ollama list

The project currently uses:

    llama3.2:3b

### 4. Start the BIDMC Replay

Run:

    py mqtt/bidmc_replay.py

The script reads real BIDMC observations and publishes them through MQTT.

### 5. Start the Streamlit Dashboard

Run:

    py -m streamlit run dashboard.py

Then open:

    http://localhost:8501

The dashboard continuously updates as new observations move through the pipeline.

## Monitoring Decision Pipeline

The monitoring decision is generated using multiple analytical layers.

    Incoming BIDMC Observation
                ↓
    Personalized Baseline
                ↓
    Deviation Detection
                ↓
    Temporal Analysis
                ↓
    Multisensor Analysis
                ↓
    Isolation Forest
                ↓
    Combined Monitoring Signal
                ↓
    RAG Evidence Retrieval
                ↓
    Local LLM Explanation
                ↓
    Human Review

The system is intentionally designed as a **decision-support and monitoring prototype**, with the final interpretation remaining subject to human review.

## Dataset Results

After preprocessing the BIDMC dataset:

| Metric | Result |
|---|---:|
| Total recordings | 53 |
| Total observations | 25,493 |
| Observations per recording | 481 |

The cleaned dataset contains:

- Heart rate
- Pulse rate
- SpO2
- Respiratory rate

## ML Validation Results

The Isolation Forest model was trained using:

| Metric | Result |
|---|---:|
| Training observations | 17,808 |
| Testing observations | 7,685 |

The model generated anomaly predictions for the unseen temporal test observations.

The anomaly score is used as an **ML monitoring signal** and is combined with the personalized baseline and other analysis layers.

> The model's anomaly predictions should not be interpreted as medical diagnoses, disease detection, or a clinical abnormality percentage.

## Testing

The project includes testing at multiple stages.

### Dataset Inspection

    py dataset/inspect_bidmc.py

### Dataset Preprocessing

    py dataset/preprocess_bidmc.py

### ML Training

    py anomaly/train_bidmc_model.py

### PostgreSQL Analysis

    py anomaly/postgres_analysis.py

### RAG Pipeline

    py rag/rag_pipeline.py

### LLM Explanation

    py rag/test_llama_explanation.py

### Streamlit Dashboard

    py -m streamlit run dashboard.py

## Limitations

The current prototype has several limitations:

1. The BIDMC dataset contains physiological recordings rather than continuous real-world longitudinal monitoring of an individual.

2. The dataset does not contain every physiological variable that may be available in a real healthcare monitoring environment.

3. The Isolation Forest model is unsupervised and does not use clinically validated diagnostic labels.

4. An ML anomaly prediction does not indicate the presence or absence of a medical condition.

5. Personalized baseline analysis is based on the available observations within the dataset recording.

6. The RAG knowledge base is a limited prototype knowledge source and should not be treated as a complete medical knowledge system.

7. The local LLM generates explanations based on the supplied monitoring context and retrieved evidence but is not a medical expert system.

8. The project has not been clinically validated.

9. The MQTT replay simulates continuous telemetry using historical physiological observations rather than receiving measurements directly from physical medical sensors.

## Future Scope

Possible future improvements include:

- Integration with physical wearable sensors
- ESP32-based physiological data acquisition
- Additional physiological variables
- Larger and more diverse healthcare datasets
- More advanced time-series models
- Personalized adaptive anomaly detection
- Improved multisensor fusion
- Model explainability techniques
- Automated monitoring reports
- Better uncertainty estimation
- Expanded medical knowledge bases
- Federated or privacy-preserving learning
- Secure healthcare data transmission
- Role-based dashboards
- Human-in-the-loop alert management
- Clinical validation with appropriate datasets and domain experts

## Ethical and Safety Considerations

This project is designed as an academic engineering prototype.

The system:

- Does not provide medical diagnoses.
- Does not prescribe treatments.
- Does not replace healthcare professionals.
- Does not claim that an anomaly represents a disease.
- Uses monitoring-oriented language such as **potential abnormal physiological pattern**, **early-warning signal**, and **requires review**.
- Keeps the human reviewer in the decision loop.
- Uses local LLM inference rather than sending monitoring data to an external LLM API.
- Keeps authentication credentials outside the source code.

Any future clinical deployment would require extensive validation, regulatory review, security controls, privacy protections, and involvement from qualified healthcare professionals.

## Dataset Citation

The project uses:

**Pimentel, M. A. F., Johnson, A. E. W., Clifton, D. A., Tarassenko, L., & Watkinson, P. J.**

"Towards a Robust Estimation of Respiratory Rate from Pulse Oximeters."

*IEEE Transactions on Biomedical Engineering*, 64(8), 1914–1923, 2016.

Dataset:

**BIDMC PPG and Respiration Dataset v1.0.0**

PhysioNet:

https://physionet.org/content/bidmc/1.0.0/

## Disclaimer

This repository contains an academic/research prototype for exploring the integration of:

**IoT + Healthcare Monitoring + Machine Learning + RAG + Local LLMs + Explainable AI**

The system is intended for educational and research purposes only.

It must **not** be used as a substitute for professional medical evaluation, diagnosis, or treatment.

## Author

**Rupika Upadhyay**

B.Tech CSE, ABES Engineering College

Academic/research project exploring the integration of:

**IoT + Healthcare Monitoring + Machine Learning + RAG + Local LLMs + Explainable AI**

## Project Status

**Status: Working Research / Engineering Prototype**

The current implementation successfully integrates:

- Real PhysioNet BIDMC physiological data
- Dataset preprocessing
- Personalized baseline analysis
- Temporal analysis
- Multisensor pattern analysis
- Isolation Forest anomaly detection
- MQTT telemetry replay
- HiveMQ Cloud
- Node-RED
- PostgreSQL
- RAG using Sentence Transformers and ChromaDB
- Local LLM explanations using Ollama
- Dynamic Streamlit dashboard
- Evidence-grounded monitoring explanations
- Human-in-the-loop monitoring workflow

The project is currently intended for **academic demonstration, experimentation, and further research development**.