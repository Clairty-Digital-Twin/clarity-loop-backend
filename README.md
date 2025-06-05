# CLARITY Digital Twin Platform Backend

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Licensed under the Apache 2.0 License.

## Overview

CLARITY is a backend platform designed to create a **mental health digital twin**, transforming raw wearable data (e.g., from Apple Watch) into clinically meaningful insights. While the ultimate vision focuses on comprehensive mental health assessment, the current platform provides a robust foundation by processing a wide array of physiological data. It combines multi-sensor data processing with machine learning to model an individual's health, with initial strengths in sleep patterns, daily activity, cardiovascular status, and circadian rhythms.

This document details the system's current functionality—what the codebase actually does today—and outlines a roadmap of features under development, ensuring transparency about which analytical components are fully implemented versus those that are experimental or planned.

**Disclaimer:** CLARITY is currently research software and is not an FDA-cleared medical device. Its outputs are intended for informational and research purposes only and should not be used for self-diagnosis or as a substitute for professional medical advice.

## Table of Contents
- [Overview](#overview)
- [Current Capabilities](#current-capabilities)
  - [At-a-Glance: Platform Components](#at-a-glance-platform-components)
  - [1. Pretrained Actigraphy Transformer (PAT)](#1-pretrained-actigraphy-transformer-pat--actigraphy-based-sleep--behavioral-analysis)
  - [2. Sleep Processor](#2-sleep-processor--clinical-sleep-metrics-from-logged-data)
  - [3. Cardiovascular Processor](#3-cardiovascular-processor--heart-rate--hrv-feature-extraction)
  - [4. Respiratory Processor](#4-respiratory-processor--breathing-rate-and-oxygen-saturation-analysis)
  - [5. Activity Processor](#5-activity-processor--daily-activity--fitness-summary)
  - [6. Fusion Transformer](#6-fusion-transformer--multi-modal-health-state-integration)
  - [7. Summary Statistics and Insights Generation](#7-summary-statistics-and-insights-generation)
- [AI/ML Pipeline Highlights](#aiml-pipeline-highlights)
- [Architecture](#architecture)
  - [Technology Stack](#technology-stack)
- [Roadmap – Towards a Mental Health Digital Twin](#roadmap--towards-a-mental-health-digital-twin)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Development Setup](#development-setup)
  - [Docker Development](#docker-development)
  - [Quick Demo](#quick-demo)
- [API Overview](#api-overview)
- [Security & Compliance](#security--compliance)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
  - [Code Standards](#code-standards)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Current Capabilities

CLARITY's backend can ingest and process wearable data to output a rich set of health metrics. These components are active in the codebase and form an asynchronous pipeline to produce an integrated health report.

### At-a-Glance: Platform Components

| Component                       | Key Output(s)                                       | Clinical Relevance (Examples)                                           |
|---------------------------------|-----------------------------------------------------|-------------------------------------------------------------------------|
| Pretrained Actigraphy Tx (PAT)  | Sleep quality metrics, Circadian score, Depression risk | Inferred sleep patterns, Rhythm regularity, Behavioral mood indicators    |
| Sleep Processor                 | Clinical sleep stats (efficiency, latency, WASO)    | Standardized sleep assessment from logged data                          |
| Cardiovascular Processor        | HR, HRV, Resting HR, Recovery score, Cardio-circadian | Cardiovascular health, Stress levels, Fitness tracking                  |
| Respiratory Processor           | Breathing rate, SpO₂, Stability scores              | Respiratory health, Potential sleep-disordered breathing indicators     |
| Activity Processor              | Step counts, Active energy, Exercise time, Consistency | Basic activity levels, Goal tracking, Movement pattern regularity       |
| Fusion Transformer              | Unified 64-D health vector                          | Research foundation for multi-modal risk prediction                     |
| Insights Generation (Gemini)    | Natural language health summaries                   | Actionable advice, User-friendly interpretation of complex metrics      |

### 1. Pretrained Actigraphy Transformer (PAT) – Actigraphy-Based Sleep & Behavioral Analysis

The PAT model analyzes wrist motion (actigraphy) data, often derived from step counts, to infer sleep-related metrics and behavioral patterns.

- **Sleep Quality Metrics:** An 8-element vector estimating key sleep characteristics (e.g., efficiency, latency, fragmentation) from one week of actigraphy.
- **Circadian Rhythm Regularity Score:** A single score indicating the consistency of the user's daily activity/sleep cycle.
- **Depression Risk Indicator:** A scalar output reflecting activity patterns correlated with elevated depression risk (based on the Dartmouth-trained model).
- **Activity Embedding:** A 96-dimensional latent vector (padded to 128 dimensions) representing the user's weekly activity profile.
*Note: PAT currently provides aggregate sleep metrics and does not perform minute-by-minute sleep stage labeling (which is a placeholder in the current code).*

### 2. Sleep Processor – Clinical Sleep Metrics from Logged Data

This module analyzes structured sleep records (e.g., from Apple HealthKit sleep tracking) to compute research-grade sleep features:

- Sleep efficiency, sleep latency, Wake After Sleep Onset (WASO).
- Sleep stage percentages (REM, deep sleep, if stage data is provided).
- Awakenings count, sleep consistency score.
- Overall sleep quality score (0-5).
These metrics are computed using accepted formulas and thresholds from clinical sleep medicine.

### 3. Cardiovascular Processor – Heart Rate & HRV Feature Extraction

Derives cardiovascular health indicators from heart rate (HR) and heart rate variability (HRV) time series:

- Average and Max Heart Rate.
- Resting Heart Rate estimate.
- Heart Rate Variability (HRV) as SDNN, if available.
- HRV Variability (standard deviation of the HRV signal).
- Heart Rate Recovery Score (0-1 index).
- Circadian Rhythm Score (Cardio) (0-1 score for day-night HR variation).
*Note: Currently focuses on summary statistics and regularity; arrhythmia detection is on the roadmap.*

### 4. Respiratory Processor – Breathing Rate and Oxygen Saturation Analysis

Handles breathing rate and blood oxygen saturation (SpO₂) data to compute an 8-feature vector:

- Average and Resting Respiratory Rate.
- Respiratory Variability.
- Average and Minimum SpO₂.
- SpO₂ Variability.
- Respiratory Stability Score (0-1 index).
- Oxygenation Efficiency Score (0-1 index).

### 5. Activity Processor – Daily Activity & Fitness Summary

Ingests raw activity metrics (steps, distance, exercise minutes) and produces user-friendly summaries (12 key metrics):

- Total and Average Daily Steps, Peak Daily Steps.
- Total and Average Daily Distance.
- Total and Average Daily Active Energy burned.
- Total and Average Daily Exercise Minutes.
- Total Flights Climbed, Total Active Minutes.
- Activity Consistency Score (0-1).
This complements PAT by providing transparent, easily explainable activity numbers.

### 6. Fusion Transformer – Multi-Modal Health State Integration

A PyTorch-based model that fuses feature vectors from multiple modalities (e.g., sleep + cardio + activity) into a single 64-dimensional vector.

- This fused vector represents an encoded snapshot of the person's overall physiological state.
- Currently, this vector is primarily used for research and future development (e.g., for advanced multi-modal analyses) and is not directly exposed as a standalone insight to end-users.

### 7. Summary Statistics and Insights Generation

- The pipeline produces summary statistics, data coverage information, and preliminary interpretations (e.g., health indicator flags like "good/fair/poor" for cardiovascular fitness).
- Results are saved to Firestore.
- Integrates with a natural language model (Google's Gemini) to translate metrics into human-readable advice via an API, bridging the gap between raw data and understanding.

## AI/ML Pipeline Highlights

- **PAT (Pretrained Actigraphy Transformer):** For nuanced analysis of movement data to infer sleep and behavioral patterns.
- **Dedicated Processors:** For detailed analysis of logged sleep, cardiovascular, respiratory, and activity data.
- **Fusion Transformer:** To create a unified, multi-modal representation of health.
- **Google Gemini Integration:** For generating natural language health insights from the processed metrics.

## Architecture

The platform is built on Clean Architecture principles, promoting a separation of concerns and maintainability.

![Architecture Diagram PNG](docs/images/architecture_diagram.png)

```mermaid
graph TD
    A[End User/Wearable] -->|Data Input| B(FastAPI Backend);
    B --> C{Data Processing & Validation};
    C --> D[PAT Model];
    C --> E[Sleep Processor];
    C --> F[Cardio Processor];
    C --> G[Respiratory Processor];
    C --> H[Activity Processor];
    subgraph Core Analytics
        D; E; F; G; H;
    end
    I[Fusion Transformer] --> J[Unified Health Vector];
    D & E & F & G & H --> K((Firestore Database));
    K --> L[Insights Generation / Gemini];
    L -->|Insights| B;
    style B fill:#00a393,stroke:#333,stroke-width:2px,color:#fff;
    style K fill:#f9f,stroke:#333,stroke-width:2px;
```

```
┌─────────────────────────────────────────────────────────┐
│ Frameworks & Drivers (FastAPI, GCP, Firebase)           │
├─────────────────────────────────────────────────────────┤
│ Interface Adapters (Controllers, DTOs, Gateways)        │
├─────────────────────────────────────────────────────────┤
│ Application Services (Use Cases, Business Rules)        │
├─────────────────────────────────────────────────────────┤
│ Domain Entities (Health Data, User, Analysis)           │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend Core**

- FastAPI, Pydantic, PyTorch

**AI/ML**

- PAT, Google Gemini, scikit-learn, pandas

**Infrastructure**

- Google Cloud Platform (GCP), Firestore, Firebase Auth, Pub/Sub, Cloud Storage

**Development & Monitoring**

- pytest, Black, Ruff, Prometheus, Grafana

## Roadmap – Towards a Mental Health Digital Twin

While the current platform provides robust physiological monitoring, future development is focused on leveraging these capabilities to build a comprehensive mental health digital twin.

- **Fine-Grained Sleep Stage & Apnea Detection:** Enhancing PAT or related algorithms for detailed sleep staging (light, deep, REM) from actigraphy and other signals. This will be crucial for linking sleep architecture to mood regulation and cognitive function.
- **Arrhythmia and Cardiac Anomaly Detection:** Adding detection of irregular heart rhythms. Understanding cardiac patterns can provide insights into stress, anxiety, and autonomic nervous system dysregulation relevant to mental health.
- **Multi-Modal Mental Health Risk Prediction & Alerts:** Leveraging the FusionTransformer's output to train models that predict risks for mental health episodes (e.g., depressive episodes, anxiety spikes) or track overall mental well-being.
- **Circadian Rhythm Optimization for Mental Well-being:** Offering personalized guidance to improve circadian regularity, a key factor in mood disorders, energy levels, and overall mental health.
- **Longitudinal Trend Analysis for Mental Health Trajectories:** Implementing views and metrics to track changes in health patterns over months or years, identifying subtle shifts that may indicate an emerging mental health concern or response to treatment.
- **Personalized Intervention & Psychiatric Coaching:** Using insights to provide tailored health coaching with a strong psychiatric and psychological component, potentially suggesting behavioral changes or flagging when professional consultation is advisable.
- **Expanded Data Source Integration for Holistic View:**
  - **Patient-Reported Outcomes (PROs):** Integrating mood logs, validated mental health questionnaires (e.g., PHQ-9, GAD-7), and contextual life-event data.
  - **Other Biometric Data:** Incorporating data from new sensors (e.g., EDA for stress, CGM for metabolic health's impact on mood) as they become relevant and available.
- **Clinical Validation & Explainability:** Rigorous validation of algorithms against clinical standards (e.g., PAT against polysomnography) and improving model explainability to build trust and clinical utility.

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Google Cloud Project (for production)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/The-Obstacle-Is-The-Way/clarity-loop-backend.git
cd clarity-loop-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment configuration
cp .env.example .env
# Edit .env with your configuration

# Run the development server
make dev
```

The API will be available at <http://localhost:8000> with interactive docs at <http://localhost:8000/docs>.

### Docker Development

```bash
# Start all services (API + emulators)
make dev-docker

# Run tests
make test

# Check code quality
make lint
```

### Quick Demo

A quick demo setup is available via `quick_demo.sh`. Refer to `README-DEMO.md` for more details on the demo environment.

## API Overview

The platform provides a comprehensive API for user management, health data upload, analysis, and insights retrieval.

| Endpoint | Description | Authentication |
|----------|-------------|----------------|
| `POST /api/v1/auth/register` | User registration | Public |
| `POST /api/v1/auth/login` | User authentication | Public |
| `POST /api/v1/health-data/upload` | Upload health metrics | Firebase JWT |
| `GET /api/v1/health-data/` | Retrieve health data | Firebase JWT |
| `POST /api/v1/pat/analyze-step-data` | PAT actigraphy analysis | Firebase JWT |
| `POST /api/v1/insights/generate` | Generate AI health insights | Firebase JWT |

**Example `curl` Request (PAT Analysis):**
```bash
# Ensure you have a sample_pat_request.json file with your step data
# Example sample_pat_request.json:
# {
#   "user_id": "user_test_123",
#   "step_data": [
#     {"timestamp": "2024-01-15T00:00:00Z", "step_count": 10},
#     {"timestamp": "2024-01-15T00:01:00Z", "step_count": 5}
#     // ... more data points
#   ]
# }
curl -X POST http://localhost:8000/api/v1/pat/analyze-step-data \\
     -H "Authorization: Bearer YOUR_FIREBASE_JWT_TOKEN" \\
     -H "Content-Type: application/json" \\
     -d @sample_pat_request.json
```

For detailed API documentation, refer to the interactive docs at <http://localhost:8000/docs> when the server is running.

## Security & Compliance

- End-to-end encryption for health data (at rest and in transit).
- HIPAA considerations in data handling and storage.
- Data never leaves GCP's us-central1 (or your chosen region) unless encrypted; no PHI stored outside HIPAA-eligible services.
- Firebase Authentication with JWT tokens.
- Role-based access control (RBAC) planned.
- Secure API key management.

## Testing

```bash
# Run all tests
make test

# Test coverage report
make coverage
```

The project aims for high test coverage across unit, integration, API, and ML tests.

## Documentation

- **API Documentation:** Interactive OpenAPI docs available at `/docs` when the server is running.
- **Further Documentation:** Detailed guides on architecture, integrations, and development can be found in the `docs/` directory. (See [Project Documentation](docs/README.md))

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-amazing-feature`.
3. Make your changes following coding standards.
4. Ensure tests pass: `make test`.
5. Ensure linting passes: `make lint`.
6. Commit your changes: `git commit -m 'Add some amazing feature'`.
7. Push to the branch: `git push origin feature/your-amazing-feature`.
8. Open a Pull Request.

### Code Standards

- Adhere to Clean Architecture principles.
- Aim for high test coverage (target >80%).
- Use type hints and clear docstrings.
- Follow Black code formatting.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

### AI Foundation Models for Wearable Movement Data

This platform integrates the Pretrained Actigraphy Transformer (PAT), an open-source foundation model for time-series wearable movement data developed by the Jacobson Lab at Dartmouth College.

**Citation:**

```
Ruan, Franklin Y., Zhang, Aiwei, Oh, Jenny, Jin, SouYoung, and Jacobson, Nicholas C.
"AI Foundation Models for Wearable Movement Data in Mental Health Research."
arXiv:2411.15240 (2024). https://doi.org/10.48550/arXiv.2411.15240
```

**Repository:** [njacobsonlab/Pretrained-Actigraphy-Transformer](https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer)
**License:** CC-BY-4.0
**Corresponding Author:** Franklin Ruan (<franklin.y.ruan.24@dartmouth.edu>)

### Additional Acknowledgments

- Google Gemini - For advanced language model capabilities.
- Apple HealthKit - As a primary source of health data.
- The principles of Clean Architecture by Robert C. Martin.
- The open-source community.
