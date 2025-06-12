# CLARITY Digital Twin Platform - System Overview

## Executive Summary

CLARITY processes Apple Health data using machine learning models to generate health insights and natural language explanations. The system combines a Pretrained Actigraphy Transformer (PAT) for temporal analysis with Google Gemini for conversational health insights.

### Core Functionality

**Input**: Apple Health exports (heart rate, sleep, activity, HRV)  
**Processing**: PAT transformer + Google Gemini AI  
**Output**: Health pattern analysis with natural language explanations

Example use case:
```
Input: 7 days of sleep and activity data
Analysis: Sleep efficiency trends, circadian rhythm analysis
Output: "Sleep efficiency decreased 15% this week due to irregular bedtimes"
```

## System Architecture

### High-Level Data Flow

```
Apple Watch/iPhone
       ↓
   HealthKit Export (JSON)
       ↓
   CLARITY API (/healthkit/upload)
       ↓ 
   Data Validation & Preprocessing
       ↓
   ┌──────────────────┐    ┌───────────────────┐
   │ PAT Transformer  │    │     Gemini AI     │
   │ (Sleep Analysis) │    │ (Text Generation) │
   └──────────────────┘    └───────────────────┘
           ↓                        ↓
   Health Pattern Analysis    Natural Language Insights
           ↓                        ↓
   ┌──────────────────────────────────────────┐
   │        Structured Output                 │
   │     Health summaries and trends          │
   └──────────────────────────────────────────┘
```

### Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway (AWS ALB)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                FastAPI Application                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │   Auth      │ │ Health Data │ │    AI Processing        ││
│  │ (Cognito)   │ │   (CRUD)    │ │ (PAT + Gemini)          ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Data Layer                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │  DynamoDB   │ │     S3      │ │    External APIs        ││
│  │(User Data)  │ │(Raw Files)  │ │(Gemini, HealthKit)      ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Ingestion (`/healthkit/*` endpoints)
- **Purpose**: Receive and validate Apple HealthKit exports
- **Key Files**: `src/clarity/integrations/healthkit.py`, `src/clarity/api/v1/healthkit.py`
- **Data Types**: Heart rate, sleep, activity, HRV, respiratory rate, blood pressure
- **Processing**: Time-series alignment, outlier removal, normalization

### 2. PAT (Pretrained Actigraphy Transformer) 
- **Purpose**: Sleep and circadian rhythm analysis from movement data
- **Key Files**: `src/clarity/ml/pat_service.py`, `research/Pretrained-Actigraphy-Transformer/`
- **Input**: 7-day activity vectors (10,080 time points)
- **Output**: Sleep quality scores, circadian disruption detection
- **License**: CC BY-4.0 (Dartmouth College, Jacobson Lab)

### 3. Gemini AI Integration
- **Purpose**: Natural language health insight generation
- **Key Files**: `src/clarity/ml/gemini_service.py`, `src/clarity/api/v1/gemini_insights.py`
- **Capabilities**: 
  - Generate health summaries from time-series data
  - Create natural language explanations of health patterns
  - Provide contextual health recommendations
  - Process complex health data queries

### 4. Health Data Processing Pipeline
- **Purpose**: Transform raw HealthKit data into ML-ready features
- **Key Files**: `src/clarity/ml/processors/`, `src/clarity/integrations/apple_watch.py`
- **Features**:
  - **Temporal Alignment**: All data resampled to 1-minute intervals
  - **Outlier Detection**: Physiological bounds validation
  - **Signal Processing**: Butterworth filters for heart rate, median filters for respiratory rate
  - **Normalization**: Population-based z-scores using NHANES data

### 5. WebSocket Real-time Streaming
- **Purpose**: Live health data streaming and real-time communication
- **Key Files**: `src/clarity/api/v1/websocket/`
- **Use Cases**: Real-time activity monitoring, live data updates

## Data Models

### Input Data (HealthKit)
```python
{
  "heart_rate_samples": [
    {
      "timestamp": "2024-01-01T12:00:00Z",
      "value": 72,
      "unit": "bpm",
      "source": "apple_watch"
    }
  ],
  "sleep_samples": [...],
  "activity_samples": [...]
} 
```

### Processed Features (PAT Ready)
```python
{
  "actigraphy_sequence": [0.1, 0.3, 2.1, ...],  # 10,080 points
  "heart_rate_sequence": [1.2, 1.1, 1.4, ...],  # Normalized
  "sequence_length": 10080,
  "sampling_rate_minutes": 1,
  "statistics": {
    "activity_mean": 2.3,
    "zero_activity_percentage": 0.32
  }
}
```

### AI Insights Output
```python
{
  "summary": "Sleep efficiency trends show 15% decrease this week",
  "recommendations": ["Consider consistent bedtime routine"],
  "metrics": {
    "sleep_efficiency": 0.78,
    "circadian_stability": 0.65
  },
  "analysis_complete": True
}
```

## Performance Characteristics

### Scale
- **Concurrent Users**: Designed for high-throughput processing
- **Data Volume**: ~10MB per user per week (HealthKit export)
- **Processing**: Real-time for simple queries, <30s for ML analysis
- **Storage**: DynamoDB with auto-scaling

### Processing Accuracy
- **Heart Rate Processing**: Validated against clinical standards
- **Data Validation**: Multi-layer schema and physiological bounds checking
- **ML Pipeline**: Consistent preprocessing and feature extraction

### Availability
- **Infrastructure**: AWS ECS Fargate with auto-scaling
- **Health Checks**: Multi-layer monitoring (container, load balancer, application)
- **Graceful Degradation**: Core functions remain available during external service issues

## Security & Privacy

### Data Protection
- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Access Control**: JWT tokens via AWS Cognito
- **Audit Logging**: All health data access logged
- **Data Residency**: Configurable by region

### Compliance
- **HIPAA Ready**: Infrastructure meets HIPAA technical safeguards
- **AWS Compliance**: Leverages AWS SOC 2 and other certifications
- **Data Controls**: Complete data export and deletion capabilities

### Privacy Implementation
- **Minimal Data**: Only necessary health metrics stored
- **User Control**: Granular data access and retention controls
- **Separation**: PII separated from health analytics data
- **Consent Management**: Configurable permissions for data types

## Current Status

### Production Ready
- **Tests**: 807/810 tests passing (99.6% success rate)
- **Coverage**: 57% (target: 85%)
- **API**: 44 endpoints, all core functionality operational
- **Infrastructure**: Complete AWS deployment with monitoring

### Development Focus
- **Test Coverage**: Increasing coverage to production standards
- **Performance**: Optimizing ML inference times
- **Documentation**: Comprehensive technical documentation
- **Integration**: Additional health data source support

### Roadmap
- **Enhanced ML**: Additional health pattern analysis models
- **Real-time Processing**: Improved streaming data capabilities
- **Extended Integrations**: Support for additional wearable devices
- **Research Platform**: Tools for health data research and analysis

## Getting Started

### Development Setup
```bash
git clone https://github.com/your-org/clarity-loop-backend.git
cd clarity-loop-backend
make install && make dev
```

### API Usage
1. Export Apple Health data
2. POST to `/api/v1/healthkit/upload`
3. Process data through ML pipeline
4. Retrieve insights via `/api/v1/insights/*`

### Integration
- Complete API documentation available in [API Reference](02-api-reference.md)
- WebSocket support for real-time applications
- Comprehensive error handling and status reporting

---

**Next**: Read [API Reference](02-api-reference.md) for detailed endpoint documentation 