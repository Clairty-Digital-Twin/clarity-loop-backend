# CLARITY Architecture Overview

**UPDATED:** December 6, 2025 - Based on actual implementation

## System Architecture

CLARITY Digital Twin Platform uses a **clean architecture** approach with a single, well-designed FastAPI application. This is not a microservices architecture, but rather a modular monolith with clear separation of concerns.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ iOS App     │  │ Web App     │  │    API Clients      │  │
│  │ (Future)    │  │ (Future)    │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────────────────┐
                    │   Load Balancer     │
                    │  (Google Cloud)     │
                    └─────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   CLARITY Backend API                       │
│                  (FastAPI Application)                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                API Layer (v1)                       │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │    │
│  │  │   Auth   │ │  Health  │ │ Insights │ │  PAT   │  │    │
│  │  │          │ │   Data   │ │          │ │ Model  │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                Service Layer                        │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │    │
│  │  │   Auth   │ │  Health  │ │ AI/ML    │ │ Pub/Sub│  │    │
│  │  │ Service  │ │   Data   │ │ Service  │ │Service │  │    │
│  │  │          │ │ Service  │ │          │ │        │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Core Layer                         │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │    │
│  │  │ Domain   │ │  Models  │ │   ML     │ │  Core  │  │    │
│  │  │ Models   │ │    &     │ │  Core    │ │ Utils  │  │    │
│  │  │          │ │ Entities │ │          │ │        │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Infrastructure Layer                   │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │    │
│  │  │Firebase  │ │ Google   │ │ Gemini   │ │  File  │  │    │
│  │  │ Auth +   │ │ Cloud    │ │   AI     │ │Storage │  │    │
│  │  │Firestore │ │ Storage  │ │          │ │        │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 External Services                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Firebase    │  │ Google      │  │ Google AI Platform  │  │
│  │ Auth &      │  │ Cloud       │  │ (Gemini 2.5 Pro)    │  │
│  │ Firestore   │  │ Platform    │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Clean Architecture Principles

### 1. Dependency Inversion
- **High-level modules don't depend on low-level modules**
- Services depend on abstractions (ports), not concrete implementations
- Infrastructure adapters implement the ports

### 2. Separation of Concerns
- **API Layer**: HTTP handling, request validation, response formatting
- **Service Layer**: Business logic, orchestration, use cases  
- **Core Layer**: Domain models, entities, business rules
- **Infrastructure**: External integrations, databases, AI models

### 3. Testability
- **Easy to mock**: Abstract interfaces for all external dependencies
- **Unit testing**: Core business logic isolated from infrastructure
- **Integration testing**: Full API flows with real dependencies

## Directory Structure

```
src/clarity/
├── api/v1/              # API Layer - FastAPI routers
│   ├── auth.py          # Authentication endpoints
│   ├── health_data.py   # Health data upload/retrieval
│   ├── insights.py      # AI insights generation
│   └── pat.py           # PAT model analysis
├── services/            # Service Layer - Business logic
│   ├── auth/            # Authentication services
│   ├── health_data/     # Health data processing
│   ├── ai/              # AI/ML services
│   └── pubsub/          # Async processing
├── core/                # Core Layer - Domain logic
│   ├── models/          # Domain models and entities
│   ├── exceptions/      # Business exceptions
│   └── config/          # Configuration
├── ports/               # Abstractions/Interfaces
│   ├── repositories/    # Data access abstractions
│   └── services/        # Service abstractions
├── storage/             # Infrastructure - Adapters
│   ├── firestore/       # Firestore implementation
│   ├── cloud_storage/   # Google Cloud Storage
│   └── firebase_auth/   # Firebase Auth integration
├── ml/                  # ML Infrastructure
│   ├── processors/      # ML pipeline processors
│   └── models/          # Model loading and inference
└── integrations/        # External Service Integrations
    ├── gemini/          # Google AI integration
    └── healthkit/       # Apple HealthKit integration
```

## Key Components

### Authentication & Authorization
- **Firebase Authentication**: Industry-standard auth provider
- **JWT Tokens**: Stateless authentication
- **User Isolation**: Strict data access controls

### Health Data Pipeline
- **Upload**: RESTful API for health metric ingestion
- **Storage**: Google Cloud Storage for raw data
- **Processing**: Async background processing with Pub/Sub
- **Analysis**: PAT model + Gemini AI insights

### Machine Learning Stack
- **PAT Model**: Pretrained Actigraphy Transformer (Dartmouth)
- **Gemini AI**: Google's advanced language model
- **TensorFlow**: Model serving and inference
- **Background Processing**: Async ML pipeline

### Data Layer
- **Firestore**: NoSQL database for structured data
- **Cloud Storage**: Blob storage for health data files
- **Real-time Updates**: Firestore listeners for live data
- **ACID Transactions**: Consistent data operations

## Scalability Design

### Horizontal Scaling
- **Stateless Application**: No server-side session state
- **Cloud Run**: Auto-scaling container deployment
- **Load Balancing**: Google Cloud Load Balancer
- **Database Scaling**: Firestore automatic scaling

### Asynchronous Processing
- **Pub/Sub Queues**: Decouple API from heavy processing
- **Background Workers**: Separate processing containers
- **Real-time Updates**: WebSocket/SSE for live data

### Caching Strategy
- **Response Caching**: Frequently accessed insights
- **Model Caching**: Keep ML models in memory
- **Data Caching**: Cache user data for fast access

## Security Architecture

### Data Protection
- **Encryption**: All data encrypted at rest and in transit
- **User Isolation**: Firebase security rules + application-level checks
- **API Authentication**: JWT token validation on all endpoints
- **Input Validation**: Comprehensive request validation

### Privacy by Design
- **Minimal Data Collection**: Only necessary health metrics
- **User Control**: Users control their data sharing
- **Audit Logging**: Complete access trail
- **HIPAA Readiness**: Healthcare compliance design

## Deployment Architecture

### Google Cloud Platform
- **Cloud Run**: Containerized application deployment
- **Cloud Storage**: Health data file storage
- **Firestore**: Primary database
- **Pub/Sub**: Async message processing
- **Cloud Build**: CI/CD pipeline
- **Cloud Monitoring**: Observability and alerting

### Development Environment
- **Local Development**: Docker Compose setup
- **Testing**: Comprehensive test suite (729 tests)
- **CI/CD**: Automated testing and deployment
- **Monitoring**: Prometheus metrics + Grafana dashboards

## Technology Stack

### Core Technologies
- **Python 3.12**: Modern, type-safe Python
- **FastAPI**: High-performance async web framework
- **Pydantic**: Data validation and serialization
- **TensorFlow**: Machine learning model serving

### External Services
- **Firebase**: Authentication and real-time database
- **Google Cloud**: Infrastructure and storage
- **Gemini AI**: Advanced language model for insights
- **PAT Model**: Specialized actigraphy analysis

### Development Tools
- **pytest**: Comprehensive testing framework
- **mypy**: Static type checking
- **ruff**: Fast Python linting
- **Docker**: Containerization

## Implementation Status

### ✅ Completed Components
- Clean architecture foundation
- Authentication system
- Health data upload/storage
- PAT model integration
- Gemini AI insights
- Async processing pipeline
- Comprehensive testing (729 tests)

### 🚧 In Progress
- Test coverage improvement (59% → 85%)
- Error handling edge cases
- Performance optimization

### 📋 Planned
- iOS/Web client applications
- Advanced analytics dashboard
- Sleep processor module
- Additional ML models
- Enhanced monitoring/alerting