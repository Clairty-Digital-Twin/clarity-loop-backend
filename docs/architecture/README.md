# Architecture Documentation

This directory contains comprehensive architectural documentation for the Clarity Loop Backend system.

## Architecture Overview

Clarity Loop Backend implements an **async-first, event-driven architecture** optimized for high-throughput health data processing and real-time insights delivery. The system is designed with security-first principles, HIPAA-inspired compliance practices, and enterprise-grade scalability.

## System Design Principles

### 1. Async-First Architecture

- **Immediate Response**: Fast acknowledgment of data uploads for optimal user experience
- **Background Processing**: Complex analytics run asynchronously without blocking user interactions
- **Event-Driven Communication**: Services communicate through Cloud Pub/Sub messaging
- **Real-Time Updates**: Results pushed to clients via Firestore real-time listeners

### 2. Security by Design

- **Zero-Trust Model**: No implicit trust between components
- **End-to-End Encryption**: Data encrypted at rest and in transit
- **Principle of Least Privilege**: Minimal access rights for all components
- **Data Segregation**: User data isolated and access-controlled

### 3. Cloud-Native Scalability

- **Serverless Architecture**: Auto-scaling Cloud Run services
- **Managed Services**: Leverage Google Cloud's managed database and AI services
- **Multi-Region Support**: Global deployment for high availability
- **Elastic Scaling**: Automatic scaling based on demand

## Core Components

### 1. API Gateway (FastAPI on Cloud Run)

- **Purpose**: Primary entry point for all client requests
- **Technology**: FastAPI with async/await support
- **Responsibilities**:
  - Request authentication and authorization
  - Data validation and sanitization
  - Rate limiting and DDoS protection
  - Request routing to appropriate services

### 2. Actigraphy Transformer Microservice

- **Purpose**: Specialized ML service for health data analysis
- **Technology**: Custom PyTorch/TensorFlow model on Cloud Run
- **Responsibilities**:
  - Sleep pattern analysis
  - Activity classification
  - Circadian rhythm detection
  - Anomaly detection in health metrics

### 3. Data Processing Pipeline

- **Purpose**: Async processing of health data and insight generation
- **Technology**: Cloud Pub/Sub + Cloud Functions/Cloud Run
- **Responsibilities**:
  - Data ingestion and validation
  - ML model orchestration
  - Result aggregation and storage
  - Notification delivery

### 4. AI Insights Engine

- **Purpose**: Natural language insight generation
- **Technology**: Vertex AI with Gemini 2.5 Pro (`gemini-2.5-pro-preview-05-06`)
- **Model Specifications**:
  - Input Token Limit: 1,048,576 tokens (1M+ context window)
  - Output Token Limit: 65,535 tokens
  - Supported Input Types: Text, Code, Images, Audio, Video
  - Launch Stage: Public Preview
  - Primary Region: us-central1
- **Advanced Capabilities**:
  - Grounding with Google Search for factual accuracy
  - Code execution for dynamic analysis
  - System instructions for consistent clinical behavior
  - Controlled generation for precise output formatting
  - Function calling for external API integration
  - Context caching for optimized performance
  - Vertex AI RAG Engine integration
  - Chat completions for interactive interfaces
- **Responsibilities**:
  - Health trend analysis with evidence-based insights
  - Personalized recommendations based on clinical guidelines
  - Natural language report generation with medical accuracy
  - Predictive health insights using multimodal analysis

## Data Flow Architecture

```
iOS/watchOS App
      ↓
Firebase Auth → FastAPI Gateway → Data Validation
      ↓                              ↓
User Context ←→ Cloud Firestore ←→ Cloud Pub/Sub
      ↓                              ↓
Real-time UI ←←←←←←←←←←←←←←←←← Processing Pipeline
                                     ↓
                           Actigraphy Transformer
                                     ↓
                              Vertex AI (Gemini)
                                     ↓
                              Insight Storage
```

## Security Architecture

### Authentication & Authorization

- **Firebase Identity Platform**: HIPAA-compliant user authentication
- **Cloud IAM**: Service-to-service authorization
- **Custom Claims**: Role-based access control
- **JWT Tokens**: Secure session management

### Data Protection

- **Encryption at Rest**: AES-256 encryption for all stored data
- **Encryption in Transit**: TLS 1.3 for all network communication
- **Key Management**: Cloud KMS for encryption key lifecycle
- **Data Masking**: PII protection in logs and non-production environments

### Network Security

- **VPC Isolation**: Private network for backend services
- **Firewall Rules**: Ingress/egress traffic control
- **Service Mesh**: Istio for service-to-service communication
- **DDoS Protection**: Cloud Armor for application-layer protection

## Scalability & Performance

### Horizontal Scaling

- **Cloud Run**: Automatic container scaling (0-1000+ instances)
- **Load Balancing**: Global HTTP(S) load balancer
- **Database Scaling**: Firestore auto-scaling
- **CDN**: Global content delivery network

### Performance Optimization

- **Caching Strategy**: Multi-level caching (Redis, CDN, application)
- **Connection Pooling**: Efficient database connection management
- **Async Processing**: Non-blocking I/O operations
- **Resource Optimization**: CPU/memory usage monitoring and tuning

## Monitoring & Observability

### Logging Strategy

- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Centralized Logging**: Cloud Logging for all services
- **Log Levels**: Appropriate log levels for different environments
- **Sensitive Data**: No PII in logs, secure audit trails

### Metrics & Monitoring

- **Application Metrics**: Custom business metrics and KPIs
- **Infrastructure Metrics**: Resource utilization monitoring
- **Real-time Dashboards**: Cloud Monitoring dashboards
- **Alerting**: Proactive incident detection and notification

### Tracing & Debugging

- **Distributed Tracing**: Cloud Trace for request flow analysis
- **Error Tracking**: Cloud Error Reporting for exception management
- **Performance Profiling**: Cloud Profiler for optimization insights
- **Debugging Tools**: Cloud Debugger for production troubleshooting

## Documentation Index

- **[System Components](./components.md)** - Detailed component specifications
- **[Data Models](./data-models.md)** - Database schemas and data structures
- **[Security Architecture](./security.md)** - Comprehensive security design
- **[Deployment Architecture](./deployment.md)** - Infrastructure and deployment patterns
- **[Integration Patterns](./integrations.md)** - External service integration designs
- **[Performance & Scaling](./performance.md)** - Scalability and optimization strategies
