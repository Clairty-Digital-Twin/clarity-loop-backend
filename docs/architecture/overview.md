# Architecture Overview

## System Components

The Clarity Loop Backend consists of the following key components:

1. **FastAPI Service** - Main API gateway deployed on Google Cloud Run
2. **Actigraphy Transformer Service** - ML microservice for health data analysis
3. **Cloud Storage** - For raw HealthKit data storage
4. **Firestore** - For user metadata and analysis results
5. **Vertex AI (Gemini)** - For natural language insight generation

## Data Flow

1. iOS/watchOS app collects HealthKit data
2. Data is serialized and uploaded to backend via secure API
3. Backend stores raw data in Cloud Storage
4. Data is processed by Actigraphy Transformer service
5. Results are enhanced with natural language insights via Gemini
6. Final insights are stored in Firestore and returned to user

## Security Architecture

Security is implemented at multiple layers:
- Firebase Auth for user authentication
- HTTPS for all API endpoints
- Encryption at rest for all data stores
- Fine-grained IAM permissions
- Data segregation by user ID
