# Clarity Loop Backend

Backend service for the Clarity Digital Twin project - a HealthKit-based wellness app with comprehensive data collection, analytics, and insights.

## Architecture

This backend consists of:

- FastAPI service on Google Cloud Run
- Actigraphy Transformer microservice for health data analysis
- Integration with Google Cloud services (Firestore, Cloud Storage, Vertex AI)
- Secure authentication via Firebase Auth

## Development

### Prerequisites

- Python 3.9+
- Google Cloud SDK
- Firebase CLI (optional)

### Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running Locally

```
uvicorn app.main:app --reload
```

## Deployment

The backend is designed to be deployed on Google Cloud Run:

```
gcloud run deploy clarity-loop-backend --source .
```

## Security

This application implements HIPAA-inspired security practices:
- End-to-end encryption
- Secure authentication
- Least privilege access
- Data segregation by user

## License

Proprietary - All rights reserved
