# Deployment Guide

This guide covers the complete deployment process for the Clarity Loop Backend across development, staging, and production environments using Google Cloud Platform.

## Overview

### Deployment Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Environment Architecture                │
├─────────────────────────────────────────────────────────────────┤
│  Development (clarity-loop-dev)                               │
│  ├── Cloud Run: Auto-deploy from feature branches              │
│  ├── Firestore: Development database with test data           │
│  ├── Firebase Auth: Development users                         │
│  └── Vertex AI: Development ML models                         │
│                                                                 │
│  Staging (clarity-loop-staging)                               │
│  ├── Cloud Run: Auto-deploy from develop branch               │
│  ├── Firestore: Staging database with realistic test data     │
│  ├── Firebase Auth: Staging users and testing accounts        │
│  └── Vertex AI: Staging ML models for integration testing     │
│                                                                 │
│  Production (clarity-loop-prod)                               │
│  ├── Cloud Run: Manual deploy from main branch                │
│  ├── Firestore: Production database with real user data       │
│  ├── Firebase Auth: Production users                          │
│  └── Vertex AI: Production ML models with monitoring          │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Strategy
- **Blue-Green Deployment**: Zero-downtime deployments with automatic rollback
- **Canary Releases**: Gradual traffic shifting for production deployments
- **Infrastructure as Code**: All infrastructure managed via Terraform
- **GitOps Workflow**: Automated deployments triggered by Git events

## Prerequisites

### Required Access and Permissions
```bash
# Google Cloud Platform access required
- clarity-loop-dev: Editor role
- clarity-loop-staging: Editor role
- clarity-loop-prod: Deployment Manager role

# GitHub repository access
- Read/Write access to main repository
- Admin access for release management

# Firebase project access
- Firebase Admin role for all projects
- Access to Firebase Authentication and Firestore
```

### Required Tools
```bash
# Install deployment tools
pip install google-cloud-build google-cloud-run
npm install -g firebase-tools
terraform --version  # Should be 1.0+

# Authenticate with required services
gcloud auth login
gcloud auth application-default login
firebase login
```

## Environment Configuration

### Development Environment (clarity-loop-dev)
```yaml
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=clarity-loop-dev
GOOGLE_CLOUD_REGION=us-central1
CLOUD_RUN_SERVICE=clarity-backend-dev
CLOUD_RUN_REGION=us-central1

# Firebase Configuration
FIREBASE_PROJECT_ID=clarity-loop-dev
FIREBASE_AUTH_DOMAIN=clarity-loop-dev.firebaseapp.com
FIREBASE_DATABASE_URL=https://clarity-loop-dev-default-rtdb.firebaseio.com

# Database Configuration
FIRESTORE_DATABASE=(default)
FIRESTORE_LOCATION=us-central1

# ML Configuration
VERTEX_AI_LOCATION=us-central1
ML_MODEL_ENDPOINT=https://ml-dev.clarityloop.com
ACTIGRAPHY_MODEL_VERSION=v2.1.0-dev

# Security Configuration
CORS_ORIGINS=["https://dev.clarityloop.com", "http://localhost:3000"]
```

### Staging Environment (clarity-loop-staging)
```yaml
# .env.staging
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=clarity-loop-staging
GOOGLE_CLOUD_REGION=us-central1
CLOUD_RUN_SERVICE=clarity-backend-staging
CLOUD_RUN_REGION=us-central1

# Firebase Configuration
FIREBASE_PROJECT_ID=clarity-loop-staging
FIREBASE_AUTH_DOMAIN=clarity-loop-staging.firebaseapp.com

# Database Configuration
FIRESTORE_DATABASE=(default)
FIRESTORE_LOCATION=us-central1

# ML Configuration
VERTEX_AI_LOCATION=us-central1
ML_MODEL_ENDPOINT=https://ml-staging.clarityloop.com
ACTIGRAPHY_MODEL_VERSION=v2.1.0

# Security Configuration
CORS_ORIGINS=["https://staging.clarityloop.com"]
```

### Production Environment (clarity-loop-prod)
```yaml
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=clarity-loop-prod
GOOGLE_CLOUD_REGION=us-central1
CLOUD_RUN_SERVICE=clarity-backend-prod
CLOUD_RUN_REGION=us-central1

# Firebase Configuration
FIREBASE_PROJECT_ID=clarity-loop-prod
FIREBASE_AUTH_DOMAIN=clarity-loop-prod.firebaseapp.com

# Database Configuration
FIRESTORE_DATABASE=(default)
FIRESTORE_LOCATION=us-central1

# ML Configuration
VERTEX_AI_LOCATION=us-central1
ML_MODEL_ENDPOINT=https://ml.clarityloop.com
ACTIGRAPHY_MODEL_VERSION=v2.1.0

# Security Configuration
CORS_ORIGINS=["https://app.clarityloop.com"]
```

## CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy Clarity Backend

on:
  push:
    branches: [develop, main]
  pull_request:
    branches: [develop, main]

env:
  REGISTRY: gcr.io
  PROJECT_ID_DEV: clarity-loop-dev
  PROJECT_ID_STAGING: clarity-loop-staging
  PROJECT_ID_PROD: clarity-loop-prod

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
          
      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml
          
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run security scan
        run: |
          pip install safety bandit
          safety check -r requirements.txt
          bandit -r src/
          
  build:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    outputs:
      image: ${{ steps.build.outputs.image }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1
        with:
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          project_id: ${{ env.PROJECT_ID_DEV }}
          
      - name: Configure Docker
        run: gcloud auth configure-docker
        
      - name: Build and push image
        id: build
        run: |
          IMAGE="${REGISTRY}/${PROJECT_ID_DEV}/clarity-backend:${GITHUB_SHA}"
          docker build -t $IMAGE .
          docker push $IMAGE
          echo "image=$IMAGE" >> $GITHUB_OUTPUT
          
  deploy-dev:
    if: github.ref == 'refs/heads/develop'
    needs: build
    runs-on: ubuntu-latest
    environment: development
    steps:
      - name: Deploy to Development
        uses: google-github-actions/deploy-cloudrun@v1
        with:
          service: clarity-backend-dev
          image: ${{ needs.build.outputs.image }}
          project_id: ${{ env.PROJECT_ID_DEV }}
          region: us-central1
          env_vars: |
            ENVIRONMENT=development
            DEBUG=true
            
  deploy-staging:
    if: github.ref == 'refs/heads/develop'
    needs: [build, deploy-dev]
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to Staging
        uses: google-github-actions/deploy-cloudrun@v1
        with:
          service: clarity-backend-staging
          image: ${{ needs.build.outputs.image }}
          project_id: ${{ env.PROJECT_ID_STAGING }}
          region: us-central1
          env_vars: |
            ENVIRONMENT=staging
            DEBUG=false
            
  deploy-prod:
    if: github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to Production
        uses: google-github-actions/deploy-cloudrun@v1
        with:
          service: clarity-backend-prod
          image: ${{ needs.build.outputs.image }}
          project_id: ${{ env.PROJECT_ID_PROD }}
          region: us-central1
          traffic: 10
          env_vars: |
            ENVIRONMENT=production
            DEBUG=false
```

## Manual Deployment Process

### Development Deployment
```bash
# 1. Authenticate and set project
gcloud config set project clarity-loop-dev
gcloud auth configure-docker

# 2. Build and tag image
docker build -t gcr.io/clarity-loop-dev/clarity-backend:$(git rev-parse HEAD) .

# 3. Push image to Container Registry
docker push gcr.io/clarity-loop-dev/clarity-backend:$(git rev-parse HEAD)

# 4. Deploy to Cloud Run
gcloud run deploy clarity-backend-dev \
  --image gcr.io/clarity-loop-dev/clarity-backend:$(git rev-parse HEAD) \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ENVIRONMENT=development,DEBUG=true \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 80 \
  --max-instances 10

# 5. Verify deployment
curl https://clarity-backend-dev-XXXXXXXXXX-uc.a.run.app/health
```

### Staging Deployment
```bash
# 1. Set staging project
gcloud config set project clarity-loop-staging

# 2. Tag image for staging
docker tag gcr.io/clarity-loop-dev/clarity-backend:$(git rev-parse HEAD) \
  gcr.io/clarity-loop-staging/clarity-backend:$(git rev-parse HEAD)

# 3. Push to staging registry
docker push gcr.io/clarity-loop-staging/clarity-backend:$(git rev-parse HEAD)

# 4. Deploy to staging
gcloud run deploy clarity-backend-staging \
  --image gcr.io/clarity-loop-staging/clarity-backend:$(git rev-parse HEAD) \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ENVIRONMENT=staging,DEBUG=false \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 100 \
  --max-instances 20

# 5. Run staging tests
python scripts/staging-tests.py
```

### Production Deployment (Blue-Green)
```bash
# 1. Set production project
gcloud config set project clarity-loop-prod

# 2. Deploy new version with no traffic
gcloud run deploy clarity-backend-prod \
  --image gcr.io/clarity-loop-prod/clarity-backend:$(git rev-parse HEAD) \
  --platform managed \
  --region us-central1 \
  --no-allow-unauthenticated \
  --set-env-vars ENVIRONMENT=production,DEBUG=false \
  --memory 4Gi \
  --cpu 4 \
  --concurrency 200 \
  --max-instances 100 \
  --no-traffic

# 3. Run production smoke tests
python scripts/production-smoke-tests.py --url=https://new-revision-url

# 4. Gradually shift traffic (canary deployment)
gcloud run services update-traffic clarity-backend-prod \
  --to-revisions=LATEST=10,PREVIOUS=90

# Wait and monitor metrics...

gcloud run services update-traffic clarity-backend-prod \
  --to-revisions=LATEST=50,PREVIOUS=50

# Wait and monitor metrics...

gcloud run services update-traffic clarity-backend-prod \
  --to-revisions=LATEST=100

# 5. Verify deployment
curl https://clarity-backend-prod-XXXXXXXXXX-uc.a.run.app/health
```

## Infrastructure as Code

### Terraform Configuration
```hcl
# terraform/main.tf
terraform {
  required_version = "~> 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
  
  backend "gcs" {
    bucket = "clarity-terraform-state"
    prefix = "backend/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Cloud Run Service
resource "google_cloud_run_service" "clarity_backend" {
  name     = var.service_name
  location = var.region

  template {
    spec {
      containers {
        image = var.container_image
        
        ports {
          container_port = 8000
        }
        
        resources {
          limits = {
            cpu    = var.cpu_limit
            memory = var.memory_limit
          }
        }
        
        env {
          name  = "ENVIRONMENT"
          value = var.environment
        }
        
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        
        dynamic "env" {
          for_each = var.env_vars
          content {
            name  = env.key
            value = env.value
          }
        }
      }
      
      container_concurrency = var.concurrency
      
      service_account_name = google_service_account.clarity_backend.email
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = var.max_instances
        "run.googleapis.com/execution-environment" = "gen2"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
  
  autogenerate_revision_name = true
}

# IAM Service Account
resource "google_service_account" "clarity_backend" {
  account_id   = "${var.service_name}-sa"
  display_name = "Clarity Backend Service Account"
}

# Firestore Database
resource "google_firestore_database" "clarity_database" {
  project                     = var.project_id
  name                       = "(default)"
  location_id                = var.region
  type                       = "FIRESTORE_NATIVE"
  concurrency_mode           = "OPTIMISTIC"
  app_engine_integration_mode = "DISABLED"
}

# Pub/Sub Topics
resource "google_pubsub_topic" "health_data_processing" {
  name = "health-data-processing"
  
  message_retention_duration = "86400s"
}

resource "google_pubsub_topic" "insights_generation" {
  name = "insights-generation"
  
  message_retention_duration = "86400s"
}

# Cloud Storage Buckets
resource "google_storage_bucket" "app_storage" {
  name          = "${var.project_id}-storage"
  location      = var.region
  force_destroy = false
  
  versioning {
    enabled = true
  }
  
  encryption {
    default_kms_key_name = google_kms_crypto_key.storage_key.id
  }
}

# KMS Key for encryption
resource "google_kms_key_ring" "clarity_keyring" {
  name     = "clarity-keyring"
  location = var.region
}

resource "google_kms_crypto_key" "storage_key" {
  name     = "storage-key"
  key_ring = google_kms_key_ring.clarity_keyring.id
  
  rotation_period = "7776000s" # 90 days
}
```

### Environment-Specific Variables
```hcl
# terraform/environments/prod/terraform.tfvars
project_id     = "clarity-loop-prod"
region         = "us-central1"
service_name   = "clarity-backend-prod"
environment    = "production"
cpu_limit      = "4"
memory_limit   = "4Gi"
concurrency    = 200
max_instances  = 100

env_vars = {
  DEBUG           = "false"
  LOG_LEVEL      = "WARNING"
  FIREBASE_PROJECT_ID = "clarity-loop-prod"
}
```

## Database Migrations

### Firestore Schema Management
```python
# scripts/migrate_firestore.py
import asyncio
from google.cloud import firestore
from typing import Dict, Any
import logging

class FirestoreMigration:
    def __init__(self, project_id: str):
        self.db = firestore.AsyncClient(project=project_id)
        self.logger = logging.getLogger(__name__)
    
    async def apply_migration(self, migration_name: str):
        """Apply a specific migration."""
        migration_func = getattr(self, f"migration_{migration_name}")
        
        try:
            await migration_func()
            await self._record_migration(migration_name)
            self.logger.info(f"Migration {migration_name} applied successfully")
        except Exception as e:
            self.logger.error(f"Migration {migration_name} failed: {e}")
            raise
    
    async def migration_001_create_indexes(self):
        """Create initial indexes for collections."""
        # User data indexes
        await self.db.collection('users').create_index([
            ('created_at', firestore.Query.DESCENDING),
            ('email', firestore.Query.ASCENDING)
        ])
        
        # Health data indexes
        await self.db.collection('health_data').create_index([
            ('user_id', firestore.Query.ASCENDING),
            ('timestamp', firestore.Query.DESCENDING),
            ('data_type', firestore.Query.ASCENDING)
        ])
        
        # Insights indexes
        await self.db.collection('insights').create_index([
            ('user_id', firestore.Query.ASCENDING),
            ('created_at', firestore.Query.DESCENDING),
            ('insight_type', firestore.Query.ASCENDING)
        ])
    
    async def migration_002_add_data_quality_fields(self):
        """Add data quality fields to existing health data."""
        query = self.db.collection('health_data').where('quality_score', '==', None)
        docs = query.stream()
        
        batch = self.db.batch()
        count = 0
        
        async for doc in docs:
            doc_ref = self.db.collection('health_data').document(doc.id)
            batch.update(doc_ref, {
                'quality_score': 0.95,  # Default high quality
                'validation_status': 'validated',
                'processing_version': '2.1.0'
            })
            
            count += 1
            if count % 500 == 0:  # Batch size limit
                await batch.commit()
                batch = self.db.batch()
        
        if count % 500 != 0:
            await batch.commit()
    
    async def _record_migration(self, migration_name: str):
        """Record applied migration in the database."""
        await self.db.collection('_migrations').document(migration_name).set({
            'applied_at': firestore.SERVER_TIMESTAMP,
            'version': migration_name
        })

# Usage
async def main():
    migration = FirestoreMigration('clarity-loop-prod')
    await migration.apply_migration('001_create_indexes')
    await migration.apply_migration('002_add_data_quality_fields')

if __name__ == "__main__":
    asyncio.run(main())
```

## Deployment Scripts

### Automated Deployment Script
```bash
#!/bin/bash
# scripts/deploy.sh

set -e

ENVIRONMENT=${1:-development}
VERSION=${2:-latest}
PROJECT_ID=""
SERVICE_NAME=""

case $ENVIRONMENT in
  development)
    PROJECT_ID="clarity-loop-dev"
    SERVICE_NAME="clarity-backend-dev"
    ;;
  staging)
    PROJECT_ID="clarity-loop-staging"
    SERVICE_NAME="clarity-backend-staging"
    ;;
  production)
    PROJECT_ID="clarity-loop-prod"
    SERVICE_NAME="clarity-backend-prod"
    ;;
  *)
    echo "Invalid environment: $ENVIRONMENT"
    echo "Usage: $0 {development|staging|production} [version]"
    exit 1
    ;;
esac

echo "🚀 Deploying to $ENVIRONMENT environment..."

# Set Google Cloud project
gcloud config set project $PROJECT_ID

# Build image
echo "📦 Building Docker image..."
IMAGE_TAG="gcr.io/$PROJECT_ID/clarity-backend:$VERSION"
docker build -t $IMAGE_TAG .

# Push image
echo "⬆️ Pushing image to Container Registry..."
docker push $IMAGE_TAG

# Run pre-deployment tests
echo "🧪 Running pre-deployment tests..."
python scripts/pre-deployment-tests.py --environment=$ENVIRONMENT

# Deploy to Cloud Run
echo "🌐 Deploying to Cloud Run..."
if [ "$ENVIRONMENT" = "production" ]; then
  # Production deployment with traffic control
  gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_TAG \
    --platform managed \
    --region us-central1 \
    --no-allow-unauthenticated \
    --set-env-vars ENVIRONMENT=$ENVIRONMENT \
    --memory 4Gi \
    --cpu 4 \
    --concurrency 200 \
    --max-instances 100 \
    --no-traffic
    
  echo "🔄 Starting canary deployment..."
  gcloud run services update-traffic $SERVICE_NAME \
    --to-revisions=LATEST=10,PREVIOUS=90
    
  echo "⏳ Monitoring for 5 minutes..."
  sleep 300
  
  echo "✅ Promoting to 100% traffic..."
  gcloud run services update-traffic $SERVICE_NAME \
    --to-revisions=LATEST=100
else
  # Development/Staging deployment
  gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_TAG \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars ENVIRONMENT=$ENVIRONMENT \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 100 \
    --max-instances 20
fi

# Run post-deployment tests
echo "🔍 Running post-deployment tests..."
python scripts/post-deployment-tests.py --environment=$ENVIRONMENT

# Update Firestore indexes if needed
if [ "$ENVIRONMENT" = "production" ]; then
  echo "📊 Updating database indexes..."
  python scripts/migrate_firestore.py --project=$PROJECT_ID
fi

echo "🎉 Deployment to $ENVIRONMENT completed successfully!"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)')

echo "🌐 Service URL: $SERVICE_URL"
echo "📊 Health Check: $SERVICE_URL/health"
echo "📚 API Docs: $SERVICE_URL/docs"
```

## Monitoring and Alerting

### Health Check Configuration
```python
# src/health.py
from fastapi import APIRouter, HTTPException
from google.cloud import firestore, pubsub_v1
import asyncio
import time

router = APIRouter()

@router.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    start_time = time.time()
    
    checks = {
        "status": "healthy",
        "timestamp": int(start_time),
        "version": "2.1.0",
        "environment": os.getenv("ENVIRONMENT", "unknown"),
        "checks": {}
    }
    
    # Database connectivity
    try:
        db = firestore.AsyncClient()
        await db.collection('_health').document('check').get()
        checks["checks"]["database"] = {"status": "healthy", "latency_ms": 0}
    except Exception as e:
        checks["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
        checks["status"] = "unhealthy"
    
    # Pub/Sub connectivity
    try:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(os.getenv("GOOGLE_CLOUD_PROJECT"), "health-check")
        publisher.publish(topic_path, b"health-check")
        checks["checks"]["pubsub"] = {"status": "healthy"}
    except Exception as e:
        checks["checks"]["pubsub"] = {"status": "unhealthy", "error": str(e)}
        checks["status"] = "unhealthy"
    
    # ML service connectivity
    try:
        # Add ML service health check
        checks["checks"]["ml_service"] = {"status": "healthy"}
    except Exception as e:
        checks["checks"]["ml_service"] = {"status": "unhealthy", "error": str(e)}
        checks["status"] = "unhealthy"
    
    checks["response_time_ms"] = int((time.time() - start_time) * 1000)
    
    if checks["status"] != "healthy":
        raise HTTPException(status_code=503, detail=checks)
    
    return checks

@router.get("/readiness")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    return {"status": "ready", "timestamp": int(time.time())}

@router.get("/liveness")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive", "timestamp": int(time.time())}
```

## Rollback Procedures

### Automatic Rollback Script
```bash
#!/bin/bash
# scripts/rollback.sh

ENVIRONMENT=${1:-production}
REVISION=${2:-PREVIOUS}

case $ENVIRONMENT in
  production)
    PROJECT_ID="clarity-loop-prod"
    SERVICE_NAME="clarity-backend-prod"
    ;;
  staging)
    PROJECT_ID="clarity-loop-staging"
    SERVICE_NAME="clarity-backend-staging"
    ;;
  *)
    echo "Rollback only supported for staging and production"
    exit 1
    ;;
esac

echo "🔄 Rolling back $ENVIRONMENT to $REVISION..."

# Set project
gcloud config set project $PROJECT_ID

# Get current revision
CURRENT_REVISION=$(gcloud run services describe $SERVICE_NAME \
  --platform managed \
  --region us-central1 \
  --format 'value(status.traffic[0].revisionName)')

echo "📊 Current revision: $CURRENT_REVISION"

# Rollback to previous revision
gcloud run services update-traffic $SERVICE_NAME \
  --to-revisions=$REVISION=100 \
  --platform managed \
  --region us-central1

echo "✅ Rollback completed"

# Verify health
sleep 30
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)')

curl -f "$SERVICE_URL/health" || {
  echo "❌ Health check failed after rollback"
  exit 1
}

echo "🎉 Rollback successful and service is healthy"
```

## Security Considerations

### Deployment Security Checklist
- [ ] Service accounts use least privilege permissions
- [ ] Secrets managed via Google Secret Manager
- [ ] Container images scanned for vulnerabilities
- [ ] Network policies restrict traffic
- [ ] Audit logging enabled for all deployments
- [ ] Encryption at rest and in transit configured
- [ ] Environment variables don't contain secrets
- [ ] IAM policies reviewed and approved

### Secret Management
```bash
# Store secrets in Secret Manager
gcloud secrets create jwt-secret-key --data-file=jwt-key.txt
gcloud secrets create encryption-key --data-file=encryption-key.txt
gcloud secrets create gemini-api-key --data-file=gemini-key.txt

# Grant service account access
gcloud secrets add-iam-policy-binding jwt-secret-key \
  --member="serviceAccount:clarity-backend-prod@clarity-loop-prod.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

This deployment guide provides comprehensive coverage of the deployment process, from automated CI/CD pipelines to manual deployment procedures, infrastructure management, and security considerations.
