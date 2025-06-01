# Google Cloud 2025 Best Practices

Modern Google Cloud architecture patterns and best practices for the Clarity Loop Backend.

## Cloud Run Best Practices

### Service Configuration
```yaml
# service.yaml - Modern Cloud Run configuration
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: clarity-api-gateway
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "2000m"
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "100"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
      - image: gcr.io/PROJECT/clarity-api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "2000m"
            memory: "2Gi"
        env:
        - name: PROJECT_ID
          value: "clarity-prod"
```

### Performance Optimization
```python
# Startup optimization for 2025
import asyncio
from functools import lru_cache

@lru_cache(maxsize=1)
def get_ml_model():
    """Cache ML model globally to reduce cold starts"""
    return load_pat_model()

# Use async initialization
async def lifespan(app: FastAPI):
    # Warm up critical services
    await initialize_firestore_client()
    await warm_up_vertex_ai()
    yield
    # Cleanup on shutdown

app = FastAPI(lifespan=lifespan)
```

## Firestore Advanced Patterns

### Document Design 2025
```javascript
// Optimal document structure for health data
{
  "users/{userId}": {
    "profile": {
      "created": "2024-01-15T08:00:00Z",
      "timezone": "America/New_York",
      "preferences": {...}
    },
    "analytics": {
      "latest_insight_id": "insight-20240116",
      "total_uploads": 127,
      "last_activity": "2024-01-16T07:30:00Z"
    }
  },
  
  "health_data/{userId}/daily/{date}": {
    "date": "2024-01-15",
    "summary": {
      "sleep_hours": 8.2,
      "activity_score": 0.78,
      "steps": 12450
    },
    "raw_samples": [...], // Paginated subcollection
    "pat_features": {...},
    "insights": [...]
  }
}
```

### Query Optimization
```python
# Efficient Firestore queries
async def get_user_insights_optimized(user_id: str, days: int = 7):
    """Optimized insight retrieval with proper indexing"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Composite index: user_id + created_at (descending)
    query = (db.collection('insights')
             .where('user_id', '==', user_id)
             .where('created_at', '>=', start_date)
             .order_by('created_at', direction=firestore.Query.DESCENDING)
             .limit(days))
    
    return [doc.to_dict() async for doc in query.stream()]
```

## Pub/Sub Modern Patterns

### Message Schema 2025
```python
# CloudEvents compliant message format
@dataclass
class HealthKitMessage:
    specversion: str = "1.0"
    type: str = "com.clarity.healthkit.uploaded"
    source: str = "clarity-api-gateway"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    datacontenttype: str = "application/json"
    data: Dict[str, Any] = field(default_factory=dict)
```

### Push Subscription with Cloud Run
```python
# Modern push endpoint
@app.post("/pubsub/healthkit-process")
async def process_healthkit_message(request: Request):
    """Cloud Pub/Sub push endpoint with proper error handling"""
    try:
        envelope = await request.json()
        
        # Verify Pub/Sub token
        token = request.headers.get('authorization', '').replace('Bearer ', '')
        await verify_pubsub_token(token)
        
        # Decode message
        message_data = base64.b64decode(envelope['message']['data'])
        message = json.loads(message_data)
        
        # Process with idempotency
        message_id = envelope['message']['messageId']
        if await is_message_processed(message_id):
            return {"status": "already_processed"}
        
        result = await process_healthkit_data(message)
        await mark_message_processed(message_id)
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        logger.error(f"Pub/Sub processing failed: {e}")
        # Return 4xx for permanent failures, 5xx for retries
        raise HTTPException(status_code=500, detail="Processing failed")
```

## IAM & Security 2025

### Service Account Patterns
```bash
# Minimal privilege service accounts
gcloud iam service-accounts create clarity-api-gateway \
  --display-name="Clarity API Gateway"

# Specific permissions only
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:clarity-api-gateway@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/firestore.user"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:clarity-api-gateway@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/pubsub.publisher"
```

### Workload Identity Federation
```yaml
# Modern authentication without service account keys
apiVersion: v1
kind: ServiceAccount
metadata:
  name: clarity-api-gateway
  annotations:
    iam.gke.io/gcp-service-account: clarity-api-gateway@PROJECT_ID.iam.gserviceaccount.com
```

## Monitoring & Observability

### Custom Metrics 2025
```python
# OpenTelemetry integration
from opentelemetry import trace, metrics
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter

# Custom metrics
meter = metrics.get_meter(__name__)
health_upload_counter = meter.create_counter(
    "clarity_health_uploads_total",
    description="Total HealthKit uploads processed"
)

processing_duration = meter.create_histogram(
    "clarity_processing_duration_seconds",
    description="Time spent processing health data"
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    # Record metrics
    processing_duration.record(
        time.time() - start_time,
        {"endpoint": request.url.path, "status": response.status_code}
    )
    
    return response
```

### SLI/SLO Dashboard
```yaml
# monitoring/slo-config.yaml
displayName: "Clarity Backend SLOs"
serviceLevelObjectives:
- serviceLevelIndicator:
    requestBased:
      distributionCut:
        range:
          max: 200  # 200ms response time
  goal: 0.95  # 95th percentile
  rollingPeriod: "86400s"  # 24 hours
```

## Vertex AI Integration

### Model Deployment 2025
```python
# Modern Vertex AI client
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

async def get_pat_predictions(features: Dict[str, Any]) -> Dict[str, Any]:
    """Get PAT model predictions with proper error handling"""
    client = aiplatform.gapic.PredictionServiceClient()
    
    endpoint = f"projects/{PROJECT_ID}/locations/us-central1/endpoints/{ENDPOINT_ID}"
    
    instances = [predict.instance.to_value(features)]
    parameters = predict.params.to_value({})
    
    try:
        response = await client.predict(
            endpoint=endpoint,
            instances=instances,
            parameters=parameters
        )
        
        return {
            "predictions": response.predictions,
            "model_version": response.model_version_id,
            "deployed_model_id": response.deployed_model_id
        }
        
    except Exception as e:
        logger.error(f"Vertex AI prediction failed: {e}")
        # Fallback to cached predictions if available
        return await get_cached_prediction(features)
```

## Cost Optimization 2025

### Resource Management
```python
# Smart scaling based on usage patterns
@app.on_event("startup")
async def configure_scaling():
    """Configure intelligent scaling policies"""
    
    # Scale down during low usage hours
    if is_low_traffic_period():
        await set_min_instances(0)
    else:
        await set_min_instances(2)
    
    # Monitor queue depth for ML service scaling
    queue_depth = await get_pubsub_queue_depth()
    if queue_depth > 100:
        await scale_ml_service(min_instances=5)
```

### Storage Lifecycle
```bash
# Automatic data lifecycle management
gsutil lifecycle set - gs://clarity-health-data <<EOF
{
  "rule": [
    {
      "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
      "condition": {"age": 30}
    },
    {
      "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
      "condition": {"age": 90}
    },
    {
      "action": {"type": "Delete"},
      "condition": {"age": 2555}  # 7 years for HIPAA
    }
  ]
}
EOF
```

## Terraform Configuration

### Modern Infrastructure as Code
```hcl
# terraform/main.tf - 2025 patterns
resource "google_cloud_run_v2_service" "api_gateway" {
  name     = "clarity-api-gateway"
  location = var.region
  
  template {
    scaling {
      min_instance_count = 1
      max_instance_count = 100
    }
    
    containers {
      image = var.api_image
      
      resources {
        limits = {
          cpu    = "2000m"
          memory = "2Gi"
        }
      }
      
      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }
    }
    
    service_account = google_service_account.api_gateway.email
  }
  
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
}

# Firestore with backup
resource "google_firestore_database" "clarity" {
  project     = var.project_id
  name        = "clarity-prod"
  location_id = "us-central"
  type        = "FIRESTORE_NATIVE"
  
  point_in_time_recovery_enablement = "POINT_IN_TIME_RECOVERY_ENABLED"
  delete_protection_state           = "DELETE_PROTECTION_ENABLED"
}
```

---

**Updated for 2025**: Modern Google Cloud services, security patterns, and cost optimization
