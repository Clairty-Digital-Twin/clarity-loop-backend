Apple HealthKit Pipeline Integration – Implementation Guide

This guide provides a step-by-step blueprint for integrating an Apple HealthKit data pipeline into the clarity-loop-backend monorepo. We follow a clean, modular architecture that separates ingestion, analysis, and insight-generation concerns ￼. Each service runs independently (as separate Cloud Run services in production) but shares common domain models and utilities within the monorepo for maintainability. All code examples are in Python (using FastAPI, PyTorch, etc.) and aligned with the project’s Clean Architecture practices ￼. By the end, a developer can implement the pipeline from data upload to AI-generated insight with minimal guesswork.

1. Monorepo Directory Structure

Organize the repository into clear modules for each microservice and shared libraries. Below is a suggested top-level structure:

clarity-loop-backend/
├── ingestion_service/        # FastAPI app for data ingestion (HealthKit uploads)
│   ├── app.py                # FastAPI entrypoint for ingestion service
│   ├── routers/healthkit.py  # API router for HealthKit upload endpoint
│   ├── models/healthkit.py   # Pydantic models for HealthKit data
│   └── ... other ingestion-specific modules
├── analysis_service/         # Cloud Run service for data processing & ML analysis
│   ├── main.py               # FastAPI entrypoint for Pub/Sub push subscription
│   ├── processors/           # Modular signal processor classes
│   │   ├── cardio_processor.py       # e.g., CardioProcessor for HR/HRV
│   │   ├── respiration_processor.py  # e.g., RespirationProcessor for RR/SpO₂
│   │   └── ... other modality processors
│   ├── ml/
│   │   ├── preprocessing.py          # Preprocessing functions (outlier removal, etc.)
│   │   ├── fusion_transformer.py     # Multimodal fusion Transformer model
│   │   ├── pat_model_stub.py         # Stubbed PAT model for local dev
│   │   └── models/... (optional actual model weights or loading code)
│   └── services/analysis_pipeline.py # Orchestrator to run preprocessing, processors, fusion
├── insight_generator/        # Service for LLM-based insight generation
│   ├── main.py               # FastAPI entrypoint for Pub/Sub push subscription
│   ├── gemini_client.py      # Wrapper for Vertex AI Gemini 2.5 API calls
│   └── models/insight_models.py # (optional) Pydantic models for insight request/response
├── shared/                   # Shared domain models, interfaces, and utilities
│   ├── models/common.py      # Common Pydantic models (e.g. base HealthKit sample schema)
│   ├── core/pubsub.py        # Pub/Sub helper (publishing messages)
│   ├── core/storage.py       # GCS helper (upload/download)
│   └── utils/...             # Misc utilities (auth, config, etc.)
├── tests/                    # Unit and integration tests
│   ├── unit/...
│   └── integration/...
├── Dockerfile.ingestion      # Dockerfile for ingestion_service
├── Dockerfile.analysis       # Dockerfile for analysis_service
├── Dockerfile.insight        # Dockerfile for insight_generator
├── docker-compose.yml        # Optional: compose file for local multi-service setup
└── Makefile                  # Dev scripts (build, test, deploy)

Rationale: This structure cleanly separates concerns: each service has its own module with its FastAPI app and related code, while shared/ holds cross-cutting models and utils (consistent with the project’s layering conventions ￼). For example, the HealthKit Pydantic schemas and validation logic can live in a shared models file (or within the ingestion service if they’re only used there), but the core data preprocessing and ML logic reside in the analysis service module (not in the API layer) ￼. This keeps the codebase modular and testable.

The ingestion_service handles HTTP requests from clients and immediate validation/storage, the analysis_service performs heavy data processing and ML inference asynchronously, and the insight_generator service encapsulates LLM prompt construction and response parsing. Shared components (like domain models or utility functions for GCS, Pub/Sub, config, etc.) prevent duplication. This layout follows Clean Architecture: the FastAPI routers act as interface adapters (controllers), calling into service or processor classes that contain the business logic ￼.

2. Implementation by Component

Below we break down each major component with code examples and file placements. These code snippets illustrate the expected functionality and how pieces fit together:

2.1 FastAPI Ingestion Service – HealthKit Upload Endpoint

The ingestion service exposes a FastAPI endpoint (e.g. POST /api/v1/healthkit/upload) to receive batches of HealthKit data from clients. It authenticates the request, validates the payload, stores the raw data to cloud storage, and enqueues a Pub/Sub message for asynchronous processing ￼. The client gets an immediate acknowledgment without waiting for analysis.

File: ingestion_service/routers/healthkit.py (register this router with the FastAPI app in app.py).

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from datetime import datetime
import uuid, os, json
from google.cloud import storage
from ingestion_service.models.healthkit import HealthKitUploadRequest
from shared.core.pubsub import publish_message  # Pub/Sub helper
from shared.utils.auth import verify_firebase_token  # (assume we have a Firebase auth util)

router = APIRouter(prefix="/api/v1/healthkit", tags=["HealthKit"])
auth_scheme = HTTPBearer()

@router.post("/upload", status_code=202)
async def upload_healthkit_data(request: HealthKitUploadRequest,
                                 token: str = Depends(auth_scheme)):
    """Receive a batch of HealthKit data, store it, and queue for analysis."""
    # 1. Authenticate and authorize the request
    user_claims = await verify_firebase_token(token.credentials)
    if user_claims.get("uid") != request.user_id:
        raise HTTPException(status_code=403, detail="Cannot upload data for a different user")
    # 2. Generate a unique upload ID for tracking
    upload_id = f"{request.user_id}-{uuid.uuid4().hex}"
    # 3. Save raw data to Google Cloud Storage (for durable storage of large payloads)
    storage_client = storage.Client()
    bucket_name = os.getenv("HEALTHKIT_RAW_BUCKET", "healthkit-raw-data")
    bucket = storage_client.bucket(bucket_name)
    blob_path = f"{request.user_id}/{upload_id}.json"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(request.json())  # store the raw JSON as-is in GCS
    # 4. Publish a Pub/Sub message to trigger analysis (async processing)
    message = {
        "upload_id": upload_id,
        "user_id": request.user_id,
        "gcs_path": f"gs://{bucket_name}/{blob_path}"
    }
    await publish_message(topic=os.getenv("PUBSUB_HEALTH_DATA_TOPIC", "health-data-upload"),
                           message=message)
    # 5. Return immediate ACK with tracking info
    return {
        "upload_id": upload_id,
        "status": "queued",
        "queued_at": datetime.utcnow().isoformat(),
        "samples_received": {
            "quantity_samples": len(request.quantity_samples),
            "category_samples": len(request.category_samples),
            "workouts": len(request.workouts)
        }
    }

In this snippet, we define a /upload route that expects a HealthKitUploadRequest (a Pydantic model describing the payload, e.g. lists of samples and workouts). The handler verifies the Firebase Auth token and ensures the user_id in the request matches the authenticated user (preventing cross-user data access) ￼ ￼. Next, it creates a unique upload_id and writes the raw JSON to GCS for safekeeping. Storing raw data in a Cloud Storage bucket (with proper encryption) ensures we don’t lose detailed data and avoids hitting Firestore size limits for large time-series ￼. The code then uses a shared publish_message utility to send a Pub/Sub message on the health-data-upload topic, containing the upload_id, user_id, and a pointer to the GCS file. This enqueues the data for downstream processing. We respond with a 202 Accepted status, returning the upload_id and a summary of received samples. The client can use this ID to poll a status endpoint or simply wait for a realtime update when insights are ready.

Key points:
 • We do minimal work synchronously: just auth, validation, storage, and message enqueue. This keeps the upload API fast (acknowledge immediately) ￼ ￼.
 • The heavy lifting (parsing and analyzing data) is deferred to background services. This decoupled design follows an event-driven approach – ingestion just publishes an event and doesn’t block on processing ￼.
 • Using Pub/Sub ensures reliable, at-least-once delivery of the processing task. If the analysis service is busy or down, the message will be retried, and the upload is not lost ￼.
 • The publish_message function (in shared/core/pubsub.py) wraps the Google Cloud Pub/Sub publisher client. It likely serializes the message dict to JSON and publishes it on the given topic asynchronously. For example:

# Inside shared/core/pubsub.py

from google.cloud import pubsub_v1
import json
publisher = pubsub_v1.PublisherClient()
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

async def publish_message(topic: str, message: dict):
    topic_path = publisher.topic_path(PROJECT, topic)
    data = json.dumps(message).encode("utf-8")
    future = publisher.publish(topic_path, data)
    await future  # ensure publish completes

This utility hides the details of connecting to Pub/Sub. (If using the Pub/Sub emulator locally, the environment variable PUBSUB_EMULATOR_HOST will be set so the client publishes to the emulator instead of GCP – more on that in section 4.)

2.2 Pub/Sub Subscription – Processing Task Consumer

The analysis service runs as a Cloud Run service subscribed to the Pub/Sub topic (via a push subscription). Pub/Sub will HTTP POST the message to an endpoint we define (e.g. /process-task). Our task is to receive the message, acknowledge it, and kick off the data processing pipeline ￼.

File: analysis_service/main.py (or a router in analysis service FastAPI app)

from fastapi import FastAPI, Request, Header, HTTPException
import base64, os, json
from google.cloud import storage
from analysis_service.services.analysis_pipeline import run_analysis_pipeline
from shared.core.pubsub import publish_message  # to publish next-stage message

app = FastAPI()

@app.post("/process-task", status_code=204)
async def process_task(request: Request, authorization: str = Header(None)):
    """Handle Pub/Sub push subscription for analysis tasks."""
    # 1. Verify the request is from Pub/Sub by checking the OIDC JWT (if in production)
    if os.getenv("ENVIRONMENT") == "production":
        token = authorization.split[" "](1) if authorization else None
        try:
            # Verify Google-signed JWT (this requires Google's certs; could use google.auth.jwt)
            # For brevity, assume a utility exists:
            verify_pubsub_jwt(token, audience=os.getenv("PUBSUB_PUSH_AUDIENCE"))
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid Pub/Sub token")
    # 2. Parse Pub/Sub message format (base64 data)
    body = await request.json()
    message_data = base64.b64decode(body["message"]["data"]).decode("utf-8")
    task = json.loads(message_data)  # contains user_id, upload_id, gcs_path, etc.
    # 3. Download raw data from GCS using the path
    storage_client = storage.Client()
    gcs_uri: str = task["gcs_path"]  # e.g. "gs://healthkit-raw-data/uid/uploadid.json"
    bucket_name = gcs_uri.split["/"](2)
    blob_path = "/".join(gcs_uri.split["/"](3:))
    raw_json = storage_client.bucket(bucket_name).blob(blob_path).download_as_text()
    health_data = json.loads(raw_json)
    # 4. Run the analysis pipeline on the data (preprocessing, feature extraction, fusion)
    results = run_analysis_pipeline(health_data)
    # 5. Publish an insight-generation message for the next stage
    insight_topic = os.getenv("PUBSUB_INSIGHTS_TOPIC", "insight-request")
    insight_msg = {
        "user_id": task["user_id"],
        "upload_id": task["upload_id"],
        "analysis_results": results  # e.g. fused feature vector or stats
    }
    await publish_message(topic=insight_topic, message=insight_msg)
    # 6. (Optionally, store results in Firestore or another storage if needed before LLM)
    return {"status": "acknowledged"}

Explanation: This is the analysis service entrypoint that Pub/Sub calls when a new upload message is available ￼ ￼. We defined a /process-task endpoint that expects Pub/Sub to POST a JSON with a base64-encoded message. The handler first (in production) verifies the OIDC JWT in the Authorization header to ensure the request truly comes from Pub/Sub (Google Pub/Sub can be configured to attach a signed token; our code would use Google’s public certs to verify it, or leverage a library) ￼. For local development, this check can be skipped or disabled (as indicated by checking an ENVIRONMENT flag above).

Next, we parse the incoming Pub/Sub message: Pub/Sub push sends a JSON with a "message" field containing data (base64 string) and attributes. We decode the data payload to get our original message dict (with user_id, upload_id, gcs_path, etc.). Using the gcs_path, we instantiate a GCS client to download the raw JSON that was previously uploaded ￼. Now we have the full health data payload in memory as health_data (likely a dict with lists of samples).

We then call the core analysis pipeline logic via run_analysis_pipeline(health_data). This function (which we’ll outline below) orchestrates preprocessing each metric and running the processor classes and fusion model to produce a unified result. The output results might be a dictionary or vector representing the user’s health state or key extracted features.

Finally, we publish a new Pub/Sub message to trigger the insight generator service. We use another topic (e.g. "insight-request") to carry the analysis results forward ￼. The message includes user_id and upload_id (so the insight service knows where to store results and for whom) and the computed features/embeddings (or a reference to them). We do not include raw personal health data at this stage – only derived, non-PHI insights or pointers – to minimize sensitive info flowing into the LLM stage. The insight generator will use this to form an AI prompt.

By returning status code 204 (No Content) or a simple JSON, we signal that the message was processed successfully. If this handler returns a 2xx response, Pub/Sub considers the push delivery acknowledged. If it fails (500 or timeout), Pub/Sub will retry the delivery per its at-least-once guarantee.

2.3 Data Preprocessing – Heart Rate Example

Before feeding data into models or calculating features, we apply preprocessing to clean and normalize each HealthKit metric. Let’s demonstrate this for Heart Rate (HR) data, which typically comes as irregularly timed samples (e.g. every 5 seconds during exercise, every few minutes otherwise). Our goal is to produce a clean, uniformly sampled time series (e.g. 1-minute intervals) with noise reduced and values normalized, as recommended by domain best practices ￼.

File: analysis_service/ml/preprocessing.py

import numpy as np
import pandas as pd

def preprocess_heart_rate(timestamps, values):
    """
    Clean and normalize heart rate time-series.
    - timestamps: list of datetime or timestamps for each HR sample
    - values: list of heart rate values (BPM) corresponding to timestamps
    Returns a list of HR values resampled to 1-min intervals, smoothed and normalized.
    """
    # 1. Create a pandas Series for convenient resampling
    ts = pd.Series(values, index=pd.to_datetime(timestamps))
    # 2. Resample to 1-minute frequency, computing the mean BPM for each minute
    hr_per_min = ts.resample('1T').mean()
    # 3. Fill short gaps by interpolation (up to 2 consecutive minutes)
    hr_interpolated = hr_per_min.interpolate(limit=2, limit_direction='forward')
    # 4. Remove obvious outliers or invalid data
    hr_interpolated = hr_interpolated.mask((hr_interpolated <= 0) | (hr_interpolated > 220), np.nan)
    #    (HR >220 BPM or <=0 are likely erroneous unless sustained; we mark them NaN)
    # 5. Apply a short moving average to smooth high-frequency noise (window = 3 minutes)
    hr_smoothed = hr_interpolated.rolling(window=3, min_periods=1, center=True).mean()
    # 6. Final gap fill: after smoothing, interpolate any remaining NaNs (if small gaps remain)
    hr_smoothed = hr_smoothed.interpolate(limit=1)
    # 7. Normalize the heart rate values.
    # Option A: Person-specific normalization using resting HR baseline
    resting_est = np.nanpercentile(hr_smoothed, 5)  # approximate resting HR (5th percentile)
    max_est = np.nanmax(hr_smoothed)
    normalized = (hr_smoothed - resting_est) / (max_est - resting_est) if max_est > resting_est else hr_smoothed
    # Option B: Population z-score normalization (e.g., mean ~70, std ~15 BPM)
    # mean_hr, std_hr = 70.0, 15.0
    # normalized = (hr_smoothed - mean_hr) / std_hr
    # 8. Replace any residual NaNs with 0 (or we could carry them through if model can handle masks)
    final_series = normalized.fillna(0.0)
    return final_series.tolist()

In this example, we use pandas for clarity, but one could implement similar logic with pure Python if needed. Here’s what we do, aligned with the plan:
 • Resampling: Convert irregular HR samples to a uniform timeline (1-minute bins) by averaging values within each minute ￼. This aligns heart rate with other per-minute signals (like steps, respiratory rate, etc.).
 • Outlier removal: HR values of 0 or unrealistically high values (e.g. a single reading of 250 BPM) are replaced with NaN (treated as missing) ￼. We assume any instantaneous spike beyond human limits is sensor error unless it persists.
 • Interpolation: Short gaps (we allow up to 2 minutes missing) are filled linearly (or forward-filled) because heart rate generally doesn’t jump drastically in 1-2 minutes if a gap is small. Larger gaps (e.g. no readings for an hour) remain NaN – we won’t fabricate long stretches of data.
 • Smoothing: We apply a mild smoothing (3-minute moving average) to filter out jitter and PPG noise ￼. This helps remove high-frequency oscillations that likely aren’t meaningful changes (especially if device reports HR with some noise).
 • Normalization: Finally, we normalize the series. In the code above we show two approaches:
 • A personal baseline normalization: subtract an estimated resting HR (low percentile of the week’s data) and divide by the user’s HR range. This yields a number roughly in [-1, 1] range representing how elevated the heart rate is relative to their baseline.
 • Alternatively, a population z-score: subtract a typical mean (≈70 BPM) and divide by std (≈15) to get standard scores. This might be useful if comparing across population norms.
Either way, normalization ensures the values are scaled for the model. For example, one could use z-scores so that an average HR ~0, high HR yields +2 or +3, etc., or 0-1 scaling relative to resting HR. The blueprint specifically suggests using either the user’s resting HR or global stats for normalization ￼. We fill any remaining missing values with 0 (assuming 0 meaning “no data” or it could be handled by model masking – here we choose simplicity).

This function would be part of a suite of preprocessing utilities (e.g., you’d have similar preprocess_hrv, preprocess_respiration, etc., each tailored to the data characteristics). The analysis pipeline will call these functions to clean raw data before passing it to processor classes. By keeping these in analysis_service/ml/preprocessing.py, we encapsulate “data cleaning” logic separately from the API and business rules, aligning with Clean Architecture (this is part of the “ML processing” layer, not the web layer) ￼.

2.4 Processor Classes – Modular Signal Processing

After preprocessing, each modality’s data goes into a processor class that extracts features or embeddings from that modality. Processors encapsulate the logic for a specific health domain (cardio, respiratory, activity, etc.) and present a common interface (e.g. process(data) -> vector). This modular design means each processor can be developed and tested independently ￼.

For instance, the CardioProcessor might take cleaned heart rate and HRV time series and output a feature vector capturing cardiovascular fitness or stress. We will create one processor per modality or group of related modalities:
 • CardioProcessor – handles heart rate (HR) and heart rate variability (HRV) metrics ￼.
 • RespirationProcessor – handles respiratory rate (RR) and oxygen saturation (SpO₂).
 • ActigraphyProcessor – handles activity data (steps, exercise minutes) – likely by invoking the PAT model for actigraphy (or proxy actigraphy features).
 • Other processors – e.g., SleepProcessor, BloodPressureProcessor, etc., as needed for additional metrics ￼.

Each processor will output a fixed-size vector (say 8, 16 dimensions) summarizing that modality. These outputs will later be fused.

File: analysis_service/processors/cardio_processor.py

import numpy as np
from analysis_service.ml.preprocessing import preprocess_heart_rate, preprocess_hrv
from analysis_service.models.pat_model_stub import PatModelStub  # if using PAT for activity

class CardioProcessor:
    """Extract features from heart rate and HRV series."""
    def __init__(self):
        # Example: perhaps load a small ML model or just prepare to compute stats.
        # (No heavy model here; just use statistical features for now.)
        self.hrv_required = True  # this processor expects HRV as well if available

    def process(self, hr_timestamps, hr_values, hrv_timestamps=None, hrv_values=None):
        # 1. Preprocess the raw heart rate and HRV data
        hr_clean = preprocess_heart_rate(hr_timestamps, hr_values)           # list of per-minute HR
        hrv_clean = preprocess_hrv(hrv_timestamps, hrv_values) if (hrv_values and self.hrv_required) else None
        hr_array = np.array(hr_clean, dtype=float)
        hrv_array = np.array(hrv_clean, dtype=float) if hrv_clean is not None else None
        # 2. Compute statistical features from HR/HRV over the period (e.g., one week)
        avg_hr = float(np.nanmean(hr_array))
        max_hr = float(np.nanmax(hr_array))
        resting_hr = float(np.nanpercentile(hr_array, 5))
        std_hr = float(np.nanstd(hr_array))
        # Example HRV features if available
        if hrv_array is not None:
            avg_hrv = float(np.nanmean(hrv_array))
            std_hrv = float(np.nanstd(hrv_array))
        else:
            avg_hrv, std_hrv = 0.0, 0.0
        # 3. Optionally, use a small learned model for capturing patterns (placeholder)
        # For example, we could feed the HR series into a tiny neural network to get an embedding.
        # Here we'll skip and just use stats.
        # 4. Assemble the feature vector
        features = [
            avg_hr, max_hr, resting_hr, std_hr,
            avg_hrv, std_hrv
        ]
        return features  # e.g. a list of 6 features

This CardioProcessor.process() takes raw HR and HRV data (with timestamps) and returns a feature vector. We first call our preprocessing functions to get cleaned, aligned time series (minute-by-minute arrays). Then we compute a few illustrative features:
 • Average HR over the week, maximum HR observed, an estimate of resting HR, and HR variability (std) over the week.
 • For HRV (if provided), average and std of HRV (SDNN) over the period.

In a more advanced scenario, as noted in design discussions, we might replace or augment these stats with an ML model that processes the time series (e.g., a 1-D CNN or transformer that encodes the entire HR sequence) ￼. For example, one could implement a small time-series encoder to capture circadian patterns or day-night variation in HR. Initially, however, simple summary features are easier to validate. The output is a fixed-length list of floats (six in this case).

We would have similar classes for other modalities:
 • RespirationProcessor.process(rr_series, spo2_series): compute features like mean resting respiratory rate, min SpO₂, etc.
 • ActigraphyProcessor.process(step_vector): possibly call the PAT model to get the 128-dim embedding for activity/sleep ￼. If PAT is heavy, we might have this processor call an external service or use a stub (see PAT stub below).
 • SleepProcessor.process(sleep_stages or sleep_duration): compute sleep quality metrics.
 • etc.

All processors share a common interface (.process() returning a vector) so that the analysis pipeline can treat them uniformly ￼. They can live in a package like analysis_service/processors/ for organization. This design follows the Single Responsibility Principle: each processor focuses on one domain of health data ￼. It also sets us up to potentially scale out – for example, if needed, each processor could be its own microservice or thread. But within this service, we typically just call them sequentially.

The analysis pipeline code (e.g., in analysis_service/services/analysis_pipeline.py) will do something like:

def run_analysis_pipeline(health_data: dict) -> dict:
    results = {}
    # If heart rate data present:
    if health_data.get("heart_rate_samples"):
        cardio = CardioProcessor()
        hr_ts, hr_vals = extract_timeseries(health_data["heart_rate_samples"])
        hrv_ts, hrv_vals = extract_timeseries(health_data.get("hrv_samples", []))
        results["cardio"] = cardio.process(hr_ts, hr_vals, hrv_ts, hrv_vals)
    # If respiration data present:
    if health_data.get("respiratory_samples"):
        resp = RespirationProcessor()
        # ... similar pattern ...
        results["respiratory"] = resp.process(...)
    # etc. for other modalities
    # After processing all, results now contains feature vectors per domain.
    # Next step: fuse them into one unified vector:
    fusion_model = FusionTransformer(input_dims={
        "cardio": len(results.get("cardio", [])),
        "respiratory": len(results.get("respiratory", [])),
        # ... other modalities
    })
    # Prepare input by converting lists to tensors:
    fused_input = { name: torch.tensor([vals], dtype=torch.float32)  # batch=1
                    for name, vals in results.items() }
    fused_vector = fusion_model(fused_input)  # shape [batch, embed_dim]
    results["fused_embedding"] = fused_vector.detach().numpy().tolist()[0]
    return results

(This is pseudo-code to show where the processors fit in the pipeline.) The idea is: for each modality present in the upload, we invoke its processor to get a feature vector. Then we feed the collection of modality vectors into the fusion model to get one unified embedding ￼. We include that fused result in the output (along with maybe the intermediate features if needed).

2.5 Fusion Transformer – Multimodal Fusion Head

To combine multiple modality-specific outputs (e.g. the cardio features, actigraphy embedding, respiration features, etc.), we use a lightweight Transformer encoder model. The transformer allows each modality’s representation to attend to others, producing an integrated view of the user’s health state ￼. We designate a special [CLS] token whose output embedding will serve as the unified health state vector ￼. This approach is more flexible and powerful than simple concatenation, as the model can learn cross-modal interactions (e.g. high HR + poor sleep might interact to indicate stress).

File: analysis_service/ml/fusion_transformer.py

import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    def __init__(self, modality_dims: dict[str, int], embed_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2):
        """
        modality_dims: dict mapping modality name -> dimension of its feature vector.
        embed_dim: size to project each modality to (and size of transformer embeddings).
        """
        super().__init__()
        # Linear projection for each modality to a common embedding dimension
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, embed_dim) for name, dim in modality_dims.items()
        })
        # Learnable [CLS] token embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                  dim_feedforward=embed_dim * 4, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, inputs: dict[str, torch.Tensor]):
        # Expect inputs as { modality_name: tensor of shape (batch, dim) }
        batch_size = None
        embeddings = []
        for name, x in inputs.items():
            if batch_size is None:
                batch_size = x.size(0)
            # Project modality feature vector to embed_dim
            # After projection, shape: (batch, embed_dim)
            h = self.proj[name](x)
            # Add a modality token dimension: (batch, 1, embed_dim)
            embeddings.append(h.unsqueeze(1))
        # Concatenate all modality tokens
        modality_seq = torch.cat(embeddings, dim=1)  # shape: (batch, M, embed_dim) where M = #modalities
        # Prepend the CLS token to sequence
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)  # (batch, 1, embed_dim)
        sequence = torch.cat([cls_tokens, modality_seq], dim=1)  # (batch, M+1, embed_dim)
        # Transformer expects shape (seq_len, batch, embed_dim), so transpose:
        sequence = sequence.transpose(0, 1)  # (M+1, batch, embed_dim)
        # Apply Transformer encoder
        encoded = self.transformer(sequence)  # (M+1, batch, embed_dim)
        # Take the output corresponding to CLS token (first token)
        cls_output = encoded[0]  # shape: (batch, embed_dim)
        return cls_output  # unified embedding for each batch element

How it works: We initialize the transformer with a projection layer for each modality’s input vector. For example, if CardioProcessor gives 6 features and RespirationProcessor gives 4, we might choose embed_dim=16, and have linear layers mapping 6→16 and 4→16. This ensures each modality is represented in the same vector space and one modality’s larger feature count doesn’t overpower others simply by length ￼. We also create a learnable cls_token (initialized as zeros here) which the model will learn to use as a global representation.

In forward, we take each modality’s input tensor, project it, and append it to a list. We then concatenate those along a new sequence dimension, and prepend a copy of cls_token at the start for each batch. The sequence length is (number_of_modalities + 1). We feed this through nn.TransformerEncoder, which applies self-attention across all tokens (modalities and the CLS token). The CLS token’s output (cls_output) now contains information aggregated from all modality embeddings via attention ￼. We return this as the fused health state vector (e.g. 64-dim if embed_dim=64).

We keep the model small: e.g., num_layers=1 or 2 and embed_dim maybe 64 or 128, with a few attention heads ￼ ￼. Because the number of tokens is small (we might have 3–6 modalities), even a single transformer layer can mix the information effectively. If we have missing modalities for a user, one approach is to either omit that token from inputs (so the sequence is shorter) or input a special “empty” vector (e.g. all zeros) – the design can handle either by adjusting the modality_dims and sequence accordingly.

Note: Training this fusion model from scratch without supervision is an open question. Initially, we might not have labels to train it, but even an untrained transformer can serve as a sophisticated pooling mechanism or we could simply treat the output as a concatenation (the linear projections plus maybe an identity attention layer). Over time, if we accumulate outcomes (like health scores or events), we could fine-tune this fusion on those targets ￼. For now, we include it for architectural completeness – it prepares the data in a form the LLM can consume (or could be fed into further ML tasks).

The unified embedding output by FusionTransformer (let’s say a 64-d vector) represents the user’s overall state in that upload ￼ ￼. This is what we’ll send to the LLM service to generate insights.

2.6 LLM Insight Generation – Gemini 2.5 Integration

The insight generator service uses the Vertex AI Gemini 2.5 large language model to turn the numerical analysis results into a meaningful natural-language insight for the user ￼. This involves prompt engineering and parsing the LLM’s output. We design this service to be stateless and lightweight: it just receives a Pub/Sub message (with the fused vector and maybe some summary stats), crafts a prompt, calls the Vertex AI API, and returns/stores the result ￼.

File: insight_generator/gemini_client.py

import os, json

# Google Vertex AI SDK (assumed installed)

from vertexai import language_models, init

class GeminiClient:
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.model = None
    async def initialize(self):
        # Initialize Vertex AI context and load the Gemini model
        init(project=self.project_id, location=self.location)
        # Load the Gemini 2.5 Pro model (assuming appropriate model name or alias)
        self.model = language_models.TextGenerationModel.from_pretrained("gemini-2.5-pro")
    async def generate_insight(self, analysis_results: dict, user_context: str = "") -> dict:
        # Ensure model is initialized
        if self.model is None:
            await self.initialize()
        # 1. Construct the LLM prompt based on analysis_results
        prompt = self._create_prompt(analysis_results, user_context)
        # 2. Call the model to generate content
        response = self.model.predict(prompt, temperature=0.3, max_output_tokens=1024, top_p=0.8)
        output_text = response.text
        # 3. Parse the LLM output expecting JSON
        try:
            insight = json.loads(output_text)
        except json.JSONDecodeError:
            # Fallback: LLM did not return valid JSON, handle gracefully
            insight = {
                "narrative": output_text.strip(),
                "key_insights": [output_text.strip()[:100]],  # just put part of text as an insight
                "recommendations": [],
                "confidence_score": None
            }
        # 4. Attach any metadata
        insight["model"] = "Gemini2.5"
        return insight

    def _create_prompt(self, results: dict, context: str) -> str:
        """Format the analysis results into a prompt asking for health insight in JSON."""
        # Example prompt template:
        lines = []
        # Include key metrics:
        if "cardio" in results:
            # Suppose cardio result is a vector [avg_hr, max_hr, resting_hr, std_hr, avg_hrv, std_hrv]
            avg_hr, max_hr, resting_hr, _, avg_hrv, _ = results["cardio"]
            lines.append(f"- Average Heart Rate: {avg_hr:.1f} bpm (resting ~{resting_hr:.1f} bpm, peak {max_hr:.1f} bpm)")
            if avg_hrv: 
                lines.append(f"- Average HRV (SDNN): {avg_hrv:.1f} ms")
        if "respiratory" in results:
            # e.g. [avg_rr, std_rr, avg_spo2]
            avg_rr = results["respiratory"][0]
            lines.append(f"- Average Respiratory Rate: {avg_rr:.1f} breaths/min")
        if "actigraphy" in results:
            # If we have an actigraphy summary or PAT embedding, perhaps include a summary stat
            activity_score = results["actigraphy"].get("activity_level", None)
            if activity_score:
                lines.append(f"- Activity Level Score: {activity_score:.1f}/10")
        # ... handle other modality outputs
        metrics_summary = "\n".join(lines)
        context_note = f"User context: {context}" if context else "User context: None"
        # Prompt template instructing the model to output JSON
        prompt = (
            "You are a health AI assistant. I will provide you with processed health data for a user.\n"
            "Data:\n"
            f"{metrics_summary}\n"
            f"{context_note}\n\n"
            "Generate a brief insight report summarizing the user's health and give recommendations. "
            "Respond in JSON format with the keys: narrative, key_insights, recommendations, confidence_score."
        )
        return prompt

We use the official Vertex AI SDK (vertexai) to interact with the model. The GeminiClient manages initialization and generation:
 • Model Initialization: We call vertexai.init() with the GCP project and location, then load the Gemini model. In production, project_id would be our cloud project; for dev, we might use a dummy project or a mock.
 • Prompt Construction: The_create_prompt method takes the analysis results (the output from the analysis service, e.g. containing things like average HR, etc.) and formats a prompt string. We include the key metrics in a readable form (e.g. bullet points of “Average Heart Rate: 72 bpm…”), plus any additional context (e.g. the user’s demographic or a note like “recovering from flu” if available). We then instruct the model to produce a JSON output with specific keys. Notice we explicitly mention “Respond in JSON format with keys: narrative, key_insights, recommendations, confidence_score.” – this is crucial to later parse the response automatically. In tests, they ensured the word “JSON” and all metrics appear in the prompt ￼.
 • Calling the LLM: We use self.model.predict(prompt, ...) with a low temperature (0.3 for more factual consistency) and a token limit (1024 tokens) ￼. The model’s raw text output is captured.
 • Parsing Output: We attempt to parse the output as JSON. If the model followed instructions, the output will be a JSON string like:

{
  "narrative": "Your heart rate was above average this week... (some summary)",
  "key_insights": ["Your resting heart rate of 60 bpm is good", "..."],
  "recommendations": ["Try to get 7-8 hours of sleep", "..."],
  "confidence_score": 0.85
}

We load this into a Python dict. If parsing fails (sometimes the model might return something not perfectly JSON), we fall back: we take the text as a narrative and at least put it under "narrative", perhaps also create a minimal "key_insights" list so that the downstream handling isn’t broken ￼. In the snippet above, our fallback just uses the raw text as the narrative and a truncated version as a “key insight” placeholder.

 • We add a "model": "Gemini2.5" tag for reference, and return the insight dict.

This GeminiClient.generate_insight would be invoked by the FastAPI route handling Pub/Sub messages for insights. For example, insight_generator/main.py might have:

app = FastAPI()
gemini_client = GeminiClient(project_id=os.getenv("GCP_PROJECT"))

@app.post("/generate-insight", status_code=200)
async def generate_insight(request: Request):
    body = await request.json()
    data = json.loads(base64.b64decode(body["message"]["data"]))  # parse Pub/Sub message
    user_id = data["user_id"]
    upload_id = data["upload_id"]
    analysis_results = data["analysis_results"]
    # Call the Gemini client to get insight
    insight = await gemini_client.generate_insight(analysis_results)
    # (Optional) Save insight to Firestore: e.g., store under insights/{user_id}/{upload_id}
    from google.cloud import firestore
    db = firestore.Client()
    doc_ref = db.collection("insights").document(user_id).collection("uploads").document(upload_id)
    doc_ref.set(insight)
    return {"status": "insight_generated", "user_id": user_id, "upload_id": upload_id}

We could also have this service directly return the insight in the HTTP response, but since it’s triggered via Pub/Sub push, usually we’d write to a database (Firestore) and just respond 200 OK. The mobile app would then get the insight from Firestore in realtime ￼. Storing in Firestore (insights/{uid}/{uploadId}) was part of the design ￼, enabling the client app to receive updates via Firestore listeners. We include it as a comment in code for completeness.

Summary: The insight service is relatively straightforward: it’s essentially a specialized client to Vertex AI. By structuring the prompt carefully and expecting JSON, we make the integration reliable – the model’s output can be parsed without error-prone string analysis. (The tests in the codebase confirm that the prompt contains all needed fields and that the JSON parsing fallback works ￼ ￼.) We also keep the heavy logic (like data processing) out of this service – it should not do any number crunching, just language tasks. This separation ensures we could scale or update the LLM part independently (e.g., swap in a different model or adjust prompts) without touching the ingestion or analysis pipeline.

2.7 PAT Model Mock – Stubbed Actigraphy Model for Local Development

The Pretrained Actigraphy Transformer (PAT) is an ML model that produces an embedding from a week’s worth of minute-level activity data (steps and sleep) ￼. In production, PAT might be a large model (perhaps 100+ million parameters) possibly served via TensorFlow or PyTorch. Using it directly in local development or tests could be slow or require heavy dependencies. Therefore, we implement a PAT model stub that mimics the interface and output shape of the real model, but with a trivial internal logic.

File: analysis_service/ml/pat_model_stub.py

import numpy as np

class PatModelStub:
    """Lightweight stub of the Pretrained Actigraphy Transformer (PAT) model."""
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
    def predict(self, actigraphy_vector: list[float]) -> list[float]:
        """
        Simulate generating an embedding from a 10080-length actigraphy vector.
        For consistency, outputs a deterministic pseudo-embedding of length embedding_dim.
        """
        arr = np.array(actigraphy_vector, dtype=float)
        if arr.size == 0:
            return [0.0] *self.embedding_dim
        # Simple summary-based fake embedding:
        avg = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        # Create an embedding filled with the avg and std repeated
        half = self.embedding_dim // 2
        embedding = [avg]* half + [std] * half
        if self.embedding_dim % 2 == 1:
            embedding.append(avg)  # if odd dimension, append one more value
        # (Alternative approach: return some fixed random vector seeded by user_id for consistency.)
        return embedding

This PatModelStub provides a .predict() method that takes an actigraphy input (typically a list of 10,080 minute-level movement values) and returns a list of length 128. Here we simply compute the average and standard deviation of the input and fill the output vector with those (half the vector as avg, half as std). This is obviously not capturing real patterns, but it gives a consistent output format. We could also seed a random generator for more variation, but using summary stats ensures the output changes if the input data changes (which could be useful in tests).

In the analysis pipeline, if we detect the environment is development or testing, we can use PatModelStub instead of the real model. For example, the ActigraphyProcessor might do:

if os.getenv("ENVIRONMENT") == "production":
    # load real PAT model (e.g., TensorFlow model from file)
    self.pat_model = load_pat_model(weights_path)
else:
    self.pat_model = PatModelStub()

Then, in ActigraphyProcessor.process(step_vector), it would call embedding = self.pat_model.predict(step_vector) to get the 128-dim representation. The rest of the pipeline treats it as if it were real. This way, local devs don’t need the PAT model weights or a GPU – the stub runs almost instantly.

The real PAT model (if integrated) would likely be loaded in memory or possibly called via a separate service. In either case, the stub’s interface should match the real one’s (e.g., method name .predict and input shape) so that swapping is easy.

Using such stubs and mocks is in line with making the development environment lightweight and fast. The unit tests in the repository hint at similar patterns (for instance, how the Gemini service is tested with patched responses rather than real API calls) ￼ ￼. We would similarly test our PAT integration by verifying that when run in dev mode, the stub is used and produces the correct shape output.

Note: If PAT’s output is critical for the LLM, one might even pre-compute or hardcode a sample embedding for known test inputs to ensure consistency. Our stub uses a simple deterministic calculation as a stand-in.

With all these pieces (ingestion API, Pub/Sub, preprocessing, processors, fusion, LLM, and stubs) in place, we have the blueprint of the system. The next step is to containerize and configure these services for different environments.

3. Docker & Deployment Setup

We package each component into Docker containers for both local development and production deployment. The monorepo contains separate Dockerfiles for the ingestion service, analysis service, and insight generator, since each will run as an independent service (process) in production.

3.1 Local Development Environment (Docker Compose + Emulators)

For local development and testing, we use Docker Compose to orchestrate our services along with Google Cloud service emulators (Pub/Sub, Firestore, etc.). This allows us to run the entire pipeline on a developer’s machine without needing actual cloud resources.

Key points for local setup:
 • Use Emulators/Mocks: Configure the Google Cloud SDK emulators for Pub/Sub, Firestore, and a fake GCS server. This is achieved by environment variables that the Google libraries recognize. For example:
 • PUBSUB_EMULATOR_HOST=localhost:8085 causes the Pub/Sub client to send all publishes to the local emulator instead of Google’s API ￼.
 • FIRESTORE_EMULATOR_HOST=localhost:8080 does similar for Firestore.
 • STORAGE_EMULATOR_HOST=<http://localhost:9090> (if using a GCS emulator such as fsouza/fake-gcs-server on port 9090).
 • Environment Config: We provide a .env file for development that sets ENVIRONMENT=development (and in production deployment we’ll set it to “production”). This flag is used in our code to toggle behaviors (like skipping Pub/Sub JWT verification, using stubbed models, etc.). The quickstart .env.example contains entries for these, e.g.:

ENVIRONMENT=development
GOOGLE_CLOUD_PROJECT=clarity-loop-development
FIRESTORE_EMULATOR_HOST=localhost:8080
PUBSUB_EMULATOR_HOST=localhost:8085
STORAGE_EMULATOR_HOST=<http://localhost:9090>

and other necessary config ￼. When running via Docker Compose, these env vars will be injected into the containers.

 • Docker Compose File: We define services for each component and for the emulators. For example:

version: "3.9"
services:
  ingestion:
    build:
      context: .
      dockerfile: Dockerfile.ingestion
    ports:
      - "8000:8000"  # expose API
    environment:
      - ENVIRONMENT=development
      - GOOGLE_CLOUD_PROJECT=clarity-loop-development
      - PUBSUB_EMULATOR_HOST=host.docker.internal:8085
      - FIRESTORE_EMULATOR_HOST=host.docker.internal:8080
      - STORAGE_EMULATOR_HOST=<http://host.docker.internal:9090>
  analysis:
    build:
      context: .
      dockerfile: Dockerfile.analysis
    environment:
      # similar env as above (no ports needed since it only receives Pub/Sub push)
      - ENVIRONMENT=development
      - GOOGLE_CLOUD_PROJECT=clarity-loop-development
      - PUBSUB_EMULATOR_HOST=host.docker.internal:8085
      - FIRESTORE_EMULATOR_HOST=host.docker.internal:8080
      - STORAGE_EMULATOR_HOST=<http://host.docker.internal:9090>
  insight:
    build:
      context: .
      dockerfile: Dockerfile.insight
    environment:
      - ENVIRONMENT=development
      - GOOGLE_CLOUD_PROJECT=clarity-loop-development
      - PUBSUB_EMULATOR_HOST=host.docker.internal:8085
      - FIRESTORE_EMULATOR_HOST=host.docker.internal:8080
      - STORAGE_EMULATOR_HOST=<http://host.docker.internal:9090>
  pubsub-emulator:
    image: google/cloud-sdk:latest
    command: gcloud beta emulators pubsub start --host-port=0.0.0.0:8085
    ports:
      - "8085:8085"
  firestore-emulator:
    image: google/cloud-sdk:latest
    command: gcloud beta emulators firestore start --host-port=0.0.0.0:8080
    ports:
      - "8080:8080"
  storage-emulator:
    image: fsouza/fake-gcs-server:latest
    command: -scheme http -port 80 -external-url <http://localhost:9090>
    ports:
      - "9090:80"

This setup will spin up the three Python services and the three emulators. We use host.docker.internal to let containers reach the emulator on the host network (Docker nuance). Now, the ingestion container’s Pub/Sub client will see PUBSUB_EMULATOR_HOST and publish to the local emulator; the analysis container’s Storage client will see STORAGE_EMULATOR_HOST and use the fake GCS, and so on.

 • Dockerfiles: Each Dockerfile would start from a base image (e.g. python:3.11-slim), copy the code, install requirements, and set the entrypoint to run the FastAPI app (usually via Uvicorn). For example, Dockerfile.ingestion might expose port 8000 and have CMD ["uvicorn", "ingestion_service.app:app", "--host", "0.0.0.0", "--port", "8000"]. The analysis and insight ones similarly but note: analysis and insight might not need to be externally accessible in dev (they only talk via Pub/Sub), but we can still run their FastAPI (which includes the Pub/Sub handler endpoints).
 • Running the stack: The developer can simply do:

docker-compose up --build

or use the provided Makefile target make dev-docker ￼. The quickstart guide demonstrates bringing up all services in one go ￼. Once running, a developer could hit the ingestion API at <http://localhost:8000/api/v1/healthkit/upload> with a test payload, and watch the logs as the message flows through to analysis and insight services.

Using this isolated local stack, we can iterate quickly: the PAT model stub and the emulator usage ensure we don’t require cloud access or real user data. It’s also suitable for integration tests – one could write tests that spin up these services (or even call the pipeline functions directly) to simulate an end-to-end run with sample data.

3.2 Production Deployment (GCP Cloud Run + Pub/Sub + GCS)

For production, each service will be built into a container and deployed to Cloud Run. We use Google Cloud services for storage and messaging:
 • Google Cloud Storage (GCS): The ingestion service writes raw payloads to a secure GCS bucket (as we coded). Ensure this bucket exists (e.g., healthkit-raw-data) and that the Cloud Run service account has permission to write to it. In production, one would also enable CMEK (managed encryption keys) and proper access controls given the sensitive nature of health data ￼.
 • Pub/Sub Topics: We need to create Pub/Sub topics and subscriptions:
 • Topic e.g. health-data-upload with a push subscription that targets the analysis service’s endpoint (Cloud Run URL for /process-task). This subscription should use OIDC authentication, configuring the service account and audience to match what our code expects (the PUBSUB_PUSH_AUDIENCE env, which could be the Cloud Run service URL or a custom audience string).
 • Topic e.g. insight-request with a push subscription to the insight generator service’s endpoint (e.g. /generate-insight).
 • The Cloud Run services (analysis and insight) should be deployed with the appropriate IAM allowing Pub/Sub to invoke them. Typically, we’d use the service’s identity and add a Pub/Sub push subscription with that identity’s token.
 • Environment Variables: In Cloud Run, we set ENVIRONMENT=production for each service (and any other needed variables like bucket names, project IDs, etc.). In fact, the Makefile snippet shows using --set-env-vars="ENVIRONMENT=production" when deploying ￼. In production mode, our code will perform the real JWT verification for Pub/Sub requests and use real models (if available) instead of stubs.
 • Docker Image Build: We can build images and push to Google Container Registry or Artifact Registry. For example, using the Makefile:

make docker-build  # builds the image (could be configured to build all or one)
make docker-push   # pushes to registry

We might adjust the Makefile to build/push separate images for each service, or tag them differently. Alternatively, use Google Cloud Build with separate Dockerfile targets.

 • Deploying Cloud Run: Deploy each service:

gcloud run deploy ingestion-service --image gcr.io/your-project/clarity-backend:ingestion \
    --platform managed --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars ENVIRONMENT=production,GOOGLE_CLOUD_PROJECT=your-project,...
gcloud run deploy analysis-service --image gcr.io/your-project/clarity-backend:analysis \
    --no-allow-unauthenticated \
    --set-env-vars ENVIRONMENT=production,PUBSUB_PUSH_AUDIENCE=<expected aud>,...
gcloud run deploy insight-service --image gcr.io/your-project/clarity-backend:insight \
    --no-allow-unauthenticated \
    --set-env-vars ENVIRONMENT=production,...

The ingestion service likely should allow public (authenticated via Firebase JWT in the app) access. The analysis and insight services need not be public; Pub/Sub will invoke them securely. We include relevant env vars (like project ID, region, etc.). For example, if using Vertex AI, the VERTEX_AI_LOCATION=us-central1 and any model identifiers (like GEMINI_MODEL=gemini-2.5-pro) could be set here as well.

 • Scaling and concurrency: We can configure concurrency and memory for each Cloud Run service. The analysis service might need more memory/CPU if it loads ML models (like PAT) – possibly use a larger instance with concurrency=1 (since it might be CPU intensive per request). The insight service can be smaller (just network calls to Vertex AI).
 • Permissions: Ensure the Cloud Run service accounts have the necessary permissions: e.g. analysis service needs permission to read from the GCS bucket and publish to the insight-request Pub/Sub topic; insight service needs permission to call Vertex AI (if using default credentials, assign the “Vertex AI User” role), and to write to Firestore for storing insights. These can be granted via IAM roles on the respective resources.

With these in place, the pipeline in production works as in dev, just with real services. A user uploads via the FastAPI endpoint, gets “queued”. The processing happens asynchronously and the insight ends up in Firestore (or could be fetched by an API call depending on design). The user’s app can listen on Firestore or call a GET endpoint to retrieve the insight when ready.

4. Configuration Toggling: Local vs. Cloud

We’ve touched on this, but to summarize: our code should detect the environment and adjust behaviors accordingly. This can be done via a simple environment variable or a config file. The project already uses patterns like config_provider.is_development() in places ￼. We will leverage an ENVIRONMENT env var and possibly others.

Configuration approach:
 • Create a shared.utils.config module or use Pydantic’s BaseSettings to define configuration values with env var overrides. For example:

from pydantic import BaseSettings
class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    GCP_PROJECT: str = "clarity-loop-dev"
    HEALTHKIT_RAW_BUCKET: str = "healthkit-raw-data"
    PUBSUB_HEALTH_DATA_TOPIC: str = "health-data-upload"
    PUBSUB_INSIGHTS_TOPIC: str = "insight-request"
    # ... other config like API keys, etc.
    class Config:
        env_file = ".env"
settings = Settings()

Then throughout the code we use settings.ENVIRONMENT or similar.

 • Dev vs Prod flags: In code, key switches include:
 • Pub/Sub JWT verification: only enforce in prod.
 • Using stub vs real PAT model: if ENVIRONMENT != "production", use PatModelStub.
 • Logging level or verbosity: maybe debug logs in dev, concise in prod.
 • Vertex AI usage: in dev, we might not have real Vertex credentials. The GeminiService in code checks if config_provider.is_development(): ... use dev-project ... else use prod project ￼. We can similarly direct the GeminiClient to either no-op (or use a smaller model, or a dummy response) if in dev. However, if we have internet, we could even call the real Vertex model in dev using a test project – that’s optional. At minimum, we ensure the dev environment doesn’t break if Vertex API is unavailable (e.g., catch errors or skip actual call).
 • External service endpoints: e.g., pointing to emulators vs. real endpoints is handled by environment variables (as shown with PUBSUB_EMULATOR_HOST etc.). The presence of those env vars is itself a toggle – the Google SDK auto-detects them.
 • Security differences: In local test, we often disable auth requirements (for instance, we might not require a valid Firebase token if running in a test context). Our FastAPI dependency verify_firebase_token could be stubbed to accept any token in dev mode. Alternatively, provide a dummy token via config for testing.

Given the .env example from quickstart, when a developer copies .env.example to .env, they get a pre-filled dev config that sets all the right toggles for using emulators ￼. For production, those emulator variables would be omitted and instead real service configs would be present (and secrets like service account JSON path or API keys might be provided via Secret Manager or Cloud Run env vars).

Finally, document these configurations in the README so developers know how to switch modes. For instance, “set ENVIRONMENT=development for local testing (with make dev-docker which uses the .env file), and ensure ENVIRONMENT=production in the cloud (the deployment scripts will set this)”.

5. Reproducibility Aids (Configs, Scripts, Documentation)

To make sure any developer can build and run this system from the blueprint, we provide supporting materials:
 • Example Config Files: We maintain an .env.example that lists all required env vars for both dev and prod (with dummy or default values). This includes Google Cloud project IDs, bucket names, API keys, etc. The developer can fill in or override as needed. For instance, in .env.example:

# Google Cloud settings

GOOGLE_CLOUD_PROJECT=your-gcp-project-id
HEALTHKIT_RAW_BUCKET=healthkit-raw-data

# Emulator toggles (use actual services if not set)

FIRESTORE_EMULATOR_HOST=localhost:8080
PUBSUB_EMULATOR_HOST=localhost:8085
STORAGE_EMULATOR_HOST=<http://localhost:9090>

# Environment flag

ENVIRONMENT=development

This makes it explicit what to configure.

 • Makefile Targets: The Makefile includes handy shortcuts:
 • make install – install dependencies.
 • make dev – run the FastAPI server locally (perhaps using Uvicorn directly, if one just wants ingestion API without Docker) ￼.
 • make dev-docker – as shown above, run Docker Compose for the full stack ￼.
 • make test – run all tests ￼.
 • make docker-build / make docker-run – build and run the Docker image(s) locally ￼.
 • make deploy – possibly a script to deploy to Cloud Run (using gcloud commands, as hints in Makefile around line 175+) ￼.
For example, after writing our code, we can do:

$ make docker-build    # Build the Docker image(s) for services
$ make docker-run      # Run one of the images locally for quick check, or use docker-compose for full
$ make test            # Run the test suite to verify everything passes

And for deployment:

$ make deploy-ingestion  # (if configured) or run gcloud manually as described above

 • README/Documentation: We update the repository README and docs with a quickstart for this feature. The quickstart might say:

 1. Setup: clone repo, create .env, run make dev-docker.
 2. Uploading Data: an example curl command or a reference to API docs (the docs could have an OpenAPI spec or an example JSON payload for HealthKit).
 3. Monitoring: how to see the output. E.g., “after upload, open the Firebase local UI or Firestore emulator console to see the new insight document” or “check the ingestion service logs in the terminal for status”.
 4. Running tests: how to run pytest, etc.
 • Sample Data & Tests: Include some sample HealthKit JSON data in docs/ or tests/. Perhaps a sample_healthkit_payload.json with a week’s worth of dummy data. Provide a snippet in docs on how that looks. This helps developers understand the expected format and quickly try the pipeline. We also write unit tests for each piece:
 • Test that preprocess_heart_rate correctly filters and normalizes data (e.g., if given a list with a 300 BPM outlier, test that it’s removed or capped) ￼.
 • Test that CardioProcessor.process returns expected feature shapes given synthetic data.
 • Test that FusionTransformer produces the correct output dimension and that increasing one modality’s values actually changes the CLS embedding (basic sanity).
 • Test that GeminiClient._create_prompt includes all metrics and the JSON instruction ￼.
 • Test that GeminiClient.generate_insight properly parses a known good JSON string and falls back on bad JSON ￼.
 • Possibly an end-to-end integration test that stubs the Gemini API call (so we’re not reliant on external service) to return a known JSON, then checks that the insight doc in Firestore (emulator) is as expected.

All these together ensure the system is implementation-ready. A new developer can follow the structured code, use the provided config and scripts, and get the entire pipeline running consistently.

⸻

Sources:
 • Clarity Loop Backend design docs on HealthKit integration and architecture ￼ ￼ ￼
 • Excerpts of the actual codebase (FastAPI endpoints, processing logic, etc.) for guidance ￼ ￼ ￼ ￼. These informed the examples above and ensure alignment with project conventions.
