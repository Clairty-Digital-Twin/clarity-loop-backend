# Chat with Your Apple HealthKit Data

## Executive Summary

We propose a Clean Architecture-based system that ingests Apple HealthKit time-series data (heart rate, HRV,
respiratory rate, sleep, actigraphy, etc.), processes it through specialized ML pipelines (including the
Pretrained Actigraphy Transformer (PAT) for movement data), and generates user-friendly insights via a Large
Language Model (LLM). The design emphasizes modularity, scalability, and future-proofing by separating concerns
into distinct layers and services. Key components include:

* Data Ingestion Service (FastAPI) to securely receive HealthKit data from iOS/watchOS, persist raw time-series
  data (e.g. to cloud storage), and enqueue processing tasks asynchronously.
* Analysis Microservices for each major data modality (e.g. an Actigraphy Service running PAT for activity/sleep,
  a Cardio Service for heart-related metrics). These services perform preprocessing, feature extraction, and ML
  inference on their respective signals.
* An Integration/Fusion Layer (which can be a dedicated service or part of the main analysis service) that combines
  multi-modal features (e.g. correlating heart rate trends with sleep patterns) into a unified health insight
  representation.
* Narrative Generation (LLM) Module using an advanced model (e.g. Google Vertex AI's Gemini) to convert analytical
  outputs into natural-language explanations. This can be invoked as an external service (per-model microservice)
  or within the analysis pipeline, with enterprise safety features applied.
* A Real-Time Results Store (e.g. Firestore DB) to deliver insights to client applications. The mobile app
  (chat interface) either receives pushed summaries or queries an endpoint that leverages LLMs in real-time to
  answer user questions using the stored data.

This architecture ensures separation of concerns (each service/model has a single responsibility), composability
(easy to add new data modalities or models), and scalability (each component can scale independently). It aligns with
Karpathy's ML system design principles by isolating each model's logic (PAT, other ML models, LLM) into independent
units and connecting them via a flexible pipeline (using events/queues). The result is a robust "digital twin" backend
that can support both batch analytics (e.g. daily summaries) and interactive Q&A with a user's health data in
natural language.

### Current Codebase Gaps and Observations

The clarity-loop-backend repository already implements a Clean Architecture foundation with domain models and a
health data API. Notable strengths include a strong layering of entities, services, and interface adapters and
rigorous validation rules for health metrics. For example, each HealthMetric (heart rate, activity, sleep, etc.)
is defined with business rules (heart rate ranges, non-negativity for steps, matching metric types to data)
enforced at the entity level. The code supports uploading a batch of metrics in a single payload and storing
them via an IHealthDataRepository abstraction.

However, there are several gaps to address for the full "Chat with your data" feature:

* Asynchronous Processing & ML Pipeline: The current implementation accepts data and returns a processing ID with
  status="PROCESSING", but the actual background processing (ML analysis) is not yet implemented. The design
  suggests using Google Cloud Pub/Sub and a separate analysis service, but in the code, save_health_data likely
  just stores data (e.g. in Firestore) without kicking off an ML job. We need to implement the Pub/Sub
  orchestration: after saving data, publish a message so an analysis worker can pick it up.
* Microservice Separation: Currently, all logic resides in one FastAPI app. For scalability and clean separation,
  we should introduce dedicated microservices for heavy ML tasks. The docs blueprint indeed envisions a "dedicated
  microservice running the PAT model for analytics" separate from the ingestion API. We need to extend this pattern
  for other metrics (HR, HRV, etc.), either as separate services per model or a single service that handles
  multi-modal fusion.
* Data Storage and Volume: The repo uses Firestore (and a MockHealthDataRepository for dev) as the data store.
  Storing large time-series (e.g. minute-level data for days) purely in Firestore may be inefficient. The blueprint
  recommends using Google Cloud Storage (GCS) for raw data files. We should confirm how much data is uploaded per
  request (for full HealthKit sync it could be substantial) and adjust storage strategy. A likely enhancement is to
  save raw JSON/CSV files to GCS (for durability and cost efficiency) and store only metadata or small summaries in
  Firestore (e.g. references to the file, status, results). This decoupling is not fully implemented yet.
* LLM Integration & Chat Interface: The current code does not yet integrate any LLM or natural language component.
  The design needs an LLM narrative generation step (using Vertex AI's GPT/Gemini or OpenAI) to provide the "chat"
  experience. This involves formulating prompts from the data and either pre-generating insights (which are stored
  for the app to display) or handling user queries on the fly. The architecture must accommodate an LLM either as a
  synchronous call during analysis or as a separate service that subscribes to analysis outputs. Additionally, the
  frontend chat UI needs an API to fetch these insights or ask questions; this is outside the current backend's
  implemented endpoints.
* Real-Time Updates: The blueprint suggests using Firestore's real-time capabilities (and optional FCM push
  notifications) to deliver results to the app. The backend code has endpoints to poll status (GET /processing/{id})
  and retrieve data (GET /health-data), but an event-driven push to the client is not yet in place. We will
  incorporate the Firestore update approach so that as soon as analysis/narrative is ready, the app can be notified
  or refresh its chat context.

In summary, the foundation is strong (Clean Architecture, validation, security scaffolding with Firebase Auth, etc.),
but the ML processing, multi-modal fusion, and LLM/chat integration are pending. We will address these in the
proposed architecture.

## Proposed System Architecture (Clean & Scalable)

Following Uncle Bob's Clean Architecture and SOLID principles, we design the system in distinct layers and services,
with all domain logic isolated from frameworks. The high-level architecture is composed of multiple loosely coupled
components communicating via defined interfaces or messaging. Below we describe each major component and how they
interact:

1. Health Data Ingestion Layer – FastAPI Service

Role: This service (the existing FastAPI app) handles incoming data from clients. It runs in Cloud Run (or similar) and exposes REST endpoints under /api/v1/health-data for uploading metrics. Key responsibilities and improvements:
 • Authentication & Security: Continue using Firebase Auth (Identity Platform) to verify JWTs on requests ￼ ￼. Each request is tied to a user UID for multi-tenant data isolation. Ensure HIPAA-level security (encryption, no PHI in logs, etc.). The existing @require_auth and get_current_user dependency already enforce that only a user can write/read their own data ￼ ￼ (the code checks health_data.user_id matches the token’s UID ￼). These checks remain vital.
 • Data Reception: The client (iOS app) will batch HealthKit records and send them (likely as JSON). For MVP, daily or on-demand batch uploads are fine. The payload structure can reuse the current HealthDataUpload model: it contains a user_id, a list of metrics, an upload_source (“apple_health”), and a timestamp ￼ ￼. Each metric in the list includes a metric_type and the corresponding data object (e.g. biometric_data for heart metrics, activity_data for steps, sleep_data for sleep) ￼ ￼. This design already supports multiple metric types in one upload, which is ideal for fusion (e.g. one upload can contain heart rate and sleep for the same period).
 • Validation & Acknowledgment: The ingestion service uses the domain HealthDataService to validate metrics (ensuring all required fields present and values in realistic ranges) ￼ ￼. If validation fails, a 400 error is raised with details. If successful, a unique processing_id (UUID) is generated ￼. The service then persists the data and returns an immediate response to the client with status “PROCESSING” and the processing_id ￼. This acknowledges receipt within seconds, letting the app know analysis is underway.
 • Data Persistence (Raw Data): To handle large time-series efficiently, we introduce storing raw payloads in Google Cloud Storage (GCS) as the blueprint prescribes. Upon receiving data, the FastAPI service will:
a. Serialize the health_data metrics list to JSON (or a compressed format) – possibly already given as JSON in request.
b. Write the file to a GCS bucket, e.g. at path gs://health-data/{user_id}/{processing_id}.json ￼. We use the Google Cloud Storage Python SDK (bucket.blob.upload_from_string) to upload ￼. Each file is namespaced by user and job, making it easy to retrieve later. This offloads storage of potentially large datasets from the database to a durable, scalable store (GCS is highly available and can handle large files) ￼.
Rationale: Storing raw HealthKit dumps in GCS improves performance and cost – Firestore is not ideal for large sequential data. GCS allows streaming reading by the analysis service and can store months of minute-level data cheaply. Files are encrypted at rest, meeting security needs ￼.
 • Task Queuing: After saving to GCS, the ingestion service will enqueue a message in Cloud Pub/Sub to trigger processing ￼. We’ll publish a JSON message to an “analysis-tasks” topic containing at least the user_id and the gs:// file path (and possibly metadata like upload timestamp or metric types included) ￼. This decouples the upload from processing – the API call returns immediately once the message is published (which is very fast) ￼. The client gets an HTTP 200 with the processing_id, and does not wait for analysis in this call. This achieves an “upload-and-forget” pattern ￼, enhancing UX since the user isn’t stuck waiting on a long request.
 • Immediate Response & Status Tracking: The returned HealthDataResponse (already defined in code) will indicate status: PROCESSING and an estimated processing time ￼. The backend should also record an initial status entry (e.g. in Firestore or an in-memory cache) so that GET /processing/{id} can report “processing” status. The current implementation likely uses the repository to save a status record along with the data. If using Firestore, we can create a document like users/{uid}/uploads/{processing_id} with a status field. This allows the status endpoint to fetch the status ￼ and also allows the analysis service to update the status (to “completed” or “failed”) later.

Outcome: The ingestion layer thus strictly handles receiving and queueing data. By following Clean Architecture, it delegates actual analysis to inner layers or other services, depends only on interfaces (e.g. repository), and remains framework-agnostic for core logic. It sets up the data for processing in a scalable way (file storage + message queue).

2. Analytics & ML Processing Layer – Microservices & Pipelines

This is the “brain” of the system: it takes the raw health data and produces meaningful features or predictions. We propose a modular pipeline with possibly multiple microservices or at least modular components for each type of analysis, aligned with Karpathy’s idea of “per-model services” for composability.

There are two architectural approaches to consider:
 • (A) Unified Analysis Service (Fusion Pipeline): One service (deployed on Cloud Run or Cloud Run Jobs) subscribes to the Pub/Sub topic and handles all data modalities in one pipeline. It will fetch the raw data file from GCS (since it has the path from the message), parse the JSON into metric objects, and then process each metric type accordingly. Within this service, we implement a strategy pattern for each metric: e.g., for activity/steps data, use PAT; for heart rate data, use a different model or algorithm; for sleep data, use appropriate logic. The service then fuses the results. This approach keeps all fusion logic in one place and avoids inter-service coordination. It is simpler for an MVP and ensures that combined insights (which may require comparing HR and sleep together) can be computed with full access to all data at once. We can still logically separate the code by model, but deploy it as one container.
 • (B) Per-Model Microservices: Multiple specialized services subscribe to the Pub/Sub topic (or to separate topics) and each handles one modality independently ￼. For instance, an Actigraphy Service processes movement and sleep data (these two are closely related since PAT can extract sleep patterns from actigraphy), a Cardio Service processes heart rate and HRV, and maybe an Others Service for respiratory or new metrics. All services would receive the same message (fan-out) but each would ignore or fetch only the data relevant to it. They would then need to coordinate to produce a unified insight – e.g. each could write partial results to a database or publish further messages. A final stage could either be an LLM service that pulls all partial results, or the client could query each. This approach is more complex to implement (needs aggregation of results), but it is highly scalable and extensible (you can add a new metric service without touching the others, and each can scale on its own). It aligns well with an event-driven microservice architecture where new subscribers can be added easily ￼.

Recommended: We suggest starting closer to approach (A) for the initial implementation, for simplicity, but structuring the code so that it could be split into separate services later. In practice, this means building the analysis as a pipeline with clear module boundaries for each metric type, using interfaces or handler classes for each. If a particular module becomes a bottleneck or is better isolated (for example, PAT might require a GPU and could be deployed separately), we can refactor that module into its own service without changing the overall workflow (thanks to using Pub/Sub or internal APIs for communication).

Now, detailing what happens in the analysis pipeline once a message is received:

2.1 Data Loading & Preprocessing: The analysis service receives a Pub/Sub push (or pull) with the processing_id, user_id, and file path. It verifies the authenticity of the Pub/Sub message (OIDC token, etc., as recommended for secure push subscriptions ￼). Then it downloads the raw data file from GCS. We then deserialize the JSON into our internal data structures (we can reuse the Pydantic models HealthDataUpload and HealthMetric for this). Each metric comes with timestamps; at this point we may sort or resample data if needed. Preprocessing steps include: filling missing data, filtering out outliers (e.g. spurious heart rate readings), and aligning time axes of different metrics (for multi-signal correlation). We might also convert units or compute additional fields (e.g. compute heart rate variability from raw RR-interval series if needed, etc.).

For sequential modeling, we likely need a fixed-length sequence as input. For example, PAT expects ~7 days of minute-level activity data (10080 points) ￼. If the data is shorter or longer, we handle that (pad, truncate, or use an appropriate context length). Similarly, for HR or HRV, we might consider daily time series or nightly summaries. The preprocessing ensures each model gets data in the format it was trained on (e.g. normalized, windowed, etc.).

2.2 Feature Extraction per Metric: Next, the service invokes the relevant model or algorithm for each metric category:
 • Actigraphy & Sleep (PAT Model): For movement (steps or accelerometer-based activity) and sleep analysis, we use the Pretrained Actigraphy Transformer. PAT is a foundation model trained on 29k participants’ wearable data ￼ ￼, capable of extracting high-level features like sleep efficiency, circadian rhythm strength, activity fragmentation, etc. We will load the appropriate pre-trained PAT weights (the repo provides pretrained encoders ￼). The input will be a sequence of activity counts (likely per minute or per 5-minute). PAT’s pipeline includes a patch embedding layer (e.g. grouping 18-minute patches) and a transformer encoder that outputs a representation of the time-series ￼ ￼. We will then apply whatever classification/regression heads or statistical analysis to derive meaningful metrics. According to the research and the API spec, PAT can output features like: sleep efficiency, rest-activity ratio, circadian stability, etc. ￼ ￼. We will extract these features and possibly intermediate representations for further use.
 • Heart Rate & HRV: We will analyze heart rate time-series (and related metrics like HRV, blood pressure if available) in parallel. If continuous heart rate data is available (e.g. periodic measurements or workout sessions), one approach is to compute features such as: resting heart rate, average and max heart rate over the period, detect any tachycardia/bradycardia events, and daily HRV trends (HRV could be a single daily value from HealthKit, often measured during sleep or morning). For HR specifically, we might not need a complex model initially; simple statistical or signal-processing techniques can yield useful insights (e.g. identifying days with unusual heart rate variability, or correlating high HR with low sleep). However, for future-proofing, we can incorporate ML here too – e.g. train a small transformer or LSTM to detect anomalies in the HR sequence. The literature shows deep learning can forecast or classify health states from HR sequences ￼, but for MVP, focusing on descriptive stats and thresholds might suffice. Key outputs might include: average resting HR, HRV baseline, notable deviations (like “Tuesday had an unusually high resting HR, possibly indicating stress or illness”), etc.
 • Respiratory Rate: If provided (often nightly average respiration from Apple Watch), this is typically one value per night. We can track trends (is it increasing over weeks? sudden jumps?), as changes in resting respiratory rate can indicate illness or recovery. We might simply compare the latest value to the user’s baseline. In a fusion context, respiratory rate can be mentioned alongside sleep (since it’s measured during sleep) – e.g. “Your breathing rate was normal last night at 15 breaths/min, indicating good recovery.” In future, if more granular respiratory data is available (like a time-series from a chest strap or O2 saturation variations), we could incorporate it similarly.
 • Other Metrics: The architecture is extensible. For example, skin temperature or blood oxygen could be added. Each new metric should have a dedicated processing module or service. Initially, though, we prioritize metrics that add clear value: Activity/Steps, Sleep, Heart Rate, HRV, Respiratory, and perhaps Blood Pressure (if users log it). The Mood or mental health indicators (mentioned in code as MoodScale, MentalHealthIndicator) could also be integrated later by correlating with physiological data, but MVP might exclude it.

Importantly, each of these processing steps should output a standardized representation of results – e.g., a JSON or Python dict with key results. For instance, after processing we might have:

{
  "user_id": "...",
  "processing_id": "...",
  "features": {
     "sleep_efficiency": 0.85,
     "avg_daily_steps": 8300,
     "resting_hr": 62,
     "hrv_balance": 0.60,
     "respiratory_rate_night": 15.0
  },
  "patterns": {
     "sleep": { "trend": "improving", "consistency_score": 0.78 },
     "activity": { "trend": "stable", "peak_times": ["07:00","18:00"] },
     "heart_rate": { "trend": "elevated on Tue", "variation": "high" },
     "...": "..."
  }
}

This structure is similar to the documented API response for /analyze/actigraphy which lists features (sleep_efficiency, etc.), patterns (daily patterns), and trends ￼ ￼. By using such a data contract, the various modules can contribute to one combined result. In Clean Architecture terms, these could be assembled into a domain entity (perhaps an InsightReport entity). We will define a Fusion logic that merges outputs: for example, it might take PAT’s circadian metrics and label them under “circadian_patterns”, take heart metrics under “cardio_metrics”, etc., producing one cohesive JSON.

This fusion step can also derive cross-modal insights – e.g. if high heart rate coincided with poor sleep, we can flag that. Such rules can be encoded in the fusion logic or eventually learned by a model. At minimum, the data contract should not lose any important info from each modality, so the LLM or front-end can draw connections if needed.

2.3 Asynchronous vs Synchronous Processing: All the above happens in the background service asynchronously relative to the initial request. We expect the analysis (especially PAT + LLM) to take on the order of a few seconds to maybe a minute for large datasets, which is why decoupling with Pub/Sub is crucial. Pub/Sub guarantees at-least-once delivery, so our service should be idempotent (processing the same message twice should yield the same result, or second time notice it’s already done). The service should acknowledge the Pub/Sub message only after successful processing so that failures cause a retry ￼ ￼.
 • Performance: Preliminary benchmarks from research suggest we can aim for sub-second inference for a single PAT model on a week of data (especially if using a small/medium model) ￼. Batch processing 100 sequences might take ~15s on a GPU ￼. We must ensure the service has enough memory/CPU (or a GPU) if needed. Cloud Run now offers limited GPU support or we could use Vertex AI custom endpoints for PAT if needed. We design so that heavy models (transformer inference) are executed efficiently – possibly using vectorized batch operations if multiple tasks queue up. We also monitor latency per request as a key metric (target < 2s for single user data analysis as per benchmarks ￼). If latency is higher, scaling out or using bigger instances would be considered.
 • Scaling: In MVP, one analysis service can handle tasks sequentially or with some concurrency (depending on Cloud Run instance concurrency settings). Because we expect bursts (e.g. many users upload around 10pm), using Pub/Sub smoothing and Cloud Run auto-scaling ensures we can spin up multiple instances to handle load. Each instance processes tasks independently. This design thus scales horizontally easily.
 • Updating Status: Once the analysis is done for a given processing_id, the service will update the status (in Firestore or wherever). For example, it can update the document uploads/{processing_id} with status “completed” and perhaps store intermediate results or a link to results. If using Firestore for result distribution, the service might write the final insights to a location like users/{uid}/insights/{processing_id} or simply users/{uid}/insights/latest. This Firestore write will be observed by the client app if it’s listening (enabling a real-time update in the UI) ￼. Additionally, the service can set a timestamp and possibly a TTL for how long raw data is kept (for data minimization).

3. Fusion and Interpretation Layer – LLM Narrative Generation

After numeric features and patterns are obtained, the next step is to turn these into an understandable narrative for the user – effectively translating data into “health insights” they can chat with. We incorporate an LLM (like Gemini 2.5 Pro on Vertex AI, or GPT-4) to generate this narrative. There are a few design choices for where this occurs:
 • Within the Analysis Service: As in the blueprint, we can have the analysis service itself call the Vertex AI ChatModel API once it has computed the features ￼. This means after getting the results JSON, the service formulates a prompt (system + user prompt) for the LLM. For example: “Summarize the following health data for the past week and give recommendations: [insert metrics and patterns].” We include the key stats: e.g. “User slept an average of 7.2 hours with 85% efficiency, had an average of 8542 steps/day, resting HR 62 bpm, HRV moderately low at 0.60, and so on.” The model’s response (text) is captured. We then post-process it (ensure it’s not making medical claims, etc., possibly using Vertex’s safety filters ￼). This approach keeps everything synchronous within one service, at the cost of that service needing access to the LLM API and potentially longer runtime per task ( a couple seconds for LLM response).
 • Separate LLM Service: Alternatively, we could treat narrative generation as a separate concern. The analysis service could publish another Pub/Sub event or call an internal endpoint, passing the computed features (not raw data, just the summary numbers) to an Insight Generation Service. This service would then do the LLM call and write the final text to Firestore. The blueprint notes this as an option (chaining Pub/Sub topics for multi-stage pipeline) ￼. The benefit is that the LLM service could be maintained or scaled independently (for instance, if we switch from Vertex to OpenAI, or if one day we use an on-prem LLM, it’s isolated). It also means the analysis service can finish quicker (it hands off to LLM and doesn’t wait). For MVP simplicity, we might stick to in-process calling, but we design the interface such that this can be pulled out easily (perhaps behind an interface IInsightGenerator that either calls Vertex API or sends a message).

Narrative Content: Regardless of where it runs, the LLM prompt and output need careful design:
 • We will provide context in the prompt such as the user’s goals or preferences if known (the /insights/generate spec allows a context with user goals, lifestyle, etc. ￼). For MVP, we might not personalize deeply, but we can include any known context (e.g. if user’s goal is “improve sleep”, emphasize sleep-related insights).
 • We specify the insight_type or focus (general vs specific) ￼. A general summary covers all metrics; we could also ask the LLM specialized questions (like “explain the user’s heart health over last week”).
 • The model should output a structured result. The example API response shows a summary plus detailed sub-sections for sleep, activity, circadian, each with key points and trend ￼ ￼, and even recommendations. We can instruct the LLM to follow such a format, or generate a few paragraphs of text depending on what UI we want. Storing a structured JSON of the narrative (with sections and maybe confidence scores per insight as in example ￼ ￼) is useful for flexible UI rendering. We might use few-shot prompting or a fixed template to ensure consistency (e.g. “Present the insights in JSON with fields [summary, sleep_insights, activity_insights, …]”).
 • Exposing to Chat: Now, to enable a chat experience, we have two pathways:
(a) Pre-generate insights – as above, produce a summary narrative after each upload and store it (the Firestore doc could contain the narrative text or sections). The mobile app’s chat UI can preload this as the assistant’s message: e.g. “Your weekly summary is ready: …”. The user can then ask follow-up questions. For follow-ups, the app would call a chat query endpoint on the backend. That endpoint would need to fetch relevant data (either the stored features or raw data or narrative) and feed it along with the user’s question into an LLM to get an answer. We can utilize LangChain (noting the repo has langchain dependency ￼) to orchestrate retrieval of data and query answering. For example, we might vectorize and store past insights in a vector DB (or even use the features JSON as knowledge) and use a QA chain. Or simpler, we can prompt the LLM with something like: “User asks: ‘How was my heart rate on Tuesday?’ Context: On Tuesday your average HR was X, you had Y high heart rate episodes. Answer in a friendly tone.” The LLM would generate an answer that the backend returns to the app. This requires an API endpoint like POST /chat/query with user question, and the backend having access to the user’s data/insights to insert into prompt.
(b) On-demand insight generation – an alternative is not to pre-generate anything, but whenever the user asks a question in the chat, the backend gathers the necessary raw data or computed stats and directly queries the LLM. This might be more flexible (the user can ask arbitrary range, or hypothetical questions). However, it can be slower if each query triggers fresh analysis or large context. For MVP, we have the periodic summary generation as a base, and allow Q&A on that summary (“Chat about your weekly report”). Over time, we can expand to allow queries like “compare this week to last week” by retrieving two summaries, etc.

Recommendation: Implement the Insight/Narrative generation as part of the pipeline for now, producing a summary after each batch upload. Additionally, provide an endpoint for ad-hoc questions that uses the stored features or narrative. This could be done by prefixing the stored summary to the LLM as context for any question (effectively, the summary plays the role of a knowledge base for that week). As we integrate LangChain or similar, we ensure no sensitive data leaves our environment unencrypted and use Vertex AI (which is covered by Google’s HIPAA compliance when used properly) ￼.

Finally, the LLM outputs (either the summary or answers) are delivered back to the user. If via Firestore: once the narrative text is stored in Firestore, the iOS app (which might be listening on users/{uid}/insights/latest) gets a real-time update and can display “Insight: …”. If via direct query: the HTTPS response from the chat endpoint contains the answer text.

Quality & Safety: Because health is sensitive, we incorporate some safeguards:
 • Prefer an LLM known for reliable outputs (Gemini in our design) and use any provided safety filters ￼ to avoid inappropriate advice. We instruct it to be factual and if unsure, to refer to data or say it doesn’t have that info.
 • We do not let the LLM see raw identifiable data; it only sees derived stats and trends, minimizing risk of PHI leakage. All processing happens in our secure cloud (no data sent to external third parties without BAA – Vertex AI stays in Google Cloud ￼).
 • We log LLM interactions for audit (without logging raw health data in plaintext, maybe storing the prompts/outputs in a secure way) – part of compliance and debugging.

4. Data Contracts Between Components

To keep the system robust and testable, we define clear data schemas (could be JSON or Protobuf) at each interface:
 • Ingestion API Payload: Already defined by HealthDataUpload Pydantic model. Example (JSON):

{
  "user_id": "<UUID>",
  "upload_source": "apple_health",
  "client_timestamp": "2025-06-02T09:00:00Z",
  "metrics": [
    { "metric_type": "heart_rate",
      "biometric_data": {
          "heart_rate": 72, "heart_rate_variability": 55,
          "systolic_bp": 120, "diastolic_bp": 80,
          "respiratory_rate": null, "skin_temperature": null
      }
    },
    { "metric_type": "activity_level",
      "activity_data": {
          "steps": 8500, "distance_meters": 6800.0,
          "calories_burned": 320.5, "active_minutes": 90,
          "exercise_type": "walking", "intensity_level": "moderate",
          "date": "2025-06-01"
      }
    },
    { "metric_type": "sleep_analysis",
      "sleep_data": {
          "total_sleep_minutes": 480, "sleep_efficiency": 0.85,
          "sleep_start": "2025-05-31T23:00:00Z", "sleep_end": "2025-06-01T07:00:00Z"
      }
    }
  ]
}

This example (based on our tests) shows multiple metric types together ￼ ￼. The backend should support all relevant HealthKit types similarly. (In future, we might accept file uploads for very large datasets, but JSON in request is fine up to moderate size.)

 • Pub/Sub Message: A lightweight JSON, e.g.

{ "user_id": "<uid>", "processing_id": "<id>", "file_uri": "gs://.../uid/xyz.json" }

Optionally include a list of metric_types included, for the consumers to know quickly what to parse (though they could inspect the file). Keep this minimal to avoid large messages.

 • Raw Data File Format: Likely the same JSON as the upload payload (we can store it directly). Alternatively, for continuous data like minute-by-minute steps, the app might pre-aggregate into an array (to avoid huge lists of individual metrics). In that case, the JSON might have a structure like:

{ "metric_type": "steps", "times": [...], "values": [...] }

However, since our ingestion API already handles arrays of HealthMetric objects, it may be acceptable to send a long list. We should be cautious with extremely large JSON (perhaps chunk by day). For now, storing the exact payload ensures fidelity.

 • Analysis Output (Features/Fusion): We propose a unified InsightResult schema, inspired by the docs ￼ ￼:

{
  "request_id": "<processing_id>",
  "user_id": "<uid>",
  "analysis_timestamp": "2025-06-02T09:05:00Z",
  "features": { ... },       // key metrics as flat numbers
  "patterns":  { ... },      // structured patterns/trends per category
  "anomalies": { ... },      // (if any anomaly detection results)
  "status": "completed"
}

This can be stored in Firestore or returned by an API. For example, features.sleep_efficiency = 0.85 as shown in the earlier snippet ￼, or patterns.circadian_trend = "declining" ￼. This data is meant for programmatic use (LLM input or advanced UI visualizations).

 • Narrative Output: If structured, e.g.:

{
  "insight_id": "<processing_id>",
  "summary": "Your sleep was efficient (85%) and you averaged ~8,500 steps/day. Keep up the good work!",
  "insights": {
    "sleep": {
       "description": "You slept an average of 8 hours with 85% efficiency, which is excellent. You maintained a regular schedule.",
       "trend": "improving"
    },
    "activity": {
       "description": "You averaged 8,542 steps per day, which meets your daily goal. Activity consistency is good with regular peaks in the morning and evening.",
       "trend": "stable"
    },
    "heart": {
       "description": "Your resting heart rate was slightly elevated on Tuesday (72 -> 80 bpm), possibly due to stress or poor sleep that night, but overall HR was in a healthy range.",
       "trend": "variable"
    }
  },
  "recommendations": [
      "Maintain your current sleep schedule to keep efficiency high.",
      "On days with higher heart rate, try short mindfulness exercises to manage stress."
  ]
}

This format (or something similar) was illustrated in docs ￼ ￼. It includes a high-level summary and detailed insights per category, plus recommendations. We would store this JSON in Firestore (users/{uid}/insights/{id}), or at least store the text fields.

 • Chat Query Interface: We will design an API such as POST /api/v1/chat with a request: { "user_id": "...", "question": "What was my average HR last week?" }. The backend will then retrieve the relevant data (we could fetch the latest InsightResult or narrative from Firestore) and call the LLM. The response might be {"answer": "Your average heart rate was about 60 bpm last week, which is within normal range for you."}. This keeps the chat interaction stateless on the server side (the state – user data – is in Firestore). We also consider using conversation context (history) if needed, but since this is “chat with your data”, each query can be relatively independent, focusing on the data.

By using these data contracts (with possible Protobuf definitions for internal services if performance demands), we ensure each component works against a clear interface. This makes it easier to test (e.g. we can unit test the PAT service by feeding it a fake JSON of steps and checking it returns expected features) and to replace implementations (e.g. swap out Vertex AI for another model by just changing the InsightGenerator component).

5. Exposing Insights to the LLM-Based Chat Interface

Finally, the architecture supports a chat interface where the user can query their data conversationally. The heavy lifting for this is actually in the LLM integration we discussed, but from a system perspective:
 • The mobile app (iOS) will have a chat UI. Initially, it might simply display the narrative summary as a chat message from an “AI Health Assistant”. The user can then type questions.
 • When the user asks a question, the app sends it to our backend (with the user’s auth token). We may create a new endpoint (e.g. GET /api/v1/insights/latest to retrieve the latest summary, or POST /api/v1/insights/query for questions).
 • The backend (which might reuse the same FastAPI app or a separate function) will retrieve the necessary data for answering. Likely, it will load the latest insights from Firestore (or if the question references a specific date range, it might load those specific records).
 • It will then compose a prompt for the LLM. We can use a system prompt like: “You are a health assistant with access to the user’s recent health metrics. Answer the user’s question using the data provided, and offer advice if appropriate.” Then provide the data context (either the summary or raw stats) and the user’s question. Using LangChain, we could treat the Firestore data as a knowledge base (we could even vectorize time-series or textual summaries and do similarity search, but given the small scope, probably not needed – we can inject directly relevant data).
 • The LLM returns an answer, which we send back to the app.

Because the user might ask something not covered by the latest summary, our design ensures flexible data access. For example, if someone asks “How did my metrics change compared to last month?”, our backend could retrieve the last 4 weeks of summary stats, compute the differences, and include those in the prompt or even let a code interpreter model calculate (that’s future enhancement). The key is the backend can orchestrate data retrieval and let the LLM focus on explanation.

Real-Time Consideration: If we implement streaming chat responses (token streaming), we might need a websocket or similar, but that’s optional. A simple request-response suffices for now given likely short answers.

Notably, Apple is reportedly working on a similar concept of an AI health coach that uses data from across devices to give advice ￼ ￼. Our design keeps us ahead by leveraging the user’s own data for personalized insights – something highly engaging and valuable. By exposing the insights through a conversational interface, we meet modern UX expectations (users can ask follow-ups naturally, rather than just reading a static report).

6. Batch vs Real-Time Ingestion Support

Our architecture supports both batch (periodic) and real-time streaming of data:
 • Batch Mode (MVP focus): Users (or the app automatically) upload chunks of data (e.g. daily at midnight or after a workout). The system processes each batch asynchronously as described. This covers the main use case of daily/weekly insights. It’s simpler and aligns with how HealthKit typically provides data (through queries for a period).
 • Near Real-Time Mode (future): For scenarios like continuously monitoring a workout or stress event, we might ingest data in a streaming fashion. HealthKit can deliver updates (e.g. new heart rate reading) in near real-time via background delivery. To handle this, we could either:
 • Enhance the ingestion API to accept smaller, frequent posts (the system can handle it, but too frequent posts might overwhelm if each triggers heavy analysis – so we’d likely only do lightweight processing in real-time, and full PAT analysis at intervals).
 • Or use a streaming pipeline (the blueprint suggests possibly using Apache Beam or Dataflow for streaming ML in future ￼). A streaming pipeline could continuously aggregate and feed data into models (perhaps using sliding windows).

For now, we prioritize batch: it’s simpler and most insights (sleep, daily activity) naturally fit daily cadence. However, the architecture is future-proof: the event-driven design with Pub/Sub can evolve into handling streams by treating each small update as a message. The PAT model can be retrained or adapted for streaming (maybe using stateful inference or shorter sequences). If needed, we might incorporate something like Kafka or Beam to better handle high-frequency data; e.g., a future enhancement might stream minute-level data through an online anomaly detector to immediately flag something (like a very high HR). We note this is a possible extension but beyond MVP scope.

To support both modes, our data contracts and pipeline should be flexible: e.g., the ingestion could label whether it’s a “full batch” vs “incremental update” in the message, and the analysis service could either accumulate data until a period is complete or do on-the-fly partial analysis. A pragmatic approach: stick to batch for now, and later add a real-time aggregator service that collects streaming data into short batches (like a window of 60 minutes) and reuses the same analysis components.

7. Performance, Scalability, and Limitations

Performance Bottlenecks & Mitigations:
 • Database Access: Firestore reads/writes are limited in throughput (and cost) especially if writing many data points individually. By writing bulk data to GCS and only key results to Firestore, we mitigate this. Firestore will mainly handle small documents (status, summary text) which it is good at, and GCS handles heavy payloads. We must be cautious with Firestore reads if the chat endpoint frequently queries historical data – perhaps caching recent results in memory or using Firestore indexes efficiently with queries by user and date. If needed for analytics (e.g. trends over months), consider moving aggregated metrics to BigQuery or Timeseries DB for advanced querying (not needed initially).
 • Parallelism: Python’s async (used in FastAPI) is leveraged for I/O. The analysis service can be an async FastAPI or a Cloud Run background worker. We should avoid Global Interpreter Lock issues by either multiprocessing or using libraries that release GIL (for heavy numeric ops, NumPy/PyTorch do so in C code). If one user’s job is extremely large, it could slow a worker – but Cloud Run can auto-scale horizontally if multiple jobs come in. We’ll also set reasonable time limits on processing (Cloud Run max timeout ~15 minutes; our tasks should finish well under that for one batch).
 • Model Efficiency: PAT is relatively lightweight for a transformer ￼, but still, using the “small” or “medium” configuration for inference may be wise for speed ￼ ￼. If more accuracy is needed, we can scale up to “large” but ensure the instance has the memory. Also consider using TorchScript or ONNX to optimize model inference if needed. The LLM call will likely dominate latency – using a streaming capable model might help user-perceived latency (show partial answer), but since we’re using Vertex AI which is optimized and presumably fairly fast (and can handle many concurrent requests) ￼, we rely on that. We will monitor how long narrative generation takes; if it becomes a bottleneck (say we allow many ad-hoc chat queries), we might have to introduce rate limiting or even fine-tune a smaller local model for quick QA (just a thought for future).
 • Scalability: The use of stateless services and Pub/Sub means each component can scale out. Pub/Sub will buffer bursts of messages and deliver as instances are available ￼. If our user base grows, we can increase Cloud Run concurrency or instance limits. Vertex AI LLM scales automatically under the hood, but we must consider cost – perhaps caching frequent questions’ answers or limiting queries per minute per user (the system could enforce some rate-limit on the chat API). Also, in anticipating more metrics (multi-modal expansion), the pipeline can handle them by just adding new processing modules or new microservices – thanks to the pub/sub decoupling, a new service can subscribe without affecting the existing flow ￼. For example, if we add a Nutrition Service for dietary data, it could listen to the same queue and produce insights that either are merged or answered separately.
 • Failure Handling: If any step fails (e.g. PAT model throws an exception on unexpected data format, or LLM API times out), we have to handle it gracefully. The analysis service should catch exceptions and mark the job as “failed” with an error message (and maybe retry a limited number of times if it’s a transient error). This ensures the user isn’t left hanging – the app could show “Analysis failed, please retry” or similar via status endpoint. Logging and monitoring (via Cloud Logging or custom logs) will be set up for all services to quickly debug issues. We could also implement alerting (e.g. if many jobs fail or latency spikes, Cloud Monitoring alerts us) ￼.

Existing Limitations and How Addressed:
 • The Clean Architecture overhead (so many layers) is generally worth it for maintainability. There is some complexity in wiring dependencies (the DI container), but it’s manageable. One thing to ensure is that our new modules (analysis services, etc.) also follow clean principles – e.g., PAT inference code should be in a use-case interactor or service class, not tangled in a controller. We might create a clarity.services.actigraphy_service or similar that the analysis service uses. This keeps things testable (we could inject a dummy model for tests).
 • Another limitation: Precision of ML insights. The system might find correlations that are not causal. We must be careful in the narrative not to overstate. This is more of a product consideration, but in architecture we can plan to allow manual rules or overrides. For instance, if HRV is very low, we might always add a recommendation “Consider relaxing activities” – perhaps defined by domain experts. Having a rule-based layer before final output could be a future addition for safety/quality (this could be implemented as a post-processing on LLM output or as part of the prompt).
 • HIPAA and PII: We treat all health data as sensitive. Data in transit is encrypted (HTTPS, and Pub/Sub with OIDC tokens to ensure only our service receives messages ￼). Data at rest: GCS and Firestore by default encrypt data. We restrict who can access these (service accounts with least privilege). We will also employ access auditing – each request is authenticated, and our services can log which user’s data was processed when (for audit trail). Since the architecture uses Google Cloud services covered by BAA (Firestore, Pub/Sub, Cloud Storage with proper config, Vertex AI with de-identified inputs) ￼, we can be compliant. The Clean Architecture approach actually aids compliance too – business rules and audit logic are centralized, not scattered in controllers, making it easier to prove and test that, say, no user can access another’s data (we have that check in one place).

8. Architectural Diagram (Logical Flow)

(Diagram omitted in text-only format.) In summary, the end-to-end flow is:

 1. iOS/Watch App – collects HealthKit metrics and sends via REST to backend (with Firebase auth token).
 2. FastAPI Ingestion Service – validates and stores raw data (GCS) and enqueues a Pub/Sub message for processing ￼. Returns processing ID immediately.
 3. Pub/Sub – delivers the message to subscribed Analysis Service(s) asynchronously.
 4. Analysis Service – retrieves data from GCS, runs preprocessing and ML models (PAT for actigraphy, etc.) to extract features ￼. If multi-service, each handles its domain and possibly writes partial results.
 5. Fusion/Narrative (LLM) Generation – the analysis service either directly calls the LLM API with the aggregated data or publishes to an LLM service. The LLM produces a narrative insight (text) ￼.
 6. Storage & Notification – The final results (numbers + narrative) are written to Firestore in the user’s document ￼. This triggers a real-time update to the app. Optionally, an FCM push notification can also be sent (“Your health report is ready”) ￼.
 7. Client App – receives the insight (via Firestore listener or by polling an endpoint). It displays the summary in the chat UI. The user can now ask questions.
 8. Chat Query – The app sends user’s question to a chat endpoint. Backend fetches relevant data (from Firestore or caches) and calls the LLM to get an answer, which is sent back. The answer can also be stored if we want a record. The chat UI shows the answer, and the conversation can continue.

This pipeline is fully asynchronous and scalable, as each stage is decoupled by clear interfaces (HTTP or Pub/Sub). It also isolates heavy computations (ML, LLM) in backend services where they can be optimized, rather than burdening the client device.

Relevant Research and Systems References

Our architecture stands on shoulders of prior work and best practices:
 • Clean Architecture & ML Systems: The repository’s adherence to Clean Architecture ensures a robust foundation ￼. We extend that to ML by treating each ML model as a plugin to the architecture, not breaking the layering. Andrej Karpathy has advocated for modularity in AI systems, often separating model training/serving components and ensuring data pipelines are flexible to accommodate new data or models. Our use of distinct services for PAT, etc., and an event-driven pipeline for composability reflects these principles (e.g. enabling fan-out and chaining of processing steps) ￼ ￼. This approach is similar to modern ML platforms (like Uber’s Michelangelo or Google’s TFX) which pipeline data through specialized components.
 • Pretrained Actigraphy Transformer (PAT): As introduced in the paper “AI Foundation Models for Wearable Movement Data in Mental Health Research”, PAT is the first foundation model for wearable time-series, leveraging transformer architectures and patch embeddings to achieve state-of-the-art results ￼. By integrating PAT, we align with state-of-the-art research in digital health. The model’s ability to interpret long sequences of actigraphy data provides a rich feature set that we use for sleep and activity insights. The open-source nature of PAT and its lightweight design make it feasible to include in a cloud service (with optimizations) – a considerable innovation over traditional simplistic activity summaries.
 • Multi-Modal Health Data Fusion: Combining signals like HR, movement, and sleep is supported by research indicating improved accuracy in health monitoring when using multiple data streams ￼ ￼. A transformer-based method effectively leveraging heart rate, accelerometer, and other physiological signals together achieved significantly better recognition of states than single-signal models ￼. This justifies our fusion approach – by correlating metrics (e.g. seeing both the physiological and behavioral data), we deliver more nuanced insights (for instance, distinguishing poor sleep due to high HR/stress vs. due to irregular schedule). Our architecture enables such fusion now and paves the way for adding more modalities (e.g. skin conductance or mood scores) following the same pattern.
 • LLM for Personal Health Narratives: We leverage advances in generative AI (like Google’s Gemini, presumably an evolution of GPT-style models) to translate data into text. This concept is at the cutting edge – notably, Apple’s upcoming AI health coach aims to “advise users based on data from across their devices” ￼, confirming that personalized health insights via AI is an important emerging trend. Startups like Humanity and others are also using AI on health data to increase user healthspan ￼. Our system is designed to be on par with these developments, using an enterprise-grade LLM (Vertex AI) which keeps data private and complies with healthcare standards ￼. We expect this will deliver high-quality, context-aware explanations to users, improving engagement and understanding of their health metrics.
 • Prioritized Metrics for MVP Value: We prioritize metrics that research and user feedback indicate are most actionable:
 • Sleep data (duration, efficiency, consistency) – crucial for overall wellness, and PAT excels here ￼.
 • Activity (steps, active minutes) – easy to track and improve, and correlates with mental and physical health.
 • Heart Rate & HRV – key for cardio fitness and stress; changes can indicate fatigue or illness.
 • Respiratory Rate (during sleep) – a subtle metric that can flag issues like respiratory illness or overtraining when elevated.
These provide a strong MVP set. Additional metrics like Blood Pressure (if users log it) and Blood Oxygen (for those with Apple Watch SpO2) could be secondary priorities. We also note Mood/Stress entries if the user logs them (or if deduced via wearables) could be integrated – e.g. the PAT research is in context of mental health, and combining subjective mood with objective data can yield powerful insights (“On days you reported high stress, your HRV was low and sleep was shorter”). While mood tracking might be phase 2, our flexible design allows adding it (likely as another input to the LLM prompt or even a model to predict mood from signals).
 • Clean Architecture Rationale: By following Clean Architecture, our system remains testable and maintainable ￼ ￼. For instance, we can unit test the HealthDataService without any cloud services (using the Mock repository) to verify that invalid data is rejected and valid data leads to a processing event. We can integration-test the whole flow in a staging environment by simulating an upload and verifying an insight appears (the repo already has an end-to-end test scaffolding ￼). This discipline reduces regression issues as we evolve the platform. It also allows multiple interface adapters – today a REST API for iOS, tomorrow maybe a gRPC service for third-party integrations – without changing core logic.

In conclusion, the proposed architecture builds upon the current codebase’s clean design and extends it with a powerful ML pipeline and LLM capability. It ensures that as new health signals emerge or AI models improve, we can incorporate them with minimal friction – the system is modular by design. By separating data ingestion, analysis, and presentation (LLM/chat), we achieve a scalable solution that can grow from an MVP serving basic insights to a comprehensive “digital health twin” platform. This platform will enable users not just to see their data, but to converse with it, gaining understanding and guidance, which is a compelling and novel user experience in personal health management.

References:
 • Clarity Clean Architecture documentation ￼ ￼
 • Clarity Async pipeline blueprint (HealthKit, Pub/Sub, PAT microservice, Gemini LLM) ￼ ￼
 • AI Foundation Models for Wearable Movement Data… (PAT paper & README) ￼ ￼
 • Example ML API specs for actigraphy analysis and insight generation ￼ ￼
 • Apple’s planned AI health coach uses multi-modal data for personalized advice ￼ ￼
 • Research on multi-sensor fusion (heart rate + movement) improving health monitoring accuracy ￼ ￼
 • Google Vertex AI (Gemini) chosen for compliant, high-quality LLM text generation ￼ (ensuring data stays in our GCP project).

Apple HealthKit “Stub-First” Plan

(Turning today’s architecture into code you can paste & run)

⸻

0 . Why stub?

A stub adapter lets the rest of the pipeline behave as if real HealthKit data already exists. You:

 1. Unblock feature work (PAT, Gemini, chat UI) without waiting for Apple-side auth/export plumbing.
 2. Lock interfaces early so nothing breaks later when you swap in the real adapter.
 3. Automate CI—every push runs an end-to-end test that uploads “fake” HealthKit, triggers Pub/Sub, runs PAT, calls Gemini, and verifies Firestore gets an insight document.

⸻

1 . Folder & file layout (add to clarity-loop-backend)

clarity_loop_backend/
├─ clarity/                    # domain & use-cases (already exists)
│  ├─ adapters/
│  │   ├─ __init__.py
│  │   ├─ base.py              # IMetricIngestAdapter  (⇩ §2)
│  │   └─ apple_health_stub.py # AppleHealthStubAdapter (⇩ §3)
│  └─ services/
│      └─ ingestion_service.py # orchestrates adapters
├─ tests/
│  └─ e2e_healthkit_stub_test.py
└─ samples/
    └─ healthkit_stub_batch.json

⸻

2 . Define the adapter interface (clarity/adapters/base.py)

from typing import List, Protocol
from clarity.domain.models import MetricPayload, HealthDataUpload

class IMetricIngestAdapter(Protocol):
    """Converts raw source files/objects into standardized MetricPayloads."""
    async def parse(self, raw_source: dict | str | bytes) -> HealthDataUpload: ...

Why: any future adapter (real HealthKit SDK, Garmin, Oura, CSV, etc.) just implements parse() and the ingestion service doesn’t care where data came from.

⸻

3 . Implement the stub adapter (clarity/adapters/apple_health_stub.py)

import json, uuid, random, datetime as dt
from typing import List
from clarity.domain.models import MetricPayload, HealthDataUpload
from clarity.adapters.base import IMetricIngestAdapter

METRIC_TYPES = ["heart_rate", "activity_level", "sleep_analysis"]

class AppleHealthStubAdapter(IMetricIngestAdapter):
    """Fake adapter that returns deterministic-but-realistic HealthKit data."""
    def __init__(self, seed:int=42):
        random.seed(seed)

    async def parse(self, raw_source:dict|str|bytes) -> HealthDataUpload:
        # 1. Ignore raw_source; generate synthetic data instead
        user_id = f"stub-{uuid.uuid4()}"
        timestamp = dt.datetime.utcnow().isoformat()

        metrics: List[MetricPayload] = []
        # ---- Heart Rate / HRV ----
        metrics.append(MetricPayload(
            metric_type="heart_rate",
            biometric_data={
                "heart_rate": random.randint(55, 75),
                "heart_rate_variability": random.randint(40, 80)
            }
        ))
        # ---- Activity ----
        steps = random.randint(4000, 12000)
        metrics.append(MetricPayload(
            metric_type="activity_level",
            activity_data={
                "steps": steps,
                "distance_meters": round(steps * 0.8, 1),
                "calories_burned": round(steps * 0.04, 1),
                "active_minutes": random.randint(30, 120),
                "exercise_type": "walking",
                "intensity_level": "moderate",
                "date": dt.date.today().isoformat()
            }
        ))
        # ---- Sleep ----
        metrics.append(MetricPayload(
            metric_type="sleep_analysis",
            sleep_data={
                "total_sleep_minutes": random.randint(360, 510),
                "sleep_efficiency": round(random.uniform(0.75, 0.9), 2),
                "sleep_start": (dt.datetime.utcnow()-dt.timedelta(hours=8)).isoformat(),
                "sleep_end": dt.datetime.utcnow().isoformat()
            }
        ))

        return HealthDataUpload(
            user_id = user_id,
            upload_source = "apple_health_stub",
            client_timestamp = timestamp,
            metrics = metrics
        )

### Notes

* Deterministic with a fixed seed for repeatable CI.
* Uses domain Pydantic models so validation still runs.
* Returns one HealthDataUpload; real adapter may split large exports into multiple.

⸻

### 4. Wire the adapter into the ingestion service

# clarity/services/ingestion_service.py  (simplified)

class IngestionService:
    def __init__(self, repo:IHealthDataRepository, publisher:PubSubPublisher):
        self.repo, self.publisher = repo, publisher
        # register adapters
        self.adapters = {
            "apple_health_stub": AppleHealthStubAdapter(),
            # future: "apple_health": RealHealthKitAdapter(),
        }

    async def ingest(self, source:str="apple_health_stub", raw:dict|str|bytes=None):
        adapter = self.adapters[source]
        health_upload = await adapter.parse(raw)
        processing_id = await self.repo.save_health_upload(health_upload)
        await self.publisher.enqueue(processing_id, health_upload.user_id)
        return processing_id

FastAPI endpoint can simply forward raw=None for the stub route:

@router.post("/health-data/stub")
async def ingest_stub(service: IngestionService = Depends(get_service)):
    pid = await service.ingest(source="apple_health_stub")
    return {"processing_id": pid, "status": "PROCESSING"}

⸻

## 5. End-to-End CI test (tests/e2e_healthkit_stub_test.py)

    import httpx, asyncio, pytest
    
    pytestmark = pytest.mark.asyncio
    BASE = "http://localhost:8000"   # or live staging URL
    
    async def test_stub_flow():
        async with httpx.AsyncClient(base_url=BASE) as client:
            r = await client.post("/health-data/stub")        # new route
            assert r.status_code == 200
            pid = r.json()["processing_id"]
    
            # poll status up to 60 s
            for _ in range(30):
                s = await client.get(f"/processing/{pid}")
                if s.json()["status"] == "COMPLETED":
                    break
                await asyncio.sleep(2)
            else:
                raise AssertionError("Processing never completed")
    
            # fetch final insight
            insight = await client.get(f"/insights/{pid}")
            assert insight.status_code == 200
            assert "summary" in insight.json()

* Add this test to your GitHub Actions workflow so every push validates the pipeline.

## 6. How to swap in real HealthKit adapter later

| Step | Action | Surface touched? |
|------|--------|----------------|
| 1 | Build RealHealthKitAdapter that parses Apple's export.xml or watchOS background deliveries and returns HealthDataUpload | Only new file in clarity/adapters/ |
| 2 | Register adapter in IngestionService.adapters as "apple_health" | 1-line change |
| 3 | Update mobile app to POST /health-data with source=apple_health and raw payload | Front-end only |
| 4 | Update CI to use real sample export (keep stub test as a smoke test) | tests/ folder |

Because every downstream stage—Pub/Sub, analysis, PAT, Gemini—already consumes the canonical HealthDataUpload, nothing else changes.

⸻

## 7. Generating richer stub data (optional)

If you need day-level sequences for PAT:

* Add a helper generate_activity_sequence(days:int=7) -> list[int] that returns 10 080 step counts (minute-resolution).
* Store it in payload["activity_sequence"] so the PAT service can test patch embeddings, etc.
* Keep sequence lengths realistic (0 – 300 steps/min).

⸻

## 8. Documentation snippet (paste into /docs/apple_health_stub.md)

### Apple HealthKit Stub Adapter (v1)

#### Purpose

Enable end-to-end pipeline validation without real HealthKit exports.

#### How it works

* FastAPI route __POST /health-data/stub__ triggers `AppleHealthStubAdapter`.
* Adapter synthesizes realistic metrics (HR, steps, sleep) and returns a `HealthDataUpload`.
* Upload is persisted (GCS) and `analysis-tasks` Pub/Sub message is published.
* Analysis service runs PAT + statistic rules → Fusion JSON → Gemini narrative.
* Narrative stored in Firestore; mobile app listener displays “stub” insight.

#### Key files

* `clarity/adapters/apple_health_stub.py` – data generator
* `samples/healthkit_stub_batch.json`     – example output
* `tests/e2e_healthkit_stub_test.py`      – CI smoke test

#### Replacing with real adapter

Swap `"apple_health_stub"` with `"apple_health"` once `RealHealthKitAdapter` implements parsing of:

* `export.xml` (manual user export) __or__
* HK background deliveries via App Extension (future).

__No other backend service requires modification.__

⸻

## 9. Run it locally (developer cheat-sheet)

```bash
# Boot local Firestore emulator, Pub/Sub emulator (optional)
gcloud emulators pubsub start --project test-proj &
gcloud emulators firestore start --host-port=localhost:8080 &
export PUBSUB_EMULATOR_HOST=localhost:8085
export FIRESTORE_EMULATOR_HOST=localhost:8080

```

```bash
# Start FastAPI server
uvicorn clarity.web.main:app --reload
```

```bash
# Trigger stub upload
curl -X POST http://localhost:8000/health-data/stub
```

```bash
# Watch logs of analysis worker (if running)
docker compose logs -f analysis
```

```bash
# View Firestore document
python scripts/print_firestore_insights.py | jq
```

### Deliverable checklist

* Copy folder layout & code snippets into repo.
* Commit apple_health_stub.md in /docs/.
* Add the new FastAPI route and register the adapter.
* Ensure Cloud Run analysis worker detects stub uploads (same Pub/Sub).
* Push; verify GitHub Action passes (e2e_healthkit_stub_test.py).

Once those green-checks light up, you can develop PAT analytics and Gemini prompts with confidence—knowing real HealthKit data can drop in later by swapping adapters, no pipeline surgery required.
