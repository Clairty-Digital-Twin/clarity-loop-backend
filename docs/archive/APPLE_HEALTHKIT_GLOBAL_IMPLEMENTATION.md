# Apple HealthKit Integration Blueprint for Clarity Loop Backend

## System Architecture Overview

The proposed solution is an **event-driven, asynchronous pipeline** that cleanly separates data ingestion, analysis, and insight generation. The architecture follows Clean Architecture principles, ensuring each concern (ingestion, processing, ML inference, fusion, LLM insight generation) is isolated and scalable. The flow is as follows: a user's iOS/watchOS device collects HealthKit metrics and uploads them via a FastAPI endpoint; the backend immediately stores the raw data and enqueues a background task, allowing the mobile app to get an instant ACK without waiting for analysis. A **Pub/Sub message** triggers an analysis service on Cloud Run, which processes the data and produces a fused health state vector. This result is then published to another Pub/Sub topic, consumed by an insight-generation service that invokes the Gemini 2.5 LLM for narrative output. Finally, the **insight** (text and/or structured data) is stored in Firestore for real-time delivery to the client app. All components are fully decoupled and asynchronous, enabling independent scaling and robust failure isolation (e.g. if the ML service is slow or down, ingestion is unaffected).

```mermaid
flowchart LR
    subgraph Device (iOS/Watch)
        A[HealthKit data<br/>(HR, HRV, RR, SpO₂, etc.)] -->|HTTPS POST /v1/health-data| B[(FastAPI Ingestion API)]
    end
    subgraph Backend (GCP)
        B -->|Validate & store raw JSON| GCS[GCS Bucket (PHI-encrypted)]
        B -->|Publish message| T[(Pub/Sub Topic: health-data-upload)]
        T -->|Push w/ OIDC JWT| E[Cloud Run: Analysis Service]
        subgraph Analysis Service (Cloud Run)
            E1[Preprocess each modality] --> E2[CardioProcessor<br/>(HR, HRV)]
            E1 --> E3[RespirationProcessor<br/>(RR, SpO₂)]
            E1 --> E4[OtherProcessors<br/>(BP, Temp, VO₂, ECG...)]
            E2 --> E5[[Fusion Transformer Head]]
            E3 --> E5
            E4 --> E5
            E5 --> E6[Unified Health State Vector (embedding)]
        end
        E --> E1
        E6 -->|Publish results| T2[(Pub/Sub Topic: insight-request)]
        T2 -->|Push w/ OIDC JWT| G[Cloud Run: Insight Generator]
        G -->|Call Vertex AI (Gemini 2.5)| LLM[Gemini 2.5 LLM API]
        LLM -->|Insight JSON/text| G
        G -->|Write| FS[(Firestore: insights)]
    end
    FS -->|Realtime update| UI[Mobile App (Gemini chat interface)]
```

**Explanation:** The **Ingestion API** (FastAPI on Cloud Run) authenticates the user (Firebase ID token) and accepts a batch of HealthKit samples (multiple metrics) in a JSON payload. It performs initial validation (e.g. data types, ranges) and writes the raw payload to a **Google Cloud Storage (GCS)** bucket for durable, cold storage (in a **PHI-protected, CMEK-enabled** bucket). Next, the ingestion service publishes a message to a **Pub/Sub topic** (e.g. `health-data-upload`), containing metadata like `user_id`, a pointer to the GCS file, and an `uploadId` (a unique batch ID). This design uses **at-least-once delivery** and idempotent upload IDs to avoid duplicate processing if a message is re-delivered. A Pub/Sub **push subscription** is configured to deliver these messages to the **Analysis Service's** Cloud Run endpoint (e.g. `/process-task`), with an **OIDC JWT** attached for authentication. The Analysis Service acknowledges the message and processes the data (detailed below). After computing results, it publishes an `insight-request` message with the derived features (not raw data) to trigger the **Insight Generator** service. The Insight Generator calls the Vertex AI **Gemini 2.5** LLM with a structured prompt (containing only non-PHI summary stats) and obtains a textual and/or JSON insight. This insight is then stored in **Firestore** (e.g. under `insights/{userId}/{uploadId}`), which the client app observes via realtime listeners. Optionally, a push notification (FCM) can inform the user that “Your health report is ready.” The end result is a **fully asynchronous pipeline**: *Upload → GCS → Pub/Sub → Cloud Run (analysis) → Pub/Sub → Cloud Run (LLM insight) → Firestore → App*, with no blocking calls across stages.

## File Organization and Layering

Implement the above within the existing Clean Architecture of `clarity-loop-backend` by adding new modules in logical layers:

* **Domain Models & Enums:** Define new metric types and data models in the **domain layer** (or shared `models` module). For example, introduce an `Enum` `HealthMetricType` with entries for all supported modalities: `HEART_RATE`, `HRV_SDNN`, `RESPIRATION_RATE`, `SPO2_PERCENT`, `BLOOD_PRESSURE_SYSTOLIC`, `BLOOD_PRESSURE_DIASTOLIC`, `BODY_TEMPERATURE`, `VO2_MAX`, `ECG_CLASSIFICATION`, `WORKOUT`, etc., alongside existing types like `STEP_COUNT`. Create Pydantic models for structured data, e.g.:

  ```python
  class HeartRateSample(BaseModel):
      timestamp: datetime
      bpm: float = Field(..., gt=0, lt=250)
      source: Optional[str]
  class BloodPressureSample(BaseModel):
      timestamp: datetime
      systolic: int = Field(..., gt=50, lt=250)
      diastolic: int = Field(..., gt=30, lt=150)
      position: Optional[str]  # e.g. seated/standing
  ```

  and similarly for HRV (ms), SpO₂ (%), Temperature (°C), etc., each with appropriate validation (physiological ranges). Also define composite models like `WorkoutSession` (with fields for type, duration, calories, avg HR, etc.). These models enforce **business rules** (e.g. non-negative steps, plausible heart rate bounds) at the entity level.

* **Ingestion API (Interface Adapter):** Extend or create FastAPI endpoints in `src/clarity/api/health_data.py` (for example) to handle new uploads. For instance, a `POST /v1/health-data` endpoint that expects the payload format outlined above (with a list of samples). This layer should parse the JSON into Pydantic models (e.g. a `HealthDataUpload` model containing `List[HealthSample]`), perform immediate validation, and call into an application service to handle storage and publishing. Ensure that the **authentication dependency** (`@require_auth`) is applied, so only authorized users can hit this endpoint. Also verify the `user_id` from the token matches the data's user field to enforce multi-tenant isolation.

* **Application/Service Layer:** Implement a use-case interactor (e.g. `SaveHealthDataAndEnqueueAnalysis`) in `src/clarity/services/health_data_service.py`. This service function takes the parsed data and current user, then:

  1. Stores the raw data file to GCS (e.g. `gs://health-raw/{uid}/{uploadId}.json`). Use Firestore or an **IHealthDataRepository** only for light metadata if needed (large time-series go to GCS to avoid Firestore size limits).
  2. Publishes a Pub/Sub message to the `health-data-upload` topic. Use the Google Cloud PubSub Python client in async mode. The message body can include `uid`, `uploadId`, the GCS file path, and perhaps a summary of types included. Tag the message with attributes for filtering if needed (e.g. a `data_type: "healthkit_batch"` attribute).
  3. Returns a response to the client with a tracking ID (maybe the `uploadId`) and status "PROCESSING". The client can use this ID to query status or just wait for the Firestore update.

* **Analysis Service:** Create a new Cloud Run service (could be a separate FastAPI app, e.g. `analysis_app.py`, deployed as `analysis-service`). This will subscribe to Pub/Sub messages (via push). In code, implement a **Pub/Sub push handler** endpoint (e.g. `/process-task`) that verifies the JWT from Pub/Sub (Google-signed OIDC token) and acknowledges the message. Inside, it will parse the message JSON (to get `uid`, file path, etc.), then coordinate the analysis:

  * Download the raw data JSON from GCS (using the GCS client). Deserialize into the Pydantic models (e.g. list of samples).
  * Call the **preprocessing and signal processing modules** (detailed in the next section) to transform each metric's raw series into cleaned, aligned form and then into feature vectors or embeddings.
  * Perform multi-modal **fusion** to obtain a single unified health state vector for this batch.
  * Publish a new Pub/Sub message to the `insight-request` topic with the user ID and the computed features/embedding (or store the fusion result in a temporary Firestore/Storage location and send a reference).
  * (Optionally, if we choose synchronous insight generation instead of a second service, the analysis service could directly call the LLM at this point. However, the **authoritative design** uses a separate stage for LLM to keep the pipeline fully async and scalable.)

* **Insight Generation Service:** Implement another Cloud Run service (`insight-generator`) that handles Pub/Sub messages from `insight-request`. It receives the fused state (and any additional context like user demographics if needed for personalization), then formulates the LLM prompt and calls Vertex AI. This service can be lightweight (no heavy ML libraries, just HTTP to Vertex and Firestore). After getting the LLM output, it writes the final insight result to Firestore.

* **Storage & Data Layer:** Use **Firestore** for storing processed results and small metadata. Proposed Firestore collections:

  * `proxy_vectors/{uid}/{uploadId}` for intermediate normalized time-series (e.g. the minute-by-minute z-score vector for steps, HR, etc.),
  * `embeddings/{uid}/{uploadId}` for ML model outputs like the PAT embedding,
  * `insights/{uid}/{uploadId}` for the final insight text/JSON and any summary metrics.
    Access to these is controlled by security rules so users can only read their own docs. Raw data stays in GCS (with bucket-level security).

All new code will follow the layering conventions of the project. For example, preprocessing logic might live in a new module `src/clarity/ml/preprocessing.py` (or under `domain` if treated as business logic), and the fusion model in `src/clarity/ml/fusion.py`. By keeping ML details in the `ml` or service layer and not in the API layer, we adhere to the Clean Architecture separation. This makes it easy to swap implementations (e.g. use a different model) and to **unit test** the logic in isolation (by injecting dummy repositories or models).

## Data Preprocessing for Each Modality

Each physiological signal from HealthKit is preprocessed using **current best-practice time-series cleaning** techniques (per 2024–2025 literature) before analysis. Preprocessing is crucial to handle noise, missing data, and differing sample rates. Key steps (applied as appropriate per modality) include **deduplication, interpolation, filtering, outlier removal, resampling, and normalization**:

* **Heart Rate (HR)** – Apple Watch provides heart rate in BPM, often at 5-second intervals during workouts or every few minutes at rest. We first **resample HR to a uniform timeline**, e.g. 1-minute averages, to align with other data. For cleaning, we drop obviously erroneous values (e.g. 0 or extreme spikes >220 BPM unless sustained) and fill short gaps via linear interpolation or forward-fill (for up to a few minutes gap). We apply a mild smoothing filter to remove high-frequency jitter; for instance, a moving average over 3–5 points can smooth noise while preserving true variability. Additionally, we can apply a **low-pass filter** (e.g. Butterworth with cutoff \~0.5 Hz) to remove motion artifacts from PPG signals. After cleaning, we **normalize HR** per individual or population norms: e.g. subtract the user's resting HR and divide by their HR range, or simply z-score using global mean±SD for adults (around 60–100 BPM resting range). This yields a time-series ready for model ingestion.

* **Heart Rate Variability (HRV)** – HealthKit provides HRV as SDNN (standard deviation of NN intervals) in milliseconds, typically computed over 1-minute or 5-minute windows. We take the HRV series (often one value per minute) and ensure alignment with the HR timeline (using the same minute bins, possibly by taking median if multiple values in a minute). HRV data can be noisy; we remove outliers (e.g. a single-minute SDNN spike that is 5× the median of neighbors, which could indicate a sensor glitch) to avoid skewing analysis. If HRV is sparse (e.g. only a few readings per day), we treat missing intervals by interpolation only if the gap is small; otherwise we mark them as missing so models can recognize lack of data. We may also compute **derived HRV metrics** like RMSSD or pNN50 from raw R-R intervals if available, but assuming HealthKit only gives SDNN, we'll use that directly. Finally, we **normalize HRV** by log-transforming or z-scoring (HRV distributions are often skewed; a log can stabilize variance). The aligned and normalized HR and HRV can be considered together in downstream processing.

* **Respiratory Rate (RR)** – Measured in breaths per minute, typically available as periodic samples (e.g. from Breath Rate in sleep or workout contexts). We clean RR by removing physiologically implausible values (valid resting range \~8–20, and during exercise up to \~40–50; any value outside 5–60 likely noise). We then resample to 1-minute or 5-minute intervals to match HR timeline, using averaging if needed. Short gaps can be interpolated; long gaps (when the watch isn't measuring RR) remain empty. A smoothing filter (e.g. a 3-point median filter) can be applied to remove transient spikes (for example, a momentary jump that doesn’t align with adjacent values). Normalize RR by z-scoring against an average resting rate of \~12 (or the user's baseline if known).

* **Blood Oxygen Saturation (SpO₂)** – Given as a percentage (usually 90–100% range on Apple Watch). SpO₂ is typically sampled on demand or periodically (not continuous). We treat SpO₂ readings as point samples. Preprocessing involves discarding obviously invalid readings (e.g. 0 or <80% which likely indicate sensor error unless the user has a medical condition). Because SpO₂ changes very slowly, we can linearly interpolate between daily measurements if needed, but generally we’ll treat each measurement as a valid sample at a timestamp. No complex filtering is needed beyond perhaps averaging if there are multiple readings in a short period. For integration, we might reduce SpO₂ to summary features (e.g. average SpO₂ over last night, lowest SpO₂ value observed in period). If feeding a time-series model, resample to daily or hourly frequency (with missing values for hours with no data). Normalize in percentage points (e.g. 95% as 0.95, and possibly no further normalization needed since the range is naturally bounded near \[0,1]).

* **Blood Pressure (BP)** – Often entered manually or via a periodic cuff measurement (systolic/diastolic in mmHg). These are discrete records (not continuous streaming). Preprocessing: validate each record (systolic in \~90–180, diastolic 60–120 for typical ranges; anything far outside might be data entry error). There’s usually no need to interpolate BP; instead we treat it as episodic data. We could generate features like *most recent BP*, *average of last N readings*, or *trend* if multiple readings over weeks. For fusion, we include the latest valid BP or an average, since continuous sequence modeling on BP is not applicable due to infrequent sampling. We ensure units (mmHg) are consistent and perhaps normalize by a normal BP (e.g. 120/80) if combining with other features, or simply scale systolic and diastolic separately (e.g. subtract 120 and 80 respectively, or express as deviations from normal).

* **Body Temperature** – Apple Watch (Series 8+) provides a **wrist skin temperature deviation** (often during sleep) rather than absolute body temp, and HealthKit might store basal body temperature. We take temperature readings in °C (or °F normalized to °C) and remove obvious errors (e.g. 0°C or jumps of >2°C in a minute). Usually temperature is logged as an overnight delta (like “+0.3°C above baseline”). We can simply take the median nightly temperature deviation as the daily value. Preprocessing might involve smoothing over a night’s readings and extracting a single representative value per day (since the watch gives variations overnight). We normalize temperature by considering the typical human skin temperature range (\~30–36°C) – for example, express deviations in °C or as z-scores if we have population stats. Temperature time series can then be aligned by day.

* **VO₂ Max** – Provided by Apple as an estimated VO₂ max (ml/kg/min) if the user does enough outdoor walking/running. This is updated only when workouts occur (maybe a few times a week). We treat VO₂ max as a **slow-varying fitness indicator**. Preprocessing is minimal: check for valid range (e.g. 15–80 ml/kg/min for adults; out-of-range indicates error). We then track changes over time – e.g. compute the delta from previous value to see trend. For weekly insight, the latest VO₂ max or the change from last month could be used. This can be included as a scalar feature. No time-series modeling needed due to low frequency; just ensure it's stored and updated. Normalization could map it to a fitness percentile or simply treat higher as better (we might include a categorical fitness level: poor/average/athlete based on age norms).

* **ECG Summaries** – Apple ECG app yields a classification (e.g. Sinus Rhythm, Atrial Fibrillation detected, Inconclusive) and heart rate at measurement. We preprocess by mapping the classification to an **enum or one-hot vector** (e.g. Sinus=0, AFib=1, etc.) and taking note of any AFib occurrences. For each ECG session, we store the classification and maybe the average HR during that 30-second ECG. For integration, we can have features like “AFib\_detected = True/False” and “ECG\_resting\_hr”. Since ECG readings are on-demand, we likely use the *most recent classification* or a count of abnormal readings in the period. No numeric normalization needed beyond categorical encoding.

* **Workouts & Activity** – While the system already supports actigraphy (steps) via the PAT model, we extend it to full workout data. Workouts come with rich data (type of workout, duration, calories, avg heart rate, distance, etc.). Preprocessing involves parsing workout records and possibly deriving summary stats: e.g. total number of workouts this week, total active minutes, average intensity (perhaps from heart rate zones). We ensure each workout’s metrics are plausible (e.g. no negative durations, calories in a reasonable range for the duration). For time-series fusion, high-level features from workouts (like daily active minutes or a binary flag of “worked out today”) can be aligned with daily activity. Steps data specifically is already handled via the **proxy actigraphy** logic: we convert per-minute step counts into a **movement vector** by √(steps) and then z-score using reference stats. This yields the 10,080-length weekly movement vector which PAT consumes. We will do the same for any new **activity metrics** (like distance): ensure they are expressed per minute or per day, and normalized.

After preprocessing, **all modalities are aligned to a common timeline** where possible. The simplest alignment is by minute (for high-frequency data like HR, HRV, steps) and by day for low-frequency data (like BP, VO₂ max). We produce a coherent snapshot of the past week (or relevant period) for the user where each signal is cleaned and ready. If certain data are missing for some days (e.g. no SpO₂ reading on a given day), that gap can either be filled with a neutral value or marked so the model knows it's missing (some advanced fusion models handle missing modality by design). Each signal is now represented in a machine-learning-friendly form – e.g. as a vector of length N (N=10080 for minute-level week data, or N=7 for daily values, etc.), or as a small set of summary features.

**References for preprocessing best practices:** We followed recent recommendations for wearable data cleaning, such as combining multiple techniques (filtering, outlier removal, imputation) to ensure robustness. For example, Liu *et al.* (2021) detect HRV peaks by band-pass filtering and moving average smoothing before feature extraction. The approach here similarly applies domain-specific filters (e.g. smoothing HR) and leverages known physiological ranges to clean data. By the end of this stage, the data are **high-quality, synchronized time-series** suitable for feeding into the specialized processors and models.

## Modular Signal Processing Components

With clean data prepared, the pipeline next processes each signal (or group of related signals) through modular **processor** classes. Each processor encapsulates the logic for feature extraction and ML inference on a particular modality category, promoting separation of concerns and easy testing. We define processors such as:

* **CardioProcessor:** Handles heart-related metrics – e.g. Heart Rate and HRV (and possibly blood pressure as it relates to cardiovascular health). This module takes the preprocessed HR and HRV time series (minute-level arrays for the week, or summary stats) and produces higher-level features or embeddings. Two approaches are considered: (a) use classical features – e.g. resting HR (lowest 5th percentile of the week), peak HR, average daily HR, HRV average, HRV trend – resulting in a feature vector of, say, 10–20 dimensions; and/or (b) leverage a small ML model to encode the time-series pattern. For instance, we can feed the 7-day HR sequence into a lightweight **time-series model** (such as a 1-D CNN or a transformer encoder) to generate an embedding that captures circadian patterns and variability. A recent technique, **PatchTST**, suggests splitting time series into patches and using self-attention to capture long-term dependencies efficiently; we could use a simplified version of PatchTST for HR/HRV. However, a simple MLP on summary stats might suffice initially. The CardioProcessor outputs a fixed-length vector (e.g. 16 or 32 dims) representing the user's cardiac state over the week (including aspects of fitness, stress, recovery as inferred from HR/HRV dynamics).

* **RespirationProcessor:** Deals with respiratory metrics – Respiratory Rate and SpO₂. It might take the minute-by-minute respiration series (after cleaning) and compute features like average awake RR, average sleep RR (if we can infer sleep periods), variability of RR, and any episodes of high breathing rate. SpO₂ might be distilled to minimum SpO₂ observed and average SpO₂. These features could be concatenated into, say, a 5-10 dimensional vector. If more sophisticated, we could imagine a small model that looks at nightly SpO₂ patterns or correlates drops in SpO₂ with high HR (potentially indicating stress or apnea events), but given sparse data, basic statistical features are likely enough. The output is a respiration state vector (few dims capturing oxygenation and breathing stability).

* **ActivityProcessor:** (If not already fully covered by PAT) Handles raw activity metrics beyond steps – e.g. workouts, active energy, distance. However, since the **Pretrained Actigraphy Transformer (PAT)** already provides an embedding from the weekly movement vector, we will incorporate PAT directly rather than duplicating its functionality. For context, the PAT model takes a 10,080-length minute-level activity vector (our “proxy actigraphy”) and outputs a 128-dimensional **embedding** that summarizes the user's activity and sleep patterns. We obtain this by calling the PAT model (either via an internal library or a microservice endpoint) with the z-scored movement vector. The PAT output includes a *CLS embedding* (128 floats) and possibly token-level embeddings or predictions. We use the CLS embedding as the representation of the user's overall activity/sleep state. (If workouts provide additional nuance, we could extend PAT’s input by marking workout periods in the input sequence or by augmenting the embedding with workout stats, but that is an advanced consideration beyond this immediate scope.)

* **OtherProcessors:** For modalities like **BloodPressureProcessor**, **ThermoProcessor** (temperature), or **ECGProcessor**, since their data are not dense time-series, these might be simple functions rather than complex models. For BP, we might output a 2-dim feature: normalized latest systolic and diastolic difference from ideal (e.g. \[ΔSys, ΔDia]). For temperature, perhaps one feature: average nightly deviation for the week. For VO₂ max, one feature: latest VO₂ max or trend (increase/decrease). For ECG, one-hot flags for any AFib detected. We bundle these smaller pieces into a vector we can call the "biometrics & vitals features". It could be on the order of 5-10 features total (depending on available data).

Each processor is implemented as a **self-contained class** with a common interface (e.g. a `process(data) -> feature_vector` method). This modular design means each can be developed and tested independently (e.g. CardioProcessor tests would feed synthetic HR data and verify the features or embeddings output). In code, these might reside in `clarity/ml/processors.py` or a `clarity/processors/` package, with classes `CardioProcessor`, `RespirationProcessor`, etc. They may share utility functions (like a common outlier detection util).

For the ML models inside these processors (like PAT or a small transformer for HR): we leverage **existing models or architectures** where possible. PAT is provided (with S/M/L weight files in `models/` directory) and can be loaded via TensorFlow or PyTorch in the ActigraphyProcessor (ensuring to use the same preprocessing as the PAT training: min-by-min vector, sqrt transform, z-score with NHANES stats). For other sequences (HR, etc.), if we choose to use a transformer or neural network, we could use a **PyTorch model** defined in `clarity/ml/models/` – e.g. a `SmallTimeSeriesTransformer` with a few layers of attention. Alternatively, since our dataset for these is smaller, a well-tuned MLP on statistical features may be more robust. The blueprint in the code suggests trying **PatchTST** for HR/HRV, which is a transformer approach that splits the sequence into patches and applies self-attention (Nie *et al.*, ICLR 2023). Whichever method, we ensure the input format matches what the model expects (padding or truncating sequences to fixed length, etc.). If needed, we can preload any pre-trained weights (though likely we will train these sub-models on internal or public datasets for vital signs once enough data is collected).

**Note:** By designing separate processors, we uphold the Single Responsibility Principle – e.g., the **CardioProcessor** is solely responsible for “making sense of heart-related data.” This maps well to potential separate microservices if we ever wanted (e.g. a dedicated heart analytics service). In this unified implementation, they are simply modules within the analysis service, but the architecture could scale out by deploying each processor on its own service subscribed to the queue (Pub/Sub allows multiple subscriptions). For now, we proceed with them running sequentially within one service for simplicity.

## Fusion Layer Design (Unified Health State Vector)

Once each modality-specific processor produces its feature vector or embedding, we need to **fuse** these outputs into a single **unified health state vector**. This vector will represent the user's overall physiological state in a form suitable for the LLM or any downstream analytic models. The fusion is implemented via a **transformer-style fusion head**, inspired by recent multimodal learning research that shows token-wise attention is effective for combining different modalities.

**Fusion input:** We have multiple modality representations of varying dimension: e.g. `actigraphy_embedding` (128-dim from PAT), `cardio_features` (\~16-dim), `resp_features` (\~8-dim), `other_features` (\~8-dim). We first project each of these feature sets into a **common embedding space** – e.g. use a linear layer to map each vector to a fixed length (say 32 or 64). This creates a set of modality tokens: `X_activity (1×d)`, `X_cardio (1×d)`, `X_resp (1×d)`, `X_other (1×d)`, all of equal dimension *d* (e.g. 64). We then form a sequence of these modality embeddings.

**Transformer fusion head:** We feed the sequence of modality embeddings into a small **Transformer encoder**. The transformer uses self-attention to allow each modality to attend to others, effectively learning which signals corroborate or contextualize each other. For example, if the model needs to understand “stress,” it might attend to both high HR and poor sleep (actigraphy) simultaneously. Using multi-head attention in this way provides *context-aware alignment* between modalities, which tends to outperform simple concatenation (implicit early fusion). We include a special learnable `[CLS]` token at the beginning of the sequence (much like BERT) which will serve as the **fusion output token**. After a few self-attention layers (even 1–2 layers can suffice given few tokens), the `[CLS]` token's final embedding is taken as the **unified health state vector** (for instance, 64-dim).

We might also experiment with cross-attention fusion where one modality is treated as primary and others as context, but the symmetric transformer approach with a CLS token is straightforward and general. **Modality-specific projections** (the linear layers) ensure that no single modality with larger original dimension dominates the representation by volume, achieving representation homogeneity. If some modalities are missing (e.g. no BP data), we can either omit that token or use a placeholder token (the transformer can learn to ignore it). The output vector can be thought of as encoding the salient health features of the week: e.g. overall activity level, sleep quality, cardiovascular status, recovery state, etc., all distilled.

We will likely set *d* (the fusion latent size) relatively small (64 or 128) to keep the model lightweight. The number of transformer layers can be 1–2 (with say 4 attention heads) given the small number of tokens. This **fusion model** can be trained or fine-tuned in the future on known outcomes (like health score prediction) to optimize its weights, but initially, we can use it in an unsupervised manner: essentially it will just propagate the inputs. Even untrained, a transformer with fixed random weights and a CLS token can serve to concatenate information (though training would be ideal). In absence of training data, we might choose to simply concatenate all feature vectors into one long vector and then apply a feed-forward dense layer to mix them (which is equivalent to one transformer block without attention). However, to follow best practice, we propose the transformer head for its flexibility and proven effectiveness in multimodal tasks. Indeed, recent studies confirm the *effectiveness of transformer-based multimodal fusion* for medical data classification, and our design aligns with those findings.

**Output:** The unified vector (say 64-dim) is the final numeric representation of the user's state. We can optionally attach interpretation to some dimensions (if we had built it that way), but primarily it’s for consumption by the LLM or any rule-based insight logic. This vector could also be stored in Firestore (`embeddings/{uid}/{uploadId}`) for future reference or for answering ad-hoc questions via vector search if we had such features.

To summarize, the **fusion layer** ensures that the **outputs from all processors (including the PAT actigraphy embedding) are integrated** into a single representation. This mirrors the concept of a "digital twin" or comprehensive health embedding that holistically reflects the user’s condition. By using a transformer-style approach, we allow the model to learn correlations between modalities (for instance, recognizing that a spike in heart rate during sleep corresponds with an actigraphy-detected wake episode). This design is state-of-the-art as of 2025 and prepares the system for easy extension: if a new modality (e.g. continuous glucose monitor data) comes in the future, we can add a processor and another token to the fusion with minimal changes.

## Insight Generation via Gemini 2.5 LLM

With the unified health state vector computed, the next step is to generate **user-friendly insights** from it using an LLM (Gemini 2.5 from Google Vertex AI). We implement a **narrative generation module** that takes structured data (the state vector and possibly key derived stats) and produces a conversational summary for the user. The emphasis is on clear, helpful explanations with no sensitive identifiers, thereby maintaining privacy compliance.

**Prompt Engineering:** We use a **structured prompt** with a system and user message to guide the LLM. The **system prompt** defines the role and rules for the model, for example:

* System role: `"You are a health AI assistant that explains a user's weekly health data in simple terms. You never include personal identifiers or protected health info. Maintain an encouraging and factual tone. Provide actionable advice based on the data."`

This sets the context that the LLM should be HIPAA-conscious (no PHI leakage) and positive/supportive in tone. We will also configure Vertex AI safety settings to high, to filter any undesirable output.

The **user prompt** (or data prompt) will include the structured stats. We do not feed the raw vector directly (as it’s not human-readable); instead, we convert the salient parts of the state into text or a JSON that the model can interpret. One approach is to enumerate key insights in text form, for example:

```
User data summary:
- Average resting heart rate: 60 bpm (excellent)
- HRV (SDNN) median: 55 ms (slightly low)
- Average respiratory rate: 18 breaths/min (normal)
- SpO2: 96% (normal)
- Blood pressure: 130/85 mmHg (slightly elevated)
- Sleep duration: ~7h/night, 2 disruptions on average
- Activity: 8,000 steps/day, 3 workouts this week
- PAT model mood score: 0.8 (low risk of depression)
```

Then ask the model: **"Generate a brief health report for the past week, explaining these metrics and giving recommendations."**

However, to maximize reliability and structure, we will instruct the LLM to output a **JSON** with specific fields. For example, we might ask it to produce:

```json
{
  "summary": "<one-paragraph overview>",
  "details": {
     "heart_health": "<insight about heart rate and BP>",
     "activity": "<insight about activity and workouts>",
     "sleep": "<insight about sleep patterns>",
     "respiratory": "<insight about breathing or SpO2>"
  },
  "recommendations": [
     "<tip 1>",
     "<tip 2>"
  ]
}
```

By requesting JSON, we ensure the output is structured and machine-parseable for the app. The system prompt will reinforce this format (e.g. "Respond only in the following JSON format..."). The **user prompt** will contain the actual values from the analysis service: we plug in the numbers and facts. For instance:

```
System: You are a health assistant... (rules and JSON format instructions).

User: The user is a 35-year-old male. Here is the weekly health data:
- Resting HR: 60 bpm
- Max HR: 170 bpm
- HRV SDNN median: 55 ms
- Blood Pressure: 130/85 mmHg
- Avg SpO2: 96%
- VO2max: 42 ml/kg/min
- Sleep: 7.0 hours/night, sleep efficiency 85%
- Activity: 8000 steps/day, 3 workouts (2 running, 1 cycling)
- Weight change: -0.5 kg this week

Generate a JSON with summary, detailed insights, and 3 recommendations.
```

We will leverage templates from research and prior work. In fact, the project docs already include a structured prompt example for PAT outputs, which we can adapt. In the PAT research prompt, they list model predictions (e.g. depression risk, sleep score) and instruct the LLM to explain patterns, give recommendations, be encouraging, and respect privacy. We will follow that pattern: our prompt explicitly lists the data points and then states the requirements (explain patterns, actionable advice, positive tone, privacy). This approach is **rooted in 2025 best practices** for prompt engineering, ensuring the LLM knows exactly what format and content to provide.

**LLM Call Configuration:** Using Vertex AI’s API, we specify the **Gemini 2.5** model (assuming it’s available as `model="gemini-2.5-pro"` or similar) with a **low temperature** (e.g. 0.2–0.3) for deterministic, factual output, and appropriate `max_tokens` (perhaps 1024 to allow a detailed report). We also set `top_p` moderately (0.8) to balance coherence, and enable the highest **safety settings** since health content must be accurate and not harmful. The Insight Generator service will call the Vertex AI Chat API with these parameters, sending the system and user prompt messages, and await the response.

**Example Prompt Template (pseudocode):**

```python
system_prompt = """You are a health insights assistant. 
Follow these rules:
- Output JSON only, with keys: summary, details, recommendations.
- Use simple language, explain what each metric means for health.
- If data seems off or not provided, say "not enough data".
- Do NOT include any personal names or identifiers.
- Tone should be positive and encouraging, but honest.
"""  

user_prompt = f"""The user is a {age}-year-old {gender}. 
Weekly health summary:
Heart: Resting HR {resting_hr} bpm, HRV SDNN {hrv} ms, BP {bp_s}/{bp_d} mmHg.
Activity: {steps_per_day} steps/day, {workouts} workouts.
Sleep: {sleep_hours} hours/night, efficiency {sleep_eff}%.
Respiratory: avg RR {rr} breaths/min, SpO₂ ~{spo2}%.
Weight: {weight} {weight_unit} (change {weight_change}).
Please analyze and provide insights and 3 recommendations in JSON."""
```

This prompt will yield an output like (for example):

```json
{
  "summary": "Your overall health this week was good. You stayed active with regular exercise and maintained a healthy resting heart rate. Sleep was adequate, though a bit fragmented on some nights.",
  "details": {
    "heart_health": "Your resting heart rate (60 bpm) is excellent, indicating good cardiovascular fitness. Your blood pressure (130/85) is slightly above optimal; it's something to watch but not alarming. HRV was a bit low, which can happen due to stress or lack of recovery.",
    "activity": "You averaged 8,000 steps per day and completed 3 workouts. This level of activity is decent – meeting basic guidelines. The runs and cycling are boosting your fitness (VO2 max ~42, which is good). Keep it up!",
    "sleep": "You slept ~7 hours nightly with 85% efficiency. Generally good sleep, though a couple of nights had some interruptions. Consistent sleep helps your HRV, so try to maintain a regular schedule.",
    "respiratory": "Your oxygen saturation was around 96%, which is normal. Breathing rate averaged 18/min, also normal. No signs of respiratory issues in the data."
  },
  "recommendations": [
    "Continue regular exercise; try to hit 10,000 steps on more days for additional benefit.",
    "Keep an eye on blood pressure – consider relaxation techniques or reducing salt intake to bring 130/85 closer to 120/80.",
    "Maintain good sleep hygiene (consistent bedtime, limit screen time before bed) to improve recovery and HRV."
  ]
}
```

The above JSON is an **example output** demonstrating the desired structure and content. It provides a summary and then segmented insights, followed by concrete tips. The tone is encouraging and non-alarming, even when mentioning slightly elevated BP, aligning with a coaching style rather than a medical diagnosis.

We will ensure the **LLM response is validated** before storing: e.g., we could run a quick check that the JSON parses and contains the expected keys (summary/details/recommendations). Any obviously inappropriate or PHI content should be filtered by Vertex AI’s safety, but we'd also review format. Given the sensitive nature of health data, we log the prompts and outputs in a secure log (without exposing them to unauthorized parties) for auditability.

**HIPAA and PHI in LLM:** It's crucial that no personally identifiable info is sent to the LLM. In our design, the prompt contains only aggregate health stats and perhaps age/gender – which are not identifiers but still considered quasi-identifiers. Google’s Vertex AI under a BAA ensures the data stays in a compliant environment, and we also avoid including things like names or exact timestamps that could be identifying. The LLM is instructed not to produce PHI either (e.g. it should say "the user" or "you" rather than a name). This structured, minimal approach keeps us on the right side of privacy regulations while harnessing the power of generative AI for user insights.

Finally, once the **Insight Generator** has the LLM's JSON output, it writes it to Firestore (`insights/{uid}/{uploadId}` document). The mobile app will pick this up (perhaps via a listener on the `insights` collection) and display it in the chat interface. The insight text can be shown as a message from the "AI coach" in the chat UI. If the user asks follow-up questions, we can handle those by either retrieving relevant stored data or re-prompting the LLM with the stored summary as context – but those interactive Q\&A aspects are beyond the scope of this batch-processing blueprint. For now, we deliver the **automated weekly report** which itself is a major feature (and aligns with what Apple’s rumored AI health coach aims to do by “advising users based on device data”).

## Google Cloud Infrastructure (Terraform IaC) Checklist

To deploy this solution, we provision the required GCP resources with Terraform (or equivalent IaC). Below is a checklist of infrastructure components and configurations needed, with key settings:

* **Pub/Sub Topics & Subscriptions:**

  * Create a Pub/Sub **Topic** named **`health-data-upload`** for incoming health data processing jobs.
  * Create a **Subscription** on this topic for the Analysis Service. Use a **push subscription** with the Cloud Run Analysis Service URL as the endpoint. Enable **OIDC authentication** on the push: configure the subscription's `push_config` to include an `oidc_token` with a service account (e.g. `pubsub-push-sa@<project>.iam.gserviceaccount.com`). This causes Pub/Sub to send a JWT in the `Authorization` header of each request.
  * Grant the Pub/Sub service account the `roles/iam.serviceAccountTokenCreator` role on the specified push service account (so Pub/Sub can sign tokens as that SA). Also ensure the Cloud Run service is set to **require authentication** (Cloud Run IAM setting) so that only calls with a valid token (i.e. from Pub/Sub) are accepted.
  * Similarly, create a Pub/Sub **Topic** `insight-request` for triggering LLM processing. Create a push **Subscription** on it for the Insight Generator Cloud Run service, again with OIDC auth. Use a separate push service account for this if desired (or reuse the same process with appropriate roles).

  Terraform snippet (for illustration):

  ```hcl
  resource "google_pubsub_topic" "health_data" {
    name = "health-data-upload"
  }
  resource "google_pubsub_subscription" "health_data_sub" {
    name  = "health-data-upload-push"
    topic = google_pubsub_topic.health_data.name
    push_config {
      push_endpoint = var.analysis_service_url  # Cloud Run URL
      oidc_token {
        service_account_email = var.pubsub_push_service_account
        audience = var.analysis_service_url     # URL as audience
      }
    }
  }
  ```

  (Equivalent resources for `insight-request` topic and subscription.)

* **Cloud Storage (GCS) Bucket:**

  * Provision a GCS bucket for raw health data, e.g. **`healthkit-raw-data`**. Enable default encryption with a **Customer-Managed Encryption Key (CMEK)** if required for HIPAA compliance (or at least ensure Google-managed encryption by default, which is given). Set bucket-level permissions so that only the ingestion service account (and perhaps analysis service account) can write/read as needed. For instance, the FastAPI service SA gets `Storage Object Creator` on this bucket (to upload), and the Analysis SA gets `Storage Object Viewer` (to download files). Also set lifecycle policies if you want to auto-delete raw files after some retention (the user profile’s dataRetention could specify how long raw data is kept). Terraform example:

  ```hcl
  resource "google_storage_bucket" "healthkit_raw" {
    name          = "healthkit-raw-data"
    location      = "US"  
    force_destroy = false
    uniform_bucket_level_access = true
    lifecycle_rule {
      condition { age = 365 }
      action { type = "Delete" }
    }
    versioning { enabled = false }
    # Add encryption block if using CMEK:
    # encryption { default_kms_key_name = var.cmek_key }
  }
  resource "google_storage_bucket_iam_member" "ingest_write" {
    bucket = google_storage_bucket.healthkit_raw.name
    role   = "roles/storage.objectCreator"
    member = "serviceAccount:${var.ingest_service_account}"
  }
  resource "google_storage_bucket_iam_member" "analysis_read" {
    bucket = google_storage_bucket.healthkit_raw.name
    role   = "roles/storage.objectViewer"
    member = "serviceAccount:${var.analysis_service_account}"
  }
  ```

* **Cloud Run Services:**

  * **Ingestion API**: Already likely exists (FastAPI service). Ensure it’s deployed with appropriate environment (e.g. the Google credentials for Firestore and GCS). Give it permission to publish to Pub/Sub (assign `roles/pubsub.publisher` on the `health-data-upload` topic’s project) and to write to GCS (`roles/storage.objectCreator` as above). This service should use a dedicated service account (for clear auditing). In Terraform, you'd deploy via `google_cloud_run_service` resource or via Cloud Build triggers. Key config: set **Concurrency** to 1 (if using sync code) or higher if using async FastAPI effectively; set **CPU Always** on if needed for heavy compute (though ingestion itself is light).

  * **Analysis Service**: Deploy as a Cloud Run service (perhaps using Cloud Run Jobs if it were one-off, but here we want a server to handle push). Give it the service account `analysis_service_account`. This SA needs `roles/storage.objectViewer` (to read GCS) and `roles/pubsub.publisher` on the `insight-request` topic, and permission to write to Firestore (`roles/datastore.user` if using Firestore in Datastore mode, or appropriate Firestore role). Also allow it to use the Pub/Sub push JWT verification – essentially Cloud Run will automatically verify if we set the Cloud Run service to **"Authentication required"** and allow the Pub/Sub push SA as an Invoker. Terraform example:

    ```hcl
    resource "google_cloud_run_service" "analysis" {
      name     = "analysis-service"
      image    = "gcr.io/ourproj/clarity-analysis:latest"
      max_instances = 20
      template {
        spec {
          service_account_name = var.analysis_service_account
          containers {
            # ...
            env = [
              { name = "PROJECT_ID", value = var.project_id },
              { name = "PAT_MODEL_PATH", value = "/models/pat/PAT-M_29k_weights.h5" }
            ]
            resources {
              limits = { memory = "4Gi", cpu = "2" }
            }
          }
        }
      }
    }
    # Allow Pub/Sub to invoke Analysis (Cloud Run IAM):
    resource "google_cloud_run_service_iam_member" "analysis_invoker" {
      service = google_cloud_run_service.analysis.name
      location = google_cloud_run_service.analysis.location
      role    = "roles/run.invoker"
      member  = "serviceAccount:${var.pubsub_push_service_account}"
    }
    ```

    We may also attach a volume for the `models/` directory if bundling PAT weights, or we can bake them into the container. Ensure the container has access to CPU and memory to load the model (Large PAT \~7.6MB, which is fine). Set the **timeout** sufficiently (Cloud Run default 5 minutes; we might set 10 or 15 just in case, though typical processing should be under a minute or two). Enable logging.

  * **Insight Generator Service**: Cloud Run service with the `insight_service_account`. This needs permission to call Vertex AI API (usually the default service account has this if Vertex AI is enabled on project) and to write to Firestore. Also `pubsub.subscriber` on the `insight-request` subscription if it's pull, but for push, we just need to set up the push as we did for analysis. So similarly, Terraform a Cloud Run service, set its invoker to Pub/Sub push SA, etc. It will be a lightweight container (possibly Python FastAPI or just a Cloud Run job triggered via Pub/Sub). We configure environment variables for Vertex AI (like model name, project, region) as needed.

* **Vertex AI**: Ensure the **Vertex AI API is enabled** in the project. No Terraform resource is needed for using a public model, but if we had custom models, those would be configured here. We might set Vertex AI quotas if needed (the docs mention referencing quotas in external links, but presumably, default quotas suffice unless heavy usage). For compliance, ensure we have a BAA in place with Google (enabling **Google Cloud Identity Platform** for Firebase Auth as mentioned, and using Vertex under that umbrella, which it is).

* **Firestore**: The Firestore database should already exist (if using). We ensure **Firestore Security Rules** restrict `insights/{uid}/*` docs to only be read by that `uid`. If not already in place, write rules accordingly. Also consider an index if we plan queries (e.g. listing insights by date). In our case, direct key access by `uploadId` might suffice, so no composite index needed.

* **IAM Roles & Service Accounts**: We enumerated most above. To recap:

  * Create a **Service Account** for each Cloud Run service: `ingest-sa`, `analysis-sa`, `insight-sa`, and one for Pub/Sub push JWT (`pubsub-push-sa`). Principle of least privilege:

    * `ingest-sa`: roles/storage.objectCreator on raw bucket, roles/pubsub.publisher on health-data-upload topic, roles/datastore.user (if writing small metadata to Firestore).
    * `analysis-sa`: roles/storage.objectViewer on raw bucket, roles/pubsub.publisher on insight-request topic, roles/datastore.user (to read/write Firestore for embeddings or status).
    * `insight-sa`: roles/pubsub.publisher (if it needed to ack back or publish something, probably not), roles/datastore.user (to write insight to Firestore), and importantly whatever role allows Vertex AI API invocation (usually Vertex AI invoker or just being Editor might cover it; to be specific: `roles/aiplatform.user` could be used for custom endpoints; for using a managed model, the default service account might suffice).
    * `pubsub-push-sa`: no special roles needed except we grant token creator to Pub/Sub service as described.
  * **Pub/Sub service agent**: The auto-created Pub/Sub service account needs `iam.serviceAccountTokenCreator` on `pubsub-push-sa`.
  * All service accounts: enable Cloud Logging write (usually default) so they can log.

* **Terraform State & CI**: Ensure these resources are added to Terraform scripts. After deploying, we should have the pipeline wired: uploading data will create a Pub/Sub message and so on. We might include Terraform outputs for service URLs (to configure push endpoints easily).

This setup is fully **Infrastructure-as-Code** so that it can be reviewed, versioned, and deployed consistently in different environments (dev/staging/prod). It covers all cloud pieces: API, storage, queue, compute, ML, database, and security configurations.

## Security & Compliance Considerations

Handling health data mandates strict security and privacy compliance (HIPAA in the US). This design incorporates multiple layers of protection and auditing:

* **Authentication & Authorization:** All API calls from the app include a Firebase Auth ID token and are verified on the backend (using Firebase Admin SDK). This ensures only authenticated users can send or retrieve health data. Every request is tied to a user UID, and code checks that the user is only accessing their own data. We leverage Google Cloud Identity Platform (as an extension of Firebase Auth) to be covered under Google's HIPAA Business Associate Agreement (BAA).

* **Data in Transit:** Use HTTPS for all client-server communication. Internally, use secure protocols for service-to-service (Cloud Run to GCS, Pub/Sub, etc.). Pub/Sub push is configured with **OIDC JWT authentication**, meaning the Analysis and Insight services only accept messages from Pub/Sub with a valid signed token. This prevents malicious HTTP calls from triggering our pipeline. All data sent to the LLM (Vertex AI) stays within Google’s cloud and is encrypted in transit to the Vertex endpoint.

* **Data at Rest:** All storages (Firestore, GCS, Pub/Sub) encrypt data at rest by default. We further ensure the GCS bucket with raw data is locked down: enabling **CMEK** for that bucket so that even Google staff cannot read it without our key (if required by policy). Firestore is already encrypted and we use security rules to restrict read/write. We do **not log** sensitive payloads or PHI in plaintext. Logs will contain maybe high-level events (e.g. "HR data processed for user X at time Y") but not the actual heart rate values. If debugging requires some data, we can log it to a **secops-only log sink** with proper access control. We explicitly avoid any PHI in normal application logs.

* **Least Privilege IAM:** Each microservice has its own service account with only the permissions it needs (as detailed in the IaC section). This way, if one component is compromised, it can’t exfiltrate data from unrelated areas. For example, the Insight service can write to Firestore but cannot read raw GCS files; the Analysis service can read raw data and write derived data but cannot directly call the LLM (if we separate those roles).

* **Audit Trail:** All accesses and operations can be audited via Cloud Logging and Firestore. We log which user’s data was processed, when, and by which service. For instance, the analysis service can log an entry "Computed PAT embedding for user UID123, uploadId U20250601T071557Z" and similarly for insights. Cloud Run automatically logs requests; combined with Firebase Auth info we can trace requests user-wise. We enable **Cloud Audit Logging** for GCS and Firestore as well, which records any administrative access or permission changes. These logs together form an audit trail that can be used to demonstrate compliance and to investigate any incident.

* **HIPAA Alignment:** We choose Google Cloud services that are covered by the Google Cloud BAA: Cloud Run, Cloud Storage, Pub/Sub, Firestore, and Vertex AI (when used properly with de-identified data) are all services that can be made HIPAA-compliant. Vertex AI specifically, we ensure only minimal necessary information is sent (no direct identifiers, mostly statistics). The output from Vertex AI (the narrative) is also considered derived data; we treat it as PHI since it pertains to health, but it generally won't include identifying info. All PHI is stored and transmitted in encrypted form. Backups or exports of data will also be handled under the same policies (for example, if we backup Firestore, that bucket should be secure).

* **Monitoring & Alerts:** We implement monitoring to detect any abnormal usage. For example, if an IP outside of our expected range calls the ingestion endpoint frequently (potential intrusion) or if the number of Pub/Sub messages spikes unexpectedly (could indicate abuse or bug), we get alerted. We can use Cloud Monitoring to set up alerts on error rates, latencies, and security rule violations. Also, any denied accesses (e.g. a user trying to read another’s doc) should be logged by Firestore rules – those can trigger alerts if they ever occur (since in normal operation they should not).

* **Data Minimization:** The pipeline is designed to **minimize sensitive data handling**. We do not expose raw health data to the client once uploaded – only processed insights. The LLM sees only aggregates, not raw timelines. We keep raw data only as long as needed for processing (potentially deleting or archiving it after generating insight, depending on retention policy). By reducing the surface area of raw PHI exposure, we lower risk.

* **Testing for Security:** We will conduct thorough testing (described below) including attempting to break the auth, ensuring no data leaks via logs, and verifying that without proper auth tokens, nothing works (e.g. try calling the analysis endpoint without valid JWT and expect a 401). We also ensure compliance by code reviews focused on security (e.g. check that all external calls use HTTPS, all secrets are stored in GCP Secret Manager or env vars not code, etc.). Penetration testing can be performed on the API endpoints to catch any vulnerabilities.

In summary, the architecture not only meets functional needs but also creates a **secure enclave for health data** processing. By using managed GCP services with built-in security and configuring them carefully (IAM, OIDC, CMEK), and by keeping PHI handling to a minimum, we aim to comply with HIPAA and similar regulations. Every access is authenticated, every byte is encrypted, and every action is logged. The Clean Architecture approach even aids compliance by centralizing critical checks (no scattered ad-hoc auth logic).

## Testing Strategy and Quality Assurance

To ensure the solution is robust and meets the ≥90% test coverage goal, we devise a comprehensive testing plan covering unit tests, integration tests, and system tests. We also include testing for performance and security.

**1. Unit Tests:**
Every module (preprocessing, processors, fusion, LLM prompt, etc.) will have targeted unit tests:

* *Preprocessing functions:* For each modality’s cleaning logic, create small arrays with known issues and expected outcomes. For example, an HR sequence with an outlier 300 BPM value – after `clean_heart_rate()` we assert that this value is removed or capped. Test interpolation: if we feed HR with a gap, does it fill correctly? Test the √steps transformation: input \[0,1,4,9] steps should output \[0,1,2,3] after sqrt (simple case). Also verify z-scoring: if we use a dummy mean and std, check that output has mean \~0, std \~1.
* *Pydantic models:* Test that invalid inputs are rejected. For instance, create a `HeartRateSample(bpm=-5)` and expect a validation error (negative BPM not allowed). Test enum serialization (e.g. `HealthMetricType("HRV_SDNN")` works, and bogus types fail).
* *Processor classes:* Use **dependency injection** or monkeypatching for any model calls to test logic without needing the actual model. For CardioProcessor, we might stub out the internal transformer to just return a known vector, then verify that CardioProcessor correctly concatenates features with the model output if applicable. For the ActigraphyProcessor (PAT), since we might not load 100MB of models in a unit test, we can monkeypatch the PAT prediction call to return a fixed embedding for known input. Verify that the class handles various input lengths (pad/truncate logic). Also test edge cases like all-zero input vector, or missing modality (CardioProcessor given no HR data should perhaps return a zero-vector or raise a specific exception).
* *Fusion logic:* If we implement a transformer fusion, that might be tricky to unit test without the model, but we can at least test the shape math. For example, create dummy modality vectors, run them through our `FusionHead` module (if it's just a function concatenating or a simple attention forward pass) and ensure the output vector has expected length. If we have trainable weights, we might instantiate the module and do a forward pass with deterministic weights (e.g. set all projection matrices to identity or something for test) to verify that conceptually concatenated input yields matching output. Alternatively, if fusion is just concatenation + linear, we can test that linear layer is applied (maybe by injecting a fake weight matrix and comparing).
* *LLM prompt formatting:* We can write tests for the prompt builder to ensure that given certain input data, the prompt string contains the expected fields and no disallowed content. If using a templating function, feed known values and assert the string includes those values in the right places (e.g. age, HR values). We can also simulate an LLM response (a fake JSON string) and test our JSON parser on it to ensure the Insight Generator correctly handles the output (for example, if the LLM returns invalid JSON or missing keys, does our code catch it and handle gracefully?).

These unit tests will use fixtures for sample data. We can include a **small sample JSON** from HealthKit (perhaps from Apple's sample data or a fabricated week of data) as a file in the test resources, then step through the pipeline functions on it. For instance, feed the entire sample through preprocessing and ensure no exceptions and that we get expected summary stats.

**2. Integration Tests:**
Integration tests target the interaction between components:

* *End-to-end local test:* Spin up the FastAPI app (ingestion) with test client, the analysis logic (maybe in a thread or by calling the function directly), and a fake LLM service (since we don’t want to call the real Vertex in tests). We can simulate Pub/Sub by directly invoking the analysis function with a test message. For example, use the FastAPI test client to `POST /v1/health-data` with a sample payload, then intercept the Pub/Sub publish call (possibly by monkeypatching the Pub/Sub client in tests to just call analysis immediately or put message in a local queue). Then verify that an insight was generated in the end. This is complex, but we can break it: test that ingestion returns a 200 and writes a file (we can use a tmp local filesystem or mock GCS). Test that after ingestion, our code would publish a message – we capture the message payload and directly call analysis with it. The analysis then should produce an insight – since we won't call a real LLM, we can monkeypatch the `InsightGenerator.call_vertex()` method to just return a dummy insight (or we run the Insight generator code with a small local model or a stub).
* *Cloud environment integration:* Use the **Firestore emulator** and **Pub/Sub emulator** for a closer-to-real integration. This could be set up in CI: start the emulators, set env vars so that our code uses emulator endpoints (Firestore emulator via GOOGLE\_CLOUD\_FIRESTORE\_EMULATOR\_ADDRESS, Pub/Sub emulator via PUBSUB\_EMULATOR\_HOST). Then run a test scenario: call ingestion API (pointing to local FastAPI instance), it writes to a fake GCS (we might substitute GCS with a local file or memory since emulator for GCS isn't straightforward – alternatively, abstract storage in code so we can use a local disk for tests). Then have the ingestion code publish to the Pub/Sub emulator, which our analysis code can subscribe to or we manually pull from. This level of testing ensures our Pub/Sub message format is correct and our message handling works in a realistic way. We then verify that the final Firestore emulator has an `insights` entry with expected content.
* *Performance tests (small scale):* Not full load testing, but ensure our processing functions can handle the maximum expected input size. For example, generate an extreme test: a week’s worth of data at 1-second resolution (overkill, but to see if any part blows up) or multiple modalities filled with random values, then run through preprocessing and ensure it finishes under a time threshold (say a few seconds) and memory usage is reasonable. We can use Python `time` and `memory_profiler` in a test to gauge this. The Cloud Run limit is 15min per request – we expect to be far below that (likely <30s). We might assert that processing 1 week of data with all modalities enabled finishes in e.g. <5 seconds in a single-thread environment (this might vary depending on PAT model speed, which is \~450ms for medium model on CPU).
* *PAT model integration:* If feasible, test loading the actual PAT model (Small or Medium) and run a known input through it. We might use the example from the PAT paper or known patterns to ensure the output embedding isn't totally off (though without the original model reference, we may not know expected values; we can at least test the shape and that it produces finite numbers). Ensure the model can load in our environment (catch issues like missing dependencies, etc.). This integration test ensures that our packaging of the model files and loading code is correct.

**3. End-to-End Testing in Staging:**
Once deployed to a staging environment (with all cloud resources), perform an end-to-end test:

* Use a test user account, call the production-like API with sample data (perhaps using a curl or a small script with Firebase token). Then observe via logs or Firestore if the insight appears. This is more of a QA step but should be scripted if possible. We can even automate part of this with a Cloud Build job or integration test suite that deploys ephemeral environment and runs the scenario.

**4. Test Data & Fixtures:**
We'll prepare **fixture data** representing realistic user scenarios:

* A "normal week" dataset (moderate activity, good vitals).
* An "edge case week" (e.g. user was mostly inactive, or HR sensor had a day of missing data, or user has an outlier high BP reading).
* Specific edge cases like: HR values at sensor limits, a day with no data (to test how we handle empty inputs).
* Possibly use known public datasets for validation: e.g. synthetic data that mimics patterns from literature (some papers provide sample actigraphy or vital sign data; we can incorporate that to see if our pipeline can process it without error).

**5. Test Coverage and CI:**
We will aim for ≥90% code coverage. We can enforce this via CI (e.g. using `pytest --cov` and failing if coverage <90%). The critical logic (preprocessing, processors, fusion, prompt) will be well-covered by unit tests. Some integration code (like actual Cloud Run handlers) might be trickier to cover, but we can simulate requests to those handlers in tests. The CI pipeline (GitHub Actions or Cloud Build) will run the test suite on every PR. We also include linting (flake8/black) to maintain code quality.

**6. Performance & Load Testing:**
Although user-by-user processing is not heavy (each analysis is maybe a few seconds, and Pub/Sub/Cloud Run can scale horizontally), we should simulate load to ensure no bottlenecks:

* Simulate, say, 50 users uploading simultaneously in a test environment and see that Pub/Sub queues them and Cloud Run scales up instances to handle them. We might use a script to send 50 requests and then check that insights for all 50 are generated. This isn't exactly a unit test; it's more of a staging load test. We expect Pub/Sub to buffer and Cloud Run to scale to meet demand (we can verify from logs or metrics that scale-out happened).
* Memory test: Confirm that loading PAT model and processing multiple requests in sequence doesn’t lead to memory leaks. This can be done by a long-running test that calls the analysis code repeatedly (simulate 100 analyses sequentially in one process) and monitor memory usage to ensure it stabilizes.

**7. Security Testing:**
We incorporate tests for security aspects:

* Attempt unauthorized access: call ingestion without a token, expect 401. Call ingestion with a valid token but for one user and try to access another user’s data (if there’s any such endpoint), expect 403.
* Test Pub/Sub auth: this is harder to unit test locally, but we can mimic the behavior by calling the analysis endpoint with a fake JWT vs a real signed JWT. Since verifying a Google-signed OIDC token might involve Google’s certs, in test we might disable auth verification or use a dummy. However, we can at least test our handler requires the header. For example, if our push handler uses a decorator to check auth, we can test that providing no header results in 401.
* Penetration test basics: ensure no obvious XSS/SQLi (not too relevant as our inputs are numeric mostly, and we use ORMs/Pydantic for validation). Ensure large inputs are handled (what if someone uploads 100 years of data by mistake? Our code should perhaps truncate or handle it gracefully, not crash).
* Verify that sensitive data isn't accidentally logged: we could inspect the log outputs in tests by capturing logger outputs. For instance, run an ingestion and see if any log contains raw health values or user identifiers that it shouldn't. We may add explicit tests that our logging statements (if any in processing) use placeholders or redacted data.

**8. Compliance Testing:**
Though not code tests per se, we'll do checks like:

* Run the solution through a **HIPAA compliance checklist**. E.g., ensure all data stores have encryption, ensure all services have proper access controls. We already know these from design, but it's good to verify after deployment (Cloud Config can be validated via scripts or manual review).
* If possible, use automated tools (like Forseti or GCP security scanner) on the project to catch open firewall rules, overly broad IAM, etc. As part of QA, ensure everything aligns with our security intent.

By covering the above, we will have a high confidence in the system. The combination of unit tests for logic correctness and integration tests for the pipeline will guard against regressions. For example, if someone modifies the preprocessing for HR, a unit test failing will catch if they break expected behavior. If someone changes a Pub/Sub message format, an integration test will catch a downstream failure in the insight service expecting a different field. We will also maintain test data to reflect current Apple HealthKit formats, updating tests if Apple changes their schema (hence why having our own stable intermediate schema via Pydantic helps isolate such changes).

Crucially, we aim for tests not just of the “happy path” but also all edge cases: no data, all zeros, extremely high values, inconsistent timestamps, etc. This thorough approach ensures the system is **reliable, safe, and accurate** in producing insights. Given the importance of user trust in a health app, quality assurance is paramount – our testing regime reflects that.

## Developer Implementation Tasks (🛠)

Finally, here is a step-by-step task list for developers to implement this blueprint, in logical order. Each task is marked with a 🛠 and can be tracked in the project management system:

* 🛠 **Define Metric Enums & Models:** In the domain model layer, add `HealthMetricType` enum with all new types (HR, HRV, RR, SpO2, BP, Temp, VO2\_MAX, ECG, WORKOUT, etc.). Create Pydantic models for incoming HealthKit samples (`HKSamplePayload`) if not already defined, and domain models for processed data (e.g. `ProcessedMetrics` object containing numpy arrays or lists for each metric). Include validators for data ranges. Write unit tests for model validation (e.g. HR must be positive).

* 🛠 **Extend Ingestion API Endpoint:** Modify or create FastAPI endpoint `POST /v1/health-data` to accept batch uploads (JSON with list of samples). Parse JSON to Pydantic `HealthDataUpload` model. Implement request handling logic: authenticate user, call service layer to save data. Ensure idempotency by checking `uploadId` (if same uploadId seen before for that user, skip processing or update status). Return a response with status and maybe `uploadId`. Test with example payloads (including a gzip-compressed payload scenario if applicable).

* 🛠 **Implement Save-and-Publish Use Case:** In `clarity/services/health_data_service.py`, implement `save_health_data(upload: HealthDataUpload, user_id)`. This function writes raw data to GCS (perhaps using an async GCS client) – ensure to format filename with uid and timestamp (for traceability). Then use the Pub/Sub client to publish a message to `health-data-upload` topic. The message JSON should include necessary info (e.g. `{"uid":..., "uploadId":..., "gcs_path":..., "metrics": ["HR","Steps",...]}`). Use `asyncio` if possible to parallelize GCS upload and Pub/Sub publish. Handle exceptions (if either fails, return an error status to client). Write unit tests with GCS and PubSub clients mocked (assert that publish was called with correct topic and data).

* 🛠 **Set Up GCP Pub/Sub Infrastructure:** (Infrastructure-as-code or manual for dev environment) Create topics `health-data-upload` and `insight-request`. For local dev, you can skip actual creation and use emulator, but document the real setup (Terraform script as above). Ensure the backend code has config (environment variables) for topic names and uses them.

* 🛠 **Cloud Run Analysis Service Scaffolding:** Create a new FastAPI app (or even a simple Flask) for the analysis service. Add an endpoint `/process-task` that expects Pub/Sub push. Implement verification of JWT: use Google’s token verification library or simply check the header `Authorization: Bearer <JWT>` using the project’s public key (Google provides a certs URL) – or use the Pub/Sub feature that the token's audience is the endpoint URL and issuer is Google. For initial development, you might disable strict verification and just trust that if the request hit this endpoint, it's from Pub/Sub (assuming it's internal). Once verified, parse the incoming Pub/Sub message JSON. The Pub/Sub payload will be in the request body under a field `message.data` (base64 encoded) etc., per Pub/Sub push format; implement logic to decode it. Then, acknowledge by returning a 200 response quickly (we might actually do the work synchronously and then return, or do work async – but since Cloud Run will retry on non-200, we'll do work within the request and only return 200 when done or at least queued elsewhere). Write a quick unit test for this endpoint handler: simulate a request with a valid message and ensure it calls the processing pipeline function.

* 🛠 **Develop Preprocessing Utilities:** In `clarity/ml/preprocessing.py`, implement functions like `preprocess_hr(times, values)`, `preprocess_hrv(...)`, etc., according to the logic described (outlier removal, resampling...). Use numpy and pandas as needed (pandas for resampling time series by minute might be handy). Make sure these functions can handle edge cases (empty input returns empty output gracefully). Unit test each with various synthetic inputs. Also include `steps_to_movement_proxy()` as given (sqrt and z-score) and test it with a small vector.

* 🛠 **Implement Processor Classes:** In `clarity/ml/processors.py`, implement `ActigraphyProcessor` (that loads PAT model and returns its embedding given steps vector), `CardioProcessor`, `RespirationProcessor`, etc. Each should:

  * Accept preprocessed series (perhaps as numpy arrays or lists).
  * Possibly further transform or compute features.
  * For ActigraphyProcessor: integrate with PAT. This could call an internal function `predict_actigraphy(vector)` which uses TensorFlow/PyTorch to load weights and compute embedding. This might require setting up the model architecture in code unless using an existing saved model format; possibly, PAT being provided as `.h5` weights means a Keras model architecture code is needed – perhaps available via the PAT GitHub. For now, assume we have a function to get embedding from `.h5`. Implement caching of the model (load once, reuse for subsequent invocations) to avoid reloading on each request (Cloud Run instance will handle multiple messages).
  * For CardioProcessor: implement either a small model inference or compute a feature vector (like \[resting\_hr, max\_hr, avg\_hr, hrv\_mean, hrv\_sd] etc.). If using a model like PatchTST, you'll need to have it loaded; initial version could just do features to simplify.
  * Ensure these processors are **stateless** (no reliance on global state, except cached model weights).
  * Write unit tests with sample inputs. For ActigraphyProcessor, monkeypatch the actual model prediction with a fake function that returns a known vector, then test that the output is passed through correctly. For others, test on small arrays (like HR \[60, 70, 80]) and verify features (max=80, etc.).

* 🛠 **Fusion Head Implementation:** In `clarity/ml/fusion.py`, implement the Fusion logic. Perhaps define a class `FusionHead` that on init creates some torch/keras model if using actual transformer layers. However, training that without labeled data is tricky – so one approach: implement a deterministic fusion for now (like concatenation or weighted sum) but structure the code to allow using a transformer later. For demonstration, we might implement a simple concatenation + linear: e.g. `fusion_vector = Dense(64, activation='relu')(concatenate([actigraphy_emb, cardio_feat, resp_feat, other_feat]))`. If using PyTorch, just do the tensor ops. Alternatively, incorporate `nn.TransformerEncoder` from PyTorch: treat each feature vector as a token, stack them and run through one encoder layer. If we go that route, we need to define dummy positional encodings or none (we have so few tokens, position may not matter except to distinguish modalities).

  * For now, implement as simple as possible but keep the interface such that we could plug in a learned model later. Document that improvement step.
  * Test fusion by feeding dummy inputs (e.g. ones or distinct vectors) and ensuring output shape is correct and values make sense (if not trained, it might just be some weighted average depending on weights initialization).
  * If using an ML framework, ensure to set it up to run in Cloud Run (maybe use PyTorch CPU, which is fine given small sizes).
  * Unit test: if concatenation, easy to test expected output for known inputs. If using random transformer, test that it runs and output length is as expected.

* 🛠 **End-to-End Analysis Pipeline Glue:** In the analysis service code (maybe `analysis_worker.py`), implement the main logic that the Pub/Sub handler calls. This should:

  1. Download and parse raw JSON from GCS into structured data.
  2. Call each relevant processor. Likely, the incoming data includes multiple types, so we instantiate the needed processors. For example: if "stepCount" present, do ActigraphyProcessor; if "heartRate" present, do CardioProcessor; etc. Some processors might require multiple inputs (CardioProcessor needs both HR and HRV).
  3. Collect all output vectors from processors into a dict.
  4. Pass them into the FusionHead to get unified vector.
  5. Formulate the payload for the next stage: could be just this vector (maybe encoded as list of floats) plus some key metadata (user, uploadId, perhaps some summary numbers if we want to avoid LLM doing all parsing from vector).
  6. Publish the `insight-request` Pub/Sub message with this payload.
  7. Optionally, also directly store some intermediate results to Firestore (like we might store the PAT embedding now itself in `embeddings/` collection as backup).
  8. Handle exceptions at each step: wrap in try/except and on failure, log error and optionally write a Firestore status (like mark analysis as "FAILED"). Pub/Sub will retry if we don't ack, but we should consider acking and perhaps sending a separate "failed" notification to user. Simpler: if an error occurs, let it throw and not ack; Pub/Sub will retry a few times and then dead-letter if configured. We can configure a Dead Letter Topic for errors and monitor that.

  * Write an integration test for this logic with a sample input covering multiple modalities (as much as possible with fake data) to ensure it flows without error. We will likely need to stub out actual Pub/Sub in tests; instead call the pipeline function and inspect the published message (which we can capture by monkeypatching the Pub/Sub publish method to just record its input).

* 🛠 **Insight Generator Implementation:** In the insight generator service code, implement the subscriber endpoint similar to analysis. It receives the fused vector or features. Now:

  1. Convert the numeric vector into human-readable stats. Possibly, to make the prompt more straightforward, do a reverse-mapping of features to text. E.g., if our unified vector isn't directly interpretable, maybe we also send some stats (like we calculated HR avg in CardioProcessor – we could attach that in the insight request message). Ideally, the insight request contains already computed key stats (so the LLM prompt can be filled easily). So this step might involve minimal computation, mostly formatting.
  2. Construct the LLM prompt (system + user) using a template file or f-string as designed. Ensure to fill in all fields carefully and not include any raw data that’s not allowed.
  3. Call Vertex AI API. Use google-cloud-aiplatform SDK or REST call. Since this is an asynchronous pipeline, we can afford to do a full LLM call here. Wrap it with proper timeout (if LLM doesn’t respond in e.g. 30 seconds, consider it failed). Use the parameters from config: model name, temperature, etc.
  4. Receive the response. If it's in the desired JSON format, parse it to verify JSON. If it's not valid JSON (LLMs sometimes deviate), we can either attempt a fix (maybe the response is close to JSON but with minor format issues – we could use a regex or a JSON5 parser). In worst case, if parsing fails, we could fallback to storing the raw text as the summary in Firestore.
  5. Store the final insight: create a Firestore document under `insights/{uid}/{uploadId}` containing the structured insight (as fields or as a nested JSON). Possibly also store a timestamp and any metadata (like which model was used, etc.).
  6. Acknowledge the Pub/Sub message (return 200). If any error occurred before this, handle as with analysis (could retry or dead-letter; e.g. if Vertex is down, Pub/Sub will retry which might succeed later).

  * Write tests for the prompt creation (as discussed in unit tests) and a test that a fake Vertex call is handled. We can monkeypatch the Vertex SDK method to return a predetermined string. Then test that our code writes the correct Firestore document (in test, use Firestore emulator or a stub repository object where we just record the data).
  * Also test some edge cases: no unified vector provided (e.g. if analysis mistakenly sent empty data, ensure we handle gracefully), or Vertex returns a warning or an empty result.

* 🛠 **Configure Terraform and Deploy Infra:** Write Terraform configs or Cloud CLI commands for all needed resources (as listed in IaC section). This includes creating service accounts, topics, subscriptions, and assigning IAM roles. Use Terraform to apply to dev/staging first. Verify that Pub/Sub pushes are reaching the endpoints (in staging one can deploy the services and then manually test by publishing a message). Adjust any permissions if errors (e.g. if push auth fails, ensure token audience matches exactly the service URL, etc.). Once working in staging, apply to prod. Document these resource names and any IDs in the repo’s README for reference.

* 🛠 **Run End-to-End Tests & Tuning:** Now that everything is wired, perform an end-to-end test with a variety of data. Collect logs from Cloud Run to ensure each step logs something sensible (we should have logging inside analysis: e.g. "Computed actigraphy embedding for user X", inside insight: "LLM output stored for user X"). If any part of the pipeline fails, debug and fix. For instance, if the PAT model is slow or hitting memory limits, we might decide to default to the smaller model (PAT-S) for faster responses. Or if the LLM prompt is too long and hits token limits, we may truncate or rephrase it.

  * Check that the insight content looks good and factual. Possibly refine the prompt if the LLM output is not satisfactory (this iterative prompt tuning should happen now).
  * Ensure the Firestore data is structured in a way the frontend expects. If needed, adjust Firestore writing to match the app’s reading logic (maybe the app expects a certain document schema for insights – coordinate with frontend devs).
  * Verify that performance is acceptable: e.g., try a realistic heavy input (lots of data points) and measure total latency from upload to insight. If it’s say 10 seconds, that might be okay for a background job; if it's 60+ seconds, we might need to optimize (e.g. move to PAT-M or PAT-S, reduce LLM tokens or parallelize some parts). Cloud Run logs and timing can help identify bottlenecks (which likely will be the LLM call \~ a few seconds).

* 🛠 **Implement CI/CD Pipeline:** Set up GitHub Actions or Cloud Build triggers for: running tests on PR, and building + deploying the containers on merge. Use infrastructure as code to auto-deploy to staging, perhaps manual promotion to prod. Include steps in CI to run our test suite (with coverage) and perhaps to scan for security issues (like Bandit for Python security linting). Ensure the `models/` directory with PAT weights is available to the build (maybe stored in Cloud Build artifacts or pulled from a secure storage if large – 7MB is fine to include in repo or build context though). Also ensure no secrets are in code – use GCP Secret Manager for any API keys (though Vertex AI uses application default creds, so likely none needed except the Firebase service account JSON for auth verification, which should be a secret).

* 🛠 **Monitoring and Alerts Setup:** As a final step, configure Cloud Monitoring dashboards and alert policies. For example, track Cloud Run error rates for analysis and insight services. Set up alerts for any spikes in errors or latency beyond a threshold. Also set up log-based alerts for any security-sensitive events (e.g. multiple failed auth). These tasks ensure the team gets notified of issues promptly in production.

* 🛠 **Documentation and Handoff:** Write or update documentation for the team: API docs (if any changes to request/response, update `docs/api/health-data.md` and `docs/api/insights.md` accordingly), architecture docs (could incorporate the Mermaid diagram and this explanation). Document how to run tests and how to deploy. Also, prepare a brief **security compliance report** summarizing how the solution meets requirements (this helps during any audit or review).

By following these 🛠 tasks in order, developers will incrementally build and verify the system. Early tasks set the foundation (models, enums, ingestion), middle tasks build processing and ML, and later tasks integrate everything and ensure quality and compliance. This roadmap avoids option ambiguity by committing to specific implementations at each step (async Pub/Sub pipeline, transformer fusion, Vertex AI for insights) – it’s a single gold-standard path to the desired functionality with no forks. The end result will be a maintainable, tested, and scalable system that brings Apple HealthKit data to life with advanced analytics and AI-driven insights, all within the robust architecture of Clarity Loop.

**Sources:** Recent research and best practices were applied throughout, including the Pretrained Actigraphy Transformer for activity analysis, multimodal transformer fusion strategies from current literature, and Google Cloud architecture recommendations for secure asynchronous pipelines. The design aligns with contemporary trends such as personalized AI health coaching and is built on a strong foundation of Clean Architecture and Google Cloud’s HIPAA-compliant services. All components are engineered for high testability and reliability, ensuring a production-ready integration of HealthKit data into the Clarity Loop backend.
