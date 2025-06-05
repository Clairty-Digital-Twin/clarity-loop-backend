Below is the “Apple HealthKit Proxy Actigraphy Dossier” your backend agent can paste into docs/health-data.md (or its own docs/apple-health-proxy.md).
It spells out, line-by-line, how to turn HealthKit-exposed signals into the minute-resolution movement vector that PAT expects, along with every API call, transformation, and backend contract the agent must implement.

⸻

1 | Why a “proxy” is required

Apple HealthKit does not expose the raw 50-100 Hz wrist accelerometer stream. The only always-on movement metric you can read is per-minute step count (HKQuantityTypeIdentifierStepCount)  ￼.
Therefore, for PAT we must create a proxy actigraphy vector from step-count data (or from a custom Watch workout session that streams CMAccelerometerData—outside MVP scope).

⸻

2 | On-device collection (iOS / watchOS)

Step API / Code Sketch Notes
2.1 Authorize healthStore.requestAuthorization(toShare: [], read: [.quantityType(forIdentifier: .stepCount)!]) Include HR/HRV/etc. if you plan to ingest more signals.
2.2 Subscribe to updates HKObserverQuery → HKStatisticsCollectionQuery (minute granularity, .cumulativeSum option, interval.day = 0, interval.minute = 1) Returns time-ordered buckets of HKQuantitySamples.
2.3 Gather 1 week of data Keep a ring buffer [10 080] of Double step counts. Align Monday-Sunday (ISO 8601); pad leading zeros if first week shorter.
2.4 Serialize payload JSON\n{\n "uid":"USER_ID",\n "uploadId":"W20250602T0900Z",\n "vector":[/*10 080 doubles*/],\n "unit":"count/min"\n}\n Upload via HTTPS to /v1/ingest/steps.

⸻

3 | Backend ETL  ➜  “proxy actigraphy” vector

def steps_to_movement_proxy(steps_per_min: np.ndarray) -> np.ndarray:
    # 1. Convert to "activity counts" proxy
    accel_proxy = np.sqrt(steps_per_min)        # empirically correlates with RMS acceleration
    # 2. Z-score using NHANES μ/σ shipped with PAT
    mu, sigma = lookup_norm_stats(year=2025)    # or global defaults
    z = (accel_proxy - mu) / sigma
    return z.astype("float32")                  # length 10 080

Rationale: √steps/min preserves dynamic range without letting large step bursts dominate.

⸻

4 | PAT micro-service contract (unchanged)

Endpoint Request Response
POST /v1/PAT/predict { "vector":[float32×10 080], "uid":"...", "uploadId":"..." } { "cls_embedding":[128 floats], "token_embeddings_gs_path":"gs://..." }

(Full schema in earlier doc.)

⸻

5 | Storing & querying “proxy” data
 • Raw HealthKit JSON → GCS cold-storage (PHI-protected bucket, CMEK enabled).
 • Proxy z-scored vector → Firestore proxy_vectors/{uid}/{uploadId} doc.
 • PAT CLS embedding → Firestore embeddings/{uid}/{uploadId} doc.

⸻

6 | Extending to heart-rate / HRV soon

 1. Collect:
 • HKQuantityTypeIdentifierHeartRate (BPM, 5-sec granularity).
 • HKQuantityTypeIdentifierHeartRateVariabilitySDNN (ms).
 2. Align: resample to 1-min bins (mean for HR, median for HRV).
 3. Fuse: store alongside movement vector; later feed to PatchTST or a simple MLP and concatenate with PAT embedding before sending to Gemini.

⸻

7 | Edge-case handling

Scenario Action
Week vector has <10 080 samples Pad left with zeros.
Gaps >30 min (watch off) Impute zeros (PAT treats as “no movement”).
Step count identical zeros all week Flag quality: "insufficient" to downstream; skip PAT inference.

⸻

8 | References for the agent

Topic Canonical doc
Step-count quantity type developer.apple.com/documentation/HealthKit/hkquantitytypeidentifier/stepcount  ￼
Minute-level Stats query WWDC ‘19 “Exploring New Data Representations in HealthKit” (see HKStatisticsCollectionQuery pattern)  ￼
PAT repo (I/O details) github.com/njacobsonlab/Pretrained-Actigraphy-Transformer  ￼
