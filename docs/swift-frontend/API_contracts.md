# CLARITY Pulse iOS Networking and Data Modeling Blueprint

## Swift API Data Models (DTOs)

To mirror the backend’s OpenAPI (FastAPI/Pydantic) models, we define Swift structs (DTOs) conforming to `Codable` for each request and response. All fields, types, and optionality match the API contracts exactly, using `UUID` for identifiers and `Date` for timestamps (with ISO8601 encoding).

### **Authentication DTOs** (Under `/api/v1/auth`)

* **UserRegistrationRequestDTO:** Represents new user registration data, with fields for `email`, `password`, `firstName`, `lastName`, optional `phoneNumber`, and booleans for terms/privacy acceptance.
* **UserLoginRequestDTO:** Contains `email`, `password`, a `rememberMe` flag, and optional `deviceInfo` metadata.
* **RefreshTokenRequestDTO:** Contains a single `refreshToken` string for token refresh.
* **Auth Tokens & Session:** **TokenResponseDTO** holds an `accessToken`, `refreshToken`, token type (`"bearer"`), expiration (`expiresIn` seconds), etc. **UserSessionResponseDTO** provides user profile info: `userId` (UUID), name, email, role, permissions list, status, timestamps for creation/lastLogin, and flags for `mfaEnabled` and `emailVerified`.
* **LoginResponseDTO:** Combines a `user: UserSessionResponseDTO` and `tokens: TokenResponseDTO` upon successful login (includes an optional `mfaSessionToken` if MFA is required).
* **RegistrationResponseDTO:** On registration, returns the new `userId` (UUID), email, account `status`, whether a verification email was sent, and account `createdAt` timestamp.
* **Miscellaneous:** Logout returns a simple message JSON (we model it as `LogoutResponseDTO` with `message: String`). The **verifyEmail** endpoint takes a verification code (passed as a query or body string) and returns a success message on completion.

**Swift Example:**

```swift
struct UserRegistrationRequestDTO: Codable {
    let email: String
    let password: String
    let firstName: String
    let lastName: String
    let phoneNumber: String?
    let termsAccepted: Bool
    let privacyPolicyAccepted: Bool
}
struct RegistrationResponseDTO: Codable {
    let userId: UUID
    let email: String
    let status: String  // e.g. "pending_verification" or "active"
    let verificationEmailSent: Bool
    let createdAt: Date
}
```

*All date fields use ISO8601 (UTC) format; e.g. `createdAt` is decoded as `Date` with an `.iso8601` strategy.*

### **HealthKit Upload DTOs** (Under `/api/v1/healthkit_upload`)

* **HealthKitSampleDTO:** Represents a single HealthKit sample. It includes an `identifier` (HealthKit sample UUID as string), a `type` (e.g. `"HKQuantityType"` vs `"HKCategoryType"`), a `value` which can be numeric or a complex object, an optional `unit` (e.g. `"bpm"`, `"count"`), `startDate` and `endDate` timestamps (as ISO8601 strings), optional `sourceName` (device source), optional `device` info dictionary, and a `metadata` dictionary. This struct covers both quantity and category samples by using a flexible `value` (in Swift, we may model `value` as `Double?` for numerics and a generic `Dictionary<String, Any>?` for complex cases).
* **HealthKitUploadRequestDTO:** Contains the `userId` (the authenticated user’s ID, string) and arrays of samples: `quantitySamples` and `categorySamples` (lists of `HealthKitSampleDTO`), plus raw `workouts` and `correlationSamples` if present (arrays of dictionaries), an `uploadMetadata` dictionary, and an optional `syncToken` for incremental sync. All lists default to empty if not provided.
* **HealthKitUploadResponseDTO:** Returned immediately after a successful upload request (HTTP 202). It includes a generated `uploadId` (format `<userId>-<UUID>`), a status (e.g. `"queued"`), a `queuedAt` timestamp, a `samplesReceived` count per category (keys: `"quantity_samples"`, `"category_samples"`, etc.), and a user-facing `message`.

**Swift Example:**

```swift
struct HealthKitSampleDTO: Codable {
    let identifier: String
    let type: String
    let value: CodableValue  // enum or struct to handle numeric vs dict
    let unit: String?
    let startDate: Date
    let endDate: Date
    let sourceName: String?
    let device: [String: AnyCodable]?  // use AnyCodable to store device dict
    let metadata: [String: AnyCodable]
}
struct HealthKitUploadRequestDTO: Codable {
    let userId: String
    let quantitySamples: [HealthKitSampleDTO]
    let categorySamples: [HealthKitSampleDTO]
    let workouts: [AnyCodable]   // list of workout dictionaries
    let correlationSamples: [AnyCodable]
    let uploadMetadata: [String: AnyCodable]
    let syncToken: String?
}
struct HealthKitUploadResponseDTO: Codable {
    let uploadId: String
    let status: String           // e.g. "queued"
    let queuedAt: Date
    let samplesReceived: [String: Int]  // counts per sample type
    let message: String
}
```

After uploading, a **Status** endpoint (`GET /status/{upload_id}`) returns an *upload status* object (we can model as `HealthKitUploadStatusDTO` with fields like `uploadId`, `status` (e.g. `"processing"`), `progress` (0–1), `message` (status detail), and `lastUpdated` timestamp).

### **Health Data DTOs** (Under `/api/v1/health-data`)

The Health Data API uses a more complex data model to upload and retrieve various health metrics.

* **HealthDataUploadDTO:** The request for uploading health metrics. It includes a `userId: UUID` (only the user’s own UUID is allowed), an array of `metrics`, an `uploadSource` string (e.g. `"apple_health"`), a client-side timestamp `clientTimestamp` (Date), and an optional `syncToken`. The metrics list is required (1–100 items).
* **HealthMetricDTO:** A single health metric record, capturing heterogeneous data. Fields include a unique `metricId: UUID`, a `metricType: String` (type of metric, e.g. `"heart_rate"`, `"sleep_analysis"`, etc.), and *one* of several data payloads depending on the type:

  * `biometricData` (type-specific vital signs),
  * `sleepData` (sleep analysis details),
  * `activityData` (activity/exercise stats),
  * `mentalHealthData` (mood/stress indicators).
    Only the field corresponding to `metricType` will be non-nil (the backend enforces that the correct sub-object is present for the given type). Each sub-object is defined below. In addition, each metric can have an optional `deviceId` (source device identifier), raw sensor data as `rawData` (free-form JSON), optional `metadata` (JSON), and a `createdAt` timestamp (when the record was created).
* **BiometricDataDTO:** Sub-model for vital signs metrics (used when `metricType` is e.g. heart rate, blood pressure, etc). Fields include `heartRate` (BPM), `bloodPressureSystolic`, `bloodPressureDiastolic`, `oxygenSaturation` (percent), `heartRateVariability` (ms), `respiratoryRate` (breaths/min), `bodyTemperature` (°C), and `bloodGlucose` (mg/dL), all as optional numerics.
* **SleepDataDTO:** Sub-model for sleep metrics (`metricType = "sleep_analysis"`). Fields include `totalSleepMinutes` (integer minutes slept), `sleepEfficiency` (0–1 fraction), optional `timeToSleepMinutes`, `wakeCount` (number of awakenings), an optional `sleepStages` breakdown (dictionary of stage -> minutes), and timestamps `sleepStart` and `sleepEnd` (Date).
* **ActivityDataDTO:** Sub-model for activity metrics (e.g. `"activity_level"`). Contains fields such as `steps` (count), `distance` (km), `activeEnergy` (kcal), `exerciseMinutes`, `flightsClimbed`, `vo2Max`, `activeMinutes`, and `restingHeartRate` – all optional and appropriate types.
* **MentalHealthIndicatorDTO:** Sub-model for mental health metrics (e.g. mood or stress). Fields include `moodScore` (enumeration of mood, e.g. very\_low/low/neutral/good/excellent), and various 1–10 scalar ratings: `stressLevel`, `anxietyLevel`, `energyLevel`, `focusRating`. It also tracks daily activities like `socialInteractionMinutes` and `meditationMinutes`, an optional free-text `notes` (capped length), and a `timestamp` of the assessment.
* **HealthDataResponseDTO:** Response after uploading health data. Contains a `processingId: UUID` that identifies the processing job, a `status` (enum string such as `"received"` or `"processing"`), counts of `acceptedMetrics` vs. `rejectedMetrics`, any validation errors (`validationErrors`, see below), an optional `estimatedProcessingTime` (seconds), a new `syncToken` if provided, a user-friendly `message` (“successfully queued for processing”, etc.), and a server `timestamp`.

  * **ValidationErrorDTO:** (if any metrics were rejected) Each has a `field`, `message`, and error `code` for why a metric was invalid (e.g. `"too_many_items"` if too many metrics were sent).

**Swift Example:**

```swift
enum HealthMetricType: String, Codable {
    case heart_rate, heart_rate_variability, blood_pressure, blood_oxygen
    case respiratory_rate, sleep_analysis, activity_level
    case stress_indicators, mood_assessment, cognitive_metrics, environmental
    // ... (cover all types from backend):contentReference[oaicite:43]{index=43}:contentReference[oaicite:44]{index=44}
}
struct BiometricDataDTO: Codable {
    var heartRate: Double?
    var bloodPressureSystolic: Int?
    var bloodPressureDiastolic: Int?
    var oxygenSaturation: Double?
    var heartRateVariability: Double?
    var respiratoryRate: Double?
    var bodyTemperature: Double?
    var bloodGlucose: Double?
}
struct SleepDataDTO: Codable {
    var totalSleepMinutes: Int
    var sleepEfficiency: Double
    var timeToSleepMinutes: Int?
    var wakeCount: Int?
    var sleepStages: [String: Int]?     // e.g. {"deep": 120, "rem":90, ...}
    var sleepStart: Date
    var sleepEnd: Date
}
struct ActivityDataDTO: Codable { 
    var steps: Int?
    var distance: Double?
    var activeEnergy: Double?
    var exerciseMinutes: Int?
    var flightsClimbed: Int?
    var vo2Max: Double?
    var activeMinutes: Int?
    var restingHeartRate: Double?
}
struct MentalHealthIndicatorDTO: Codable {
    var moodScore: String?              // e.g. "neutral" (use String or define MoodScale enum)
    var stressLevel: Double?
    var anxietyLevel: Double?
    var energyLevel: Double?
    var focusRating: Double?
    var socialInteractionMinutes: Int?
    var meditationMinutes: Int?
    var notes: String?
    var timestamp: Date
}
struct HealthMetricDTO: Codable {
    let metricId: UUID
    let metricType: HealthMetricType
    let biometricData: BiometricDataDTO?
    let sleepData: SleepDataDTO?
    let activityData: ActivityDataDTO?
    let mentalHealthData: MentalHealthIndicatorDTO?
    let deviceId: String?
    let rawData: [String: AnyCodable]?
    let metadata: [String: AnyCodable]?
    let createdAt: Date
}
struct HealthDataUploadDTO: Codable {
    let userId: UUID
    let metrics: [HealthMetricDTO]
    let uploadSource: String
    let clientTimestamp: Date
    let syncToken: String?
}
struct ValidationErrorDTO: Codable {
    let field: String
    let message: String
    let code: String
}
struct HealthDataResponseDTO: Codable {
    let processingId: UUID
    let status: String        // "processing", "completed", etc.
    let acceptedMetrics: Int
    let rejectedMetrics: Int
    let validationErrors: [ValidationErrorDTO]
    let estimatedProcessingTime: Int?
    let syncToken: String?
    let message: String
    let timestamp: Date
}
```

On data retrieval, `GET /health-data/` returns a **paginated list** of metrics. The response can be modeled as `PaginatedMetricsResponseDTO` containing an array of metrics (each as a dictionary or as a simplified metric view), and pagination info (page size, cursors, links). For simplicity, we may decode the `"data"` array into `[HealthMetricDTO]` or a lighter `MetricSummaryDTO` if we only need a subset of fields for listing. (The backend currently returns metrics under a `"data": [...]` key with HAL-style pagination links.)

There are also endpoints to query processing status: `GET /health-data/processing/{id}` which returns a status object or error (we can use a generic `ProcessingStatusResponseDTO` with fields like `status`, `progress`, etc., similar to HealthKit status). A legacy `GET /health-data/query` returns metrics without pagination (deprecated; not implemented in the app).

### **PAT Analysis DTOs** (Under `/api/v1/pat`)

The PAT (Pretrained Actigraphy Transformer) service endpoints handle analysis of actigraphy (activity) data.

* **StepDataRequestDTO:** Request to analyze Apple Health step counts via proxy actigraphy. Contains `stepCounts: [Int]` (e.g. 10,080 minute-by-minute steps for a week) and corresponding `timestamps: [Date]` arrays of equal length, plus an optional `userMetadata` dictionary for demographics.
* **DirectActigraphyRequestDTO:** Request to directly analyze preprocessed actigraphy data. Contains `dataPoints: [ActigraphyDataPointDTO]` (each data point with a timestamp and value), a `samplingRate: Double` (e.g. 1.0 per minute), and `durationHours: Int` (total hours of data). `ActigraphyDataPointDTO` can be a struct with `timestamp: Date` and `value: Double`.
* **AnalysisResponseDTO:** Generic response for analysis submission (used by both endpoints). Fields include a new `analysisId: String` (the UUID of the analysis job), the current `status` (`"processing"` or `"completed"`, etc.), an optional `analysis` payload (present when status is completed), `processingTimeMs` if available, an optional `message` (error or status note), and a `cached: Bool` flag indicating if the result was served from cache. The `analysis` field, when populated, contains the detailed results from the PAT model (an `ActigraphyAnalysisDTO` – structure defined by the ML model). This might include computed features, risk scores, etc. (In Swift, we can model it if the schema is known, or treat it as `ActigraphyAnalysisDTO?` which mirrors the backend’s `ActigraphyAnalysis` object.)
* **PATAnalysisResponseDTO:** Used by `GET /pat/analysis/{id}` to retrieve results or status of a given analysis. It includes the `processingId` (same as analysisId), the analysis `status` (could be `"completed"`, `"processing"`, `"failed"`, or `"not_found"`), an optional `message` (e.g. error or success note), optional `analysisDate` (when completed), and the actual results if available. Results are given as `patFeatures: [String: Double]?` (e.g. key feature values), an `activityEmbedding: [Double]?` (the vector embedding from the model), and a `metadata` dictionary with any additional details. If the analysis is still processing or not found, these result fields will be nil or empty. We model `PATAnalysisResponseDTO` accordingly.
* **HealthCheckResponseDTO:** The `GET /pat/health` returns a health status of the PAT service. The response model includes a `service` name, `status` string (e.g. healthy/unhealthy), a timestamp, a version, and embedded dictionaries for `inferenceEngine` and `patModel` status/details. We model these as nested types (or use `[String: Any]` for the detailed fields since they may vary).

**Swift Example:**

```swift
struct StepDataRequestDTO: Codable {
    let stepCounts: [Int]
    let timestamps: [Date]
    let userMetadata: [String: AnyCodable]?
}
struct ActigraphyDataPointDTO: Codable {
    let timestamp: Date
    let value: Double
}
struct DirectActigraphyRequestDTO: Codable {
    let dataPoints: [ActigraphyDataPointDTO]
    let samplingRate: Double
    let durationHours: Int
}
struct AnalysisResponseDTO<AnalysisPayload: Codable>: Codable {
    let analysisId: String
    let status: String                  // "processing", "completed", etc.
    let analysis: AnalysisPayload?      // e.g. ActigraphyAnalysisDTO when completed
    let processingTimeMs: Double?
    let message: String?
    let cached: Bool
}
struct PATAnalysisResponseDTO: Codable {
    let processingId: String
    let status: String                  // "completed", "processing", "failed", "not_found"
    let message: String?
    let analysisDate: Date?
    let patFeatures: [String: Double]?
    let activityEmbedding: [Double]?
    let metadata: [String: AnyCodable]? 
}
struct PATServiceHealthDTO: Codable {
    let service: String   // "PAT Analysis API"
    let status: String    // "healthy" or "unhealthy"
    let timestamp: Date
    let version: String
    let inferenceEngine: [String: AnyCodable]
    let patModel: [String: AnyCodable]
}
```

*Example:* To call `analyzeStepData`, the app creates a `StepDataRequestDTO` with a week of step counts and timestamps. The response is decoded into `AnalysisResponseDTO<ActigraphyAnalysisDTO>` – initially with `status="processing"` and no `analysis`. The app can poll `GET /analysis/{id}` to get a `PATAnalysisResponseDTO` when `status` becomes `"completed"` (containing `patFeatures`, etc.).

### **Gemini Insights DTOs** (Under `/api/v1/insights`)

These endpoints leverage an LLM (Gemini) to generate human-readable health insights from analysis results.

* **InsightGenerationRequestDTO:** Request to generate insights. Fields: `analysisResults` (a JSON object with analysis data, e.g. combined results from PAT or health metrics), an optional `context` string (additional user context or questions), an `insightType` (e.g. `"comprehensive"` vs `"brief"`), a boolean `includeRecommendations` (whether to include actionable advice) and `language` (ISO code, e.g. `"en"`).
* **HealthInsightDTO:** Represents a generated insight narrative. From the backend’s `HealthInsightResponse`, it includes the `userId` (of the requesting user), a textual `narrative` (the main insight summary), an array of `keyInsights` (salient bullet points), an array of `recommendations` (suggested actions), a `confidenceScore` (0.0–1.0 indicating model confidence), and a `generatedAt` timestamp.
* **InsightGenerationResponseDTO:** The response from the generate call. It has a `success: Bool` (indicating the request succeeded), a `data: HealthInsightDTO` (the insight content), and a `metadata` dictionary (includes a `requestId`, timestamp, and service info).
* **InsightHistoryResponseDTO:** Returned by `GET /insights/history/{user_id}`. It contains `success: Bool`, a `data` object with the user’s insight history, and `metadata`. The `data` likely includes an array of insight summary records and pagination. In the current implementation, each insight summary includes fields like `id` (insight ID), a truncated `narrative` preview, `generatedAt` timestamp, `confidenceScore`, and counts of `keyInsights` and `recommendations`. It also provides `totalCount` and pagination info (hasMore, limit/offset, etc.). We model `InsightHistoryResponseDTO` with a nested `InsightHistoryDataDTO` that has these fields (or simply use `[String: Any]` for flexibility).
* **ServiceStatusResponseDTO:** From `GET /insights/status`, indicating the Gemini service health. It has `success: Bool`, and `data` with fields like `service: "gemini-insights"`, `status: "healthy"|"unhealthy"`, a `model` info sub-dictionary (model name, projectId, initialized flag, capabilities list), and a timestamp. We model `ServiceStatusDataDTO` for the data and wrap it in `ServiceStatusResponseDTO`.

**Swift Example:**

```swift
struct InsightGenerationRequestDTO: Codable {
    let analysisResults: [String: AnyCodable]
    let context: String?
    let insightType: String            // e.g. "comprehensive"
    let includeRecommendations: Bool
    let language: String               // e.g. "en"
}
struct HealthInsightDTO: Codable {
    let userId: String
    let narrative: String
    let keyInsights: [String]
    let recommendations: [String]
    let confidenceScore: Double
    let generatedAt: Date
}
struct InsightGenerationResponseDTO: Codable {
    let success: Bool
    let data: HealthInsightDTO
    let metadata: [String: AnyCodable]
}
struct InsightPreviewDTO: Codable {
    let id: String
    let narrative: String       // possibly truncated
    let generatedAt: Date
    let confidenceScore: Double
    let keyInsightsCount: Int
    let recommendationsCount: Int
}
struct InsightHistoryDataDTO: Codable {
    let insights: [InsightPreviewDTO]
    let totalCount: Int
    let hasMore: Bool
    let pagination: PaginationMetaDTO  // with page, limit, etc.
}
struct InsightHistoryResponseDTO: Codable {
    let success: Bool
    let data: InsightHistoryDataDTO
    let metadata: [String: AnyCodable]
}
struct ServiceStatusDataDTO: Codable {
    let service: String
    let status: String
    let model: ModelInfoDTO
    let timestamp: Date
}
struct ModelInfoDTO: Codable {
    let modelName: String
    let projectId: String
    let initialized: Bool
    let capabilities: [String]
}
struct ServiceStatusResponseDTO: Codable {
    let success: Bool
    let data: ServiceStatusDataDTO
    let metadata: [String: AnyCodable]
}
```

In generating insights, the app will take combined results (e.g. from PAT analysis) and send an `InsightGenerationRequestDTO`. The resulting `HealthInsightDTO` contains a human-readable summary and recommendations which the app can display.

**Note:** For any fields that are dynamic or of type `dict[str, Any]` on the backend (e.g. `analysisResults`, `metadata`, etc.), we use strategies like `AnyCodable` or custom `Codable` implementations to preserve the data. We also ensure all date strings (which are ISO8601 UTC) are decoded with the proper `JSONDecoder.dateDecodingStrategy`.

## SwiftData Persistence Schema (@Model)

For local caching and offline support, we design SwiftData `@Model` classes that closely correspond to the DTOs but may diverge for app-specific needs. Each persistent model will have its own schema, separate from network DTOs, often adding local-only metadata like sync timestamps.

Key design points: we maintain a one-to-one mapping of core fields to avoid confusion, but we can simplify or flatten complex structures for easier querying. We also include derived fields and relationships as needed for the app’s UI (while keeping the DTOs as pure data carriers).

### **Auth Persistence Model: User**

Although Firebase Auth manages identity, we may cache basic user profile info locally. For example, a `User` @Model with properties for `id` (UUID), `email`, `firstName`, `lastName`, `role`, etc., matching `UserSessionResponseDTO`. We also store a `lastSyncedAt: Date` to note when we last refreshed the profile. This ensures the `/auth/me` data can be accessed offline. (If the app always pulls fresh user info on launch, this model can be optional.)

```swift
@Model class User {
   @Attribute(.unique) var id: UUID
   var email: String
   var firstName: String
   var lastName: String
   var role: String
   var permissions: [String]
   var status: String
   var emailVerified: Bool
   var mfaEnabled: Bool
   var createdAt: Date
   var lastLogin: Date?
   var lastSyncedAt: Date    // local-only metadata
}
```

### **Health Data Persistence Models**

**HealthMetricEntity:** We create a unified model to store health metrics locally, since metrics are frequently queried by date or type. This `@Model` might include:

* An `id: UUID` (the metricId),
* `type: String` (metricType),
* Common fields for timestamp and value(s). For example, we might store a primary timestamp for the metric – for many biometric metrics, the `createdAt` can serve as the measurement time. For Sleep metrics, we could store `sleepStart`/`sleepEnd` as well, or choose one representative timestamp for sorting (like sleepEnd). We include fields for key values that we want to query or display (heartRate, steps, etc.), and we can store the rest of the sub-structure as a blob or separate related entity.

One approach is **flattening** common fields into HealthMetricEntity: e.g. `heartRate`, `systolicBP`, `diastolicBP`, `steps`, `totalSleepMinutes`, etc., all as optional properties. Only the ones relevant to the metric’s type will be set. We also keep a `rawData` JSON (String or Data) for any detailed fields not modeled explicitly. Additionally, `lastSyncedAt` tracks when this record was fetched from the server. This design sacrifices some normalization for simplicity but makes querying metrics by value or date straightforward.

Alternatively, we can split into multiple entities (e.g. SleepRecord, ActivityRecord, etc.) and use a protocol or inheritance, but SwiftData’s querying can handle many optionals in one table, so a single table is acceptable given the moderate number of fields.

Example schema (flattened single table):

```swift
@Model class HealthMetricEntity {
   @Attribute(.unique) var id: UUID              // metricId
   var type: String                              // e.g. "heart_rate", "sleep_analysis"
   var date: Date                                // primary timestamp (e.g. createdAt or end of interval)
   // Common biometric fields:
   var heartRate: Double?
   var systolicBP: Int?
   var diastolicBP: Int?
   var oxygenSaturation: Double?
   var hrv: Double?
   var respiratoryRate: Double?
   var bodyTemperature: Double?
   var bloodGlucose: Double?
   // Sleep fields:
   var totalSleepMinutes: Int?
   var sleepEfficiency: Double?
   var sleepStart: Date?
   var sleepEnd: Date?
   // Activity fields:
   var steps: Int?
   var distance: Double?
   var activeEnergy: Double?
   var exerciseMinutes: Int?
   var flightsClimbed: Int?
   var vo2Max: Double?
   var activeMinutes: Int?
   var restingHeartRate: Double?
   // Mental health fields:
   var moodScore: String?
   var stressLevel: Double?
   var anxietyLevel: Double?
   var energyLevel: Double?
   var focusRating: Double?
   var socialInteractionMinutes: Int?
   var meditationMinutes: Int?
   // Metadata:
   var deviceId: String?
   var rawJson: Data?                            // store rawData/metadata as JSON blob
   var lastSyncedAt: Date
}
```

*Rationale:* This design stores each metric in one table row. We use a representative `date` for sorting (for most metrics this can be the creation or measurement timestamp; for interval data like sleep, we might use `sleepEnd` or `sleepStart` depending on UI needs). The numerous optional fields accommodate different metric types. `rawJson` holds any additional info (like full stage distributions or notes) that we didn’t explicitly model but might display in detail views. We mark `id` unique to prevent duplicates.

**Mapping:** The app will map `HealthMetricDTO` -> `HealthMetricEntity` by copying fields: e.g. set `entity.id = dto.metricId`, `entity.type = dto.metricType.rawValue`, then for each non-nil sub-data, assign its values (e.g. if `dto.biometricData` exists, fill heartRate, BP, etc.). The entire `dto` can also be JSON-encoded and stored in `rawJson` for completeness if needed. We populate `lastSyncedAt = Date()` at the time of save.

**Querying:** With this schema, we can fetch metrics by type or date range easily (e.g. all where `type == "heart_rate"` and `date` in last 7 days). We can also compute aggregates if needed by filtering and using Swift to reduce.

### **Insight Persistence Model**

**InsightEntity:** We store generated insights so users can review them offline. An `InsightEntity` may contain:

* `id: String` (the insight ID, which could be the document ID from Firestore or a requestId – the backend uses a unique ID for each insight),
* the full `narrative` text,
* optional lists of `keyInsights` and `recommendations` (which we could store as `[String]` if SwiftData supports array of strings, or as a single large text blob combining them, or as a related `InsightPoint` entity each – for simplicity, we store as blob or joined text for now),
* `confidenceScore: Double`,
* `generatedAt: Date`.
  We also include a truncated preview if we want quick list display (or we can generate that on the fly by truncating narrative to \~200 chars, matching the backend’s preview length limit). Additionally, store a reference to the `userId` (owner) to support multi-user or filtering (though likely single-user app).

```swift
@Model class InsightEntity {
   @Attribute(.unique) var id: String
   var userId: String
   var narrative: String
   var keyInsights: [String]
   var recommendations: [String]
   var confidenceScore: Double
   var generatedAt: Date
   var lastViewedAt: Date?       // track if user viewed it
}
```

When the app calls `/insights/generate`, it receives an `InsightGenerationResponseDTO`. We convert that to an `InsightEntity` and store it. For history, calling `/insights/history` returns a list of past insights; we upsert those into the store. The unique constraint on `id` prevents duplicates. If the history API provides only previews, we might need to call `/insights/{id}` to fetch full detail for offline use; our model is designed to hold full detail (the history preview can be discarded or used to decide which ones to fetch fully).

### **PAT Analysis Persistence**

For PAT analysis results, local caching might be optional (since analysis can be run on demand). But if we want to cache the results of an analysis, we could store a **PATAnalysisEntity** with fields: `id` (analysisId), `status`, `completedAt`, and possibly the `patFeatures`, `activityEmbedding`, etc., similar to `PATAnalysisResponseDTO`. This could be useful for showing the last analysis results quickly without re-fetching. Given that insights (from Gemini) are likely the user-facing output, we might skip storing raw PAT results locally, or only store if the app has a heavy analysis viewer. For completeness:

```swift
@Model class PATAnalysisEntity {
    @Attribute(.unique) var id: String
    var status: String
    var completedAt: Date?
    var patFeatures: Data?           // could store as JSON Data
    var activityEmbedding: Data?    // large vector as Data blob
    // etc., or specific fields if the app needs them individually
}
```

This entity can be populated when an analysis completes. The `patFeatures` (a dictionary of feature names to values) and embedding array could be JSON-encoded and stored, or we define a separate related entity for each feature if we need to query them. Since these are mostly for display, storing as Data is fine.

### **Mapping & Sync Strategy**

Each DTO has a corresponding `toModel()` function to convert to SwiftData model, and each model could have a `toDTO()` if needed (for example, to send cached data back to the UI or to prepare offline edits, though in this case most data flows from network to store). We keep these mappings straightforward: field-by-field assignment with type conversion (e.g. string to UUID, ISO string to Date).

We include `lastSyncedAt` on any entity that syncs with the server. After a successful API call (like fetching new metrics or insights), we update this timestamp. This helps implement sync logic (e.g. only fetch new items after lastSyncedAt).

## Networking Layer Design

We will implement a dedicated networking service (API client) using modern Swift concurrency (async/await) and `URLSession`. The design emphasizes clarity, testability, and separation of concerns:

### **APIClient Structure**

We define an `APIClient` class (or a set of classes grouped by feature) responsible for constructing requests, performing HTTP calls, and decoding responses. This client will handle **authentication**, **JSON encoding/decoding**, and **error handling** centrally:

* Use a base URL for the backend (e.g. `https://api.clarity.health/api/v1`). We can store this in one place so if the version or domain changes, it’s easy to update.
* Inject an auth token provider – for example, a closure or delegate that returns the latest Firebase JWT. The APIClient will attach the `Authorization: Bearer <token>` header to authenticated requests. All endpoints except registration/login and health checks require this JWT, so the client will automatically include it when calling those methods.
* Utilize a single `URLSession` (or `URLSession.shared`) for requests. Because we target iOS 17+, `URLSession` supports async `data(for:)` which we will use for clarity.

We can organize methods by endpoint category for readability, e.g.:

```swift
class APIClient {
    let baseURL = URL(string: "https://api.clarity.health/api/v1")!
    var tokenProvider: () async -> String?   // e.g. use FirebaseAuth to get current ID token
    
    // MARK: Auth Endpoints
    func register(_ req: UserRegistrationRequestDTO) async throws -> RegistrationResponseDTO { … }
    func login(_ req: UserLoginRequestDTO) async throws -> LoginResponseDTO { … }
    func refreshToken(_ req: RefreshTokenRequestDTO) async throws -> TokenResponseDTO { … }
    func logout(_ req: RefreshTokenRequestDTO) async throws 
         -> MessageResponseDTO { … }  // returns {"message": "..."}
    func getCurrentUser() async throws -> UserSessionResponseDTO { … }
    func verifyEmail(code: String) async throws -> MessageResponseDTO { … }

    // MARK: HealthKit Upload
    func uploadHealthKit(_ req: HealthKitUploadRequestDTO) async throws -> HealthKitUploadResponseDTO { … }
    func getHealthKitUploadStatus(uploadId: String) async throws -> HealthKitUploadStatusDTO { … }

    // MARK: Health Data
    func uploadHealthData(_ req: HealthDataUploadDTO) async throws -> HealthDataResponseDTO { … }
    func getHealthData(page: Int = 1, limit: Int = 100) async throws -> PaginatedMetricsResponseDTO { … }
    func getProcessingStatus(id: UUID) async throws -> ProcessingStatusDTO { … }

    // MARK: PAT Analysis
    func analyzeStepData(_ req: StepDataRequestDTO) async throws -> AnalysisResponseDTO<EmptyPayload> { … }
    func analyzeActigraphy(_ req: DirectActigraphyRequestDTO) async throws -> AnalysisResponseDTO<EmptyPayload> { … }
    func getPATAnalysis(id: String) async throws -> PATAnalysisResponseDTO { … }
    func getPATServiceHealth() async throws -> PATServiceHealthDTO { … }

    // MARK: Insights
    func generateInsights(_ req: InsightGenerationRequestDTO) async throws -> InsightGenerationResponseDTO { … }
    func getInsight(id: String) async throws -> InsightGenerationResponseDTO { … }
    func getInsightHistory(userId: String, limit: Int, offset: Int) async throws -> InsightHistoryResponseDTO { … }
    func getInsightsServiceStatus() async throws -> ServiceStatusResponseDTO { … }
}
```

*(For simplicity, above we use separate methods; in practice, you might generalize some patterns, but listing individually clarifies which endpoints exist.)*

Each method constructs the `URLRequest` for the corresponding endpoint: sets the HTTP method, appends path components, encodes the request body (for POST/PUT) or query params (for GET with filters), and sets headers:

* **JSON Encoding/Decoding:** We use `JSONEncoder` and `JSONDecoder` configured globally (e.g. with `.iso8601` date decoding strategy to handle ISO timestamps). The DTOs are `Codable`, so encoding/decoding is automatic. We ensure to include fractional seconds if the backend uses them (FastAPI’s default ISO format includes `Z` and possibly microseconds – using `.iso8601` should suffice for our needs, but if microsecond precision is required, we could use a custom date formatter). All `UUID` will encode/decode as 36-char strings by default via `Codable`.

* **Auth Injection:** Before sending, for protected endpoints, `APIClient` will call `tokenProvider()` to get a fresh Firebase JWT (ensuring we have a non-expired token). It then sets `request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")`. (The backend expects an `Authorization` header with bearer token.)

* **Error Handling:** We implement a unified flow to handle HTTP errors. If the response status code is not in 200-299, we attempt to decode the error body. The backend often returns structured error payloads:

  * Auth endpoints may return `detail` with `error` and `error_description`, or even an `AuthErrorResponse` with `error_details`.
  * Other endpoints use RFC 7807 Problem Details or custom error JSON (e.g. the insights service returns an `error` object with code, message, etc. wrapped in an HTTP 4xx/5xx).
    We can create a Swift `APIError: Error` enum:

    * `.networkError(underlying: URLError)` for transport failures (no connection, timeout, etc.),
    * `.decodingError(underlying: Error)` for JSON parse issues,
    * `.serverError(statusCode: Int, message: String, details: Data?)` for HTTP errors. We can further parse `details` into our error models if needed (for example, try decoding into `AuthErrorResponseDTO` or a generic `ErrorResponseDTO` that matches the shape of `{"error": { code, message, … }}`). If decoded, we can store the error code and message in the error object for easier handling (e.g. to show a user-friendly alert like “Invalid credentials” from error code).

  The client methods will `throw` appropriate `APIError`. For instance, a 401 from login returns `invalid_credentials` – we decode and throw `.serverError(statusCode: 401, message: "Invalid email or password", details: ...)`.

* **Concurrency & Cancellation:** Each API call is an `async throws` function. Internally, we use `try await URLSession.shared.data(for: request)` which respects cancellation. This makes our APIClient methods cancelable if the view model or caller cancels the task (e.g. user navigates away).

* **Testing:** By injecting the `tokenProvider` and possibly using protocols for URLSession (or using `URLProtocol` stubs), we can unit test API calls by mocking responses. Each method is small and focused, making it easy to verify that requests are correctly formed and responses properly decoded.

**Sample Usage & Flow:**
When the user logs in, for example, the view model creates a `UserLoginRequestDTO` from the input fields and calls `try await apiClient.login(req)`. The `login()` method encodes the DTO to JSON, performs a POST to `/auth/login`, and decodes a `LoginResponseDTO` on success. It then returns that to the caller. The view model can then take the tokens from `LoginResponseDTO.tokens`, store them (or pass them to Firebase if needed), and map the `user` object to our local `User` model (saving via SwiftData). Similar flows happen for other endpoints: e.g. calling `uploadHealthData` returns a `HealthDataResponseDTO` which contains a processingId; the app might then immediately call `getProcessingStatus(processingId)` to poll, or rely on push notifications if configured.

**Example Implementation Snippet:**

```swift
func uploadHealthData(_ req: HealthDataUploadDTO) async throws -> HealthDataResponseDTO {
    var request = URLRequest(url: baseURL.appendingPathComponent("health-data/upload"))
    request.httpMethod = "POST"
    request.httpBody = try JSONEncoder().encode(req)
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    // Auth header:
    if let token = await tokenProvider() {
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }
    // Perform request
    let (data, response) = try await URLSession.shared.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse else {
        throw APIError.networkError(URLError(.badServerResponse))
    }
    if httpResponse.statusCode == 201 {
        // Decode success response
        return try JSONDecoder().decode(HealthDataResponseDTO.self, from: data)
    } else {
        // Decode and throw error
        if let serverError = try? JSONDecoder().decode(ProblemDetailsDTO.self, from: data) {
            throw APIError.serverError(status: httpResponse.statusCode,
                                       message: serverError.detail,
                                       details: data)
        }
        throw APIError.serverError(status: httpResponse.statusCode,
                                   message: "Upload failed with status \(httpResponse.statusCode)",
                                   details: data)
    }
}
```

In this snippet, we assume the server might return RFC7807 Problem Details (mapped to `ProblemDetailsDTO` with at least a `detail` field) for errors like validation issues. We attempt to decode that; if it fails, we throw a generic error with status code.

Centralizing logic like this in each method (or using a helper to handle the response checking) ensures consistent error interpretation. We also log errors (the APIClient can use `os_log` or print in debug builds for troubleshooting).

### **Central Error Handling & Retries**

At a higher level (e.g. in a network manager or the ViewModel layer), we will catch `APIError`. We can inspect if it’s an auth-related error – for example, a 401 on an authenticated call might indicate an expired token. Since we are using Firebase JWTs, the recommended approach is to rely on Firebase SDK’s token refresh prior to making calls (the tokenProvider can ensure a fresh token by calling `getIDToken(result: .refresh)` if needed). However, if a token expires mid-call and we get a 401 `{"error": "invalid_token"}`, we could attempt one automatic refresh: call our `refreshToken()` endpoint or Firebase refresh, update the token, and retry the original request once.

This logic can be encapsulated in the APIClient (e.g. a `requestWithAuth` wrapper that tries the closure, and on 401 once, refreshes and retries). Since our backend even provides a `/auth/refresh` that takes a refreshToken, we could store the refreshToken from login, and if we encounter 401, call `refreshToken()` to get a new access token (and update header, then retry original call). This ensures seamless handling of expired tokens. After logout, we clear stored tokens.

All other errors propagate to the UI, where user-friendly messaging can be shown. We leverage the `error_description` or `message` from server when available – for example, on a 400 validation error from registration, the server returns detailed messages for each field; we can surface those to the user (e.g. “Password must contain at least one special character”).

## Client-Side Data Validation

While the backend performs thorough validation (via Pydantic and custom checks), the iOS app will implement basic client-side validation to improve UX and ensure data integrity before sending:

* **Format Validation:** Verify input formats like email using `NSPredicate` or `URLValidator`, ensure required fields are not empty, etc., in the UI layer. For example, before creating a `UserRegistrationRequestDTO`, check that email text contains "@" and password meets the complexity rules. We can reuse the backend’s rules (at least partially) – e.g. enforce a minimum 8 characters password with one uppercase, one lowercase, one digit, one special char. This gives immediate feedback instead of waiting for a server error.
* **Date & Numeric Ranges:** Ensure dates are reasonable (no future dates for past events, etc.) if applicable. The backend rejects future dates for HealthKit samples, so the app should avoid sending them (the HealthKit integration on device likely already filters out future timestamps). Similarly, if the user manually inputs any health data, validate ranges (e.g. heart rate between 30 and 220 BPM as per physiological limits, matching backend logic).
* **Consistency Checks:** For example, in StepDataRequestDTO, ensure `timestamps.count == stepCounts.count` (the DTO initializer or a custom validate method can assert this, mimicking the backend’s validator). For HealthDataUpload, ensure the metrics array is not empty and does not exceed 100 items (we can split larger batches if needed, since the backend will error on >100 metrics).
* **Local UUIDs:** When creating new local objects that require UUIDs (e.g. metric IDs), use UUID(). The backend will also generate IDs server-side, but if any request needs a stable client UUID (not the case for our endpoints except maybe correlation IDs), ensure correct formatting (32-hex string for healthkit uploadId segments, etc.). In general, the app will rely on server-generated IDs for resources.
* **Prevent Cross-User Access:** The app should not allow or send requests with another user’s ID. This is mostly ensured by design (we always use the current user's ID from auth context). For example, when calling upload or history, use the logged-in user's UID/UUID. The backend will 403 if userId mismatch occurs, but we prevent it by construction.

Overall, client-side validation is about enhancing user experience (e.g. highlighting a missing field or invalid format immediately) and reducing needless API calls that we know will fail. The **SOLID** principle of single responsibility is applied by keeping validation logic either in the model (through initializers) or in form view models, while the APIClient focuses only on transport.

## API Versioning and Future-Proofing

To accommodate future versions (e.g. a future `/api/v2`), we isolate v1-specific code and design the module to allow parallel v2 implementation when needed:

* **Namespace or Module:** We organize DTO definitions and API calls in a namespace (for instance, put all v1 models under an enum or namespace `API.V1`). Swift doesn’t have built-in namespaces, but we can achieve this with grouping in files and using name prefixes if necessary. For example, classes/structs could be prefixed with `V1` or put inside a struct `APIv1 { struct HealthDataUploadDTO {…} }` to differentiate from future v2 structs. This way, if v2 has changes (say different field names or data shapes), we can create a parallel set of `V2` DTOs without risking breaking v1. The networking layer can have separate endpoints for v2, or a versioned base URL.

* **Versioned Base URL:** The base path `api/v1` is hard-coded in our endpoints. We should avoid scattering `"api/v1"` strings throughout the code. Instead, define a constant or configure the baseURL so that upgrading to v2 means changing that constant. For instance, `APIClient.baseURL` could be constructed using an `apiVersion` property. Then a new API client instance (or a parameter on methods) could target v2. In practice, we might create a `APIClientV2` subclass or new class when the time comes, to implement any new endpoints or changed behavior while coexisting with v1 during a transition period.

* **Separate DTOs per Version:** Do not reuse v1 DTO types for v2 if the API contract differs. Even if some fields remain the same, create distinct v2 DTOs or use generics to handle minor differences. This prevents subtle bugs where code intended for v1 is used with v2 data. For example, if `/v2/health-data/upload` returns an extra field or changes a type, we would introduce a new `HealthDataResponseV2DTO`. By using separate types or modules, we adhere to the Open/Closed principle – we extend with v2 rather than modify v1, minimizing regression risk.

* **Conditional Logic:** If the app must handle multiple versions at once (unlikely for a single deployment, but possibly during a migration), our APIClient can negotiate version based on the user or configuration. For instance, if an endpoint is upgraded to v2, we could version our method names (e.g. `uploadHealthDataV2`) or have the method detect which version to call. However, a cleaner approach is to **abstract the API** behind protocols. We could define a protocol `HealthDataAPI` with a method `uploadHealthData(request:) -> HealthDataResponse`. Then provide V1 and V2 implementations. The app at runtime can choose the appropriate implementation (perhaps based on a feature flag or API version check). This abstracts version differences and keeps calling code the same. In our blueprint, this is an advanced consideration; initially, we implement v1 fully and keep the code modular for easy extension.

* **Documentation and Deprecation:** Clearly mark v1 models and functions as deprecated once v2 is introduced, and guide developers to the new versions. Since our blueprint is internal, this mainly means we’d stop maintaining v1 in the app once the backend switches fully to v2, but during interim, we keep them separate.

In summary, **Version 1** of the networking and models is encapsulated so we can introduce **Version 2** side-by-side. This ensures a smooth transition – e.g., we might run both v1 and v2 in the app if the backend requires it (some endpoints v1, some v2). By isolating concerns, our networking layer adheres to SOLID principles: each endpoint method is single-purpose, the APIClient is closed for modification (we add new one for v2 rather than changing internal logic everywhere), and by using dependency injection (for tokens, session, etc.), we keep it testable and flexible.

---

**Example Code: Mapping and Usage**

Finally, tying everything together, here is a brief example illustrating a typical flow (login -> fetch data -> store locally):

```swift
// 1. User Login
let loginReq = UserLoginRequestDTO(email: email, password: pwd, rememberMe: true, deviceInfo: nil)
let loginRes = try await apiClient.login(loginReq)  // LoginResponseDTO
// Save tokens and user locally
tokenStore.save(loginRes.tokens.accessToken, refresh: loginRes.tokens.refreshToken)
let userSession = loginRes.user
try modelContext.insert(User(  // SwiftData context insert
    id: userSession.userId,
    email: userSession.email,
    firstName: userSession.firstName,
    lastName: userSession.lastName,
    role: userSession.role,
    permissions: userSession.permissions,
    status: userSession.status.rawValue,
    emailVerified: userSession.emailVerified,
    mfaEnabled: userSession.mfaEnabled,
    createdAt: userSession.createdAt,
    lastLogin: userSession.lastLogin,
    lastSyncedAt: Date()
))

// 2. Upload Health Data (e.g. after collecting metrics from HealthKit)
let metricsDTO: [HealthMetricDTO] = collectMetricsFromHealthKit()  // function that transforms HK samples to DTO
let uploadReq = HealthDataUploadDTO(userId: currentUserId, metrics: metricsDTO, 
                                    uploadSource: "apple_health", clientTimestamp: Date(), syncToken: nil)
let uploadRes = try await apiClient.uploadHealthData(uploadReq)  // HealthDataResponseDTO
print("Server accepted \(uploadRes.acceptedMetrics) metrics, processing id = \(uploadRes.processingId)")
// Poll status or wait for notification...
let status = try await apiClient.getProcessingStatus(id: uploadRes.processingId)
// If completed, perhaps fetch processed data (not in v1 scope; assume async processing pipeline stores results elsewhere)

// 3. Retrieve and cache health metric list
let page1 = try await apiClient.getHealthData(limit: 50)
for metricDict in page1.data {   // here page1.data might be [[String: Any]] or already [HealthMetricDTO]
    let metricDTO = try MetricSummaryDTO(from: metricDict)  // decode dictionary to DTO if needed
    let metricEntity = HealthMetricEntity(
         id: metricDTO.metricId, type: metricDTO.metricType.rawValue, date: metricDTO.primaryDate, lastSyncedAt: Date()
    )
    // assign fields e.g. metricEntity.heartRate = metricDTO.biometricData?.heartRate etc.
    try modelContext.insert(metricEntity)
}
// Save context, etc.

// 4. Generate Insights from a PAT analysis result
let patResult = ... // assume we have a combined analysis result dictionary from PAT + other data
let insightReq = InsightGenerationRequestDTO(analysisResults: AnyCodable.from(patResult),
                                            context: "Provide a summary of my weekly health.",
                                            insightType: "comprehensive", includeRecommendations: true, language: "en")
let insightRes = try await apiClient.generateInsights(insightReq)  // InsightGenerationResponseDTO
let insight = insightRes.data  // HealthInsightDTO
// Cache the insight locally
try modelContext.insert(InsightEntity(id: UUID().uuidString,  // or use some id from response if provided
    userId: currentUserId, narrative: insight.narrative,
    keyInsights: insight.keyInsights, recommendations: insight.recommendations,
    confidenceScore: insight.confidenceScore, generatedAt: insight.generatedAt, lastViewedAt: nil))
```

In the above flow:

* We demonstrate transforming and mapping between layers (HealthKit to DTO, DTO to SwiftData model).
* The networking calls (`apiClient.login`, `.uploadHealthData`, `.getHealthData`, `.generateInsights`) each encapsulate the HTTP details. The higher-level code simply handles the returned DTOs.
* After each network response, we map to persistence: the example shows inserting a User and multiple HealthMetricEntity and InsightEntity objects. We add `lastSyncedAt = Date()` to records to mark the sync time.
* Error handling around these calls (not shown) would catch `APIError` and update the UI (e.g. show an alert if `.serverError(message: "Invalid email or password")` is thrown from login).

This blueprint ensures the iOS app’s networking and data modeling layer is **comprehensive, type-safe, and maintainable**. All data classes mirror the server schema exactly (1:1 fields and types) to minimize transformation logic, while the persistence layer is designed for efficient local queries and is kept decoupled from raw DTOs. The networking layer uses modern Swift features (async/await, Codable) and clean error handling to provide a robust service layer adhering to SOLID principles (e.g., single responsibility for each endpoint method, injected token for open/closed extension, etc.). With a clear separation between DTOs, persistence models, and API client, the app can easily evolve (for instance, adapting to a future v2 API) with minimal risk and maximal clarity.

**Sources:**

* Clarity Backend Pydantic Models (Auth, HealthKit, Health Data, PAT, Insights)
* Clarity API Endpoint Definitions, demonstrating request/response structures used to derive the above DTOs.

