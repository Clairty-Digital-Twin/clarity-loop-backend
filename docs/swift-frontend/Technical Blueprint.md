# CLARITY Pulse iOS Application Technical Blueprint

## Data Synchronization Strategy

### Local SwiftData Cache and Backend Consistency

CLARITY Pulse uses **SwiftData** (Apple’s new persistence framework in iOS 17) as the sole local cache for health metrics and AI insights. All relevant backend data – such as health summaries and generated insights – are mirrored into SwiftData models on the device to enable fast access and offline viewing. The **backend (clarity-loop-backend)** exposes RESTful endpoints for health data and insights, which the app consumes to populate and update the local cache. For example, the backend’s health data listing API returns an array of health metric records (each with an `id`, `type`, `value`, timestamp, source, etc.). The iOS app maintains a corresponding `HealthMetric` SwiftData model (e.g. with properties for type, value, unit, timestamp, source, qualityScore, etc.) and ensures that each record from the server is stored or updated in SwiftData. Each SwiftData entity uses a stable unique identifier (e.g. the server-provided UUID or ID string) as a primary key (via `@Attribute(.unique)`), so incoming data can be **upserted** (inserted if new or updated if existing) to keep the local cache in sync with the latest backend state.

### Polling-Based Synchronization (MVP Approach)

For the MVP, data synchronization is primarily **polling-based**. The app periodically fetches updates from the backend (for example, whenever the app launches, enters the foreground, or the user performs a manual refresh) to reconcile any differences between local and remote data. This is achieved via standard `URLSession` network calls. A dedicated synchronization component (e.g. a **SyncManager** or view model) will issue GET requests to relevant endpoints, parse the responses into Swift model objects, and update SwiftData accordingly. For instance, to sync health metrics, the app calls:

```swift
// Example: Fetch latest health data list
let url = URL(string: "\(baseURL)/api/v1/health-data?limit=50")!
var request = URLRequest(url: url)
request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
let (data, response) = try await URLSession.shared.data(for: request)
guard (response as? HTTPURLResponse)?.statusCode == 200 else {
    throw AppError.network // handle HTTP errors
}
let decoder = JSONDecoder()
decoder.dateDecodingStrategy = .iso8601
let serverResponse = try decoder.decode(HealthDataListResponse.self, from: data)
// `HealthDataListResponse` might mirror the JSON structure from backend:
struct HealthDataListResponse: Decodable {
    let data: [HealthDataDTO]
    let pagination: Pagination?
    /* ... */
}
```

The DTO (`HealthDataDTO`) is a plain Swift struct conforming to `Decodable` that matches the JSON schema (for example, `id`, `type`, `value`, `unit`, `timestamp`, etc., as returned by the API). After decoding, the sync logic iterates through the `data` array and inserts or updates the corresponding SwiftData entities in a single transaction. This ensures the on-device cache reflects the server’s truth. Similar flows exist for syncing **AI insights**: the app might call `GET /api/v1/insights/` to retrieve a list of past insights (each with an `insight_id`, type, status, generated date, summary, etc.), then map those into a local `Insight` SwiftData model for display in the UI.

**Periodic Refresh:** In the foreground, the app can use SwiftUI’s `.task(priority:)` modifier on views (or a Combine timer if needed) to trigger periodic refreshes. For example, a dashboard view could call a refresh task every few minutes when active, or use `.refreshable` (pull-to-refresh) for user-initiated sync. Because tasks in SwiftUI are tied to view lifecycles, the `.task(priority:)` can ensure network calls run at an appropriate priority (e.g. `.userInitiated` for immediate refresh when user pulls, or `.background` for silent periodic sync). All networking is done with `async/await`, keeping the main thread free and the UI responsive while data loads.

### Structured DTOs and SwiftData Mapping

To manage complexity, the app defines **structured data transfer objects (DTOs)** for each endpoint’s responses, making heavy use of Swift’s `Codable`. Each DTO mirrors the backend’s schema. For example:

```swift
struct HealthMetricDTO: Codable {
    let id: String
    let type: String
    let value: Double?
    let unit: String?
    let timestamp: Date
    let source: String
    let quality_score: Double?
}
```

This corresponds to fields returned by the **List Health Data** API. Similarly, an `InsightDTO` might mirror the **Get Generated Insight** response structure – including nested fields like summary text, key findings, recommendations, and health score. The app’s persistence layer uses SwiftData models that closely align with these DTOs:

```swift
import SwiftData

@Model class HealthMetric {
    @Attribute(.unique) var id: String
    var type: String
    var value: Double?
    var unit: String?
    var timestamp: Date
    var source: String
    var qualityScore: Double?
}
```

On receiving new DTOs from the server, the app performs a mapping: if a `HealthMetric` with the same `id` already exists in the local store, it’s updated (to avoid duplicate entries); otherwise a new object is inserted. This mapping logic can be encapsulated in a sync function:

```swift
func syncHealthMetrics(from dtoList: [HealthMetricDTO], using context: ModelContext) throws {
    for dto in dtoList {
        let metric = try context.fetchOne(FetchDescriptor<HealthMetric>(predicate: #Predicate { $0.id == dto.id }))
        if let existing = metric {
            // Update existing record
            existing.type = dto.type
            existing.value = dto.value
            /* ...update other fields... */
        } else {
            // Insert new record
            context.insert(HealthMetric(id: dto.id, type: dto.type, value: dto.value,
                                        unit: dto.unit, timestamp: dto.timestamp,
                                        source: dto.source, qualityScore: dto.quality_score))
        }
    }
    try context.save() // persist changes atomically
}
```

Using SwiftData’s high-level API ensures changes are saved in an **eventually consistent** manner on disk. This way, the local database becomes a faithful cache of server data. All read operations (e.g. populating SwiftUI views) can then be done on the SwiftData store, making the UI highly responsive and minimizing direct network calls during view rendering. The synchronization is one-way (server -> client) for most data in MVP, treating the backend as the source of truth.

### Eventual Consistency and Real-Time Updates

Because the CLARITY backend processes some data asynchronously (e.g. health data uploads followed by AI insight generation), the iOS app must handle **eventual consistency** – data will become consistent over time, not instantly. For example, when a user uploads new health metrics via `POST /api/v1/health-data/upload`, the server immediately returns an acknowledgment with a `processing_id` and status (often “queued” or “processing”). The actual analysis (PAT model, Gemini AI) happens in the cloud after some delay. The app should optimistically record that upload in the local cache (perhaps marking those metrics or their resulting insight as “pending”). A SwiftData entity like `Insight` might have a status field to reflect this (e.g. `.generating`). The app can then **poll** the backend for status updates: e.g. call `GET /api/v1/insights/{insight_id}` periodically until the status changes to "completed", or use a dedicated status endpoint like `GET /api/v1/health-data/processing/{id}` which returns a progress percentage. During this period, the UI indicates a processing state (such as a spinner or “Processing…” label in the insights view). Once the backend marks the process complete, the app fetches the full results (the completed insight content) and updates the SwiftData store, thereby updating the UI to show the final insight.

**Real-Time Sync Plans:** To improve latency and avoid constant polling, future versions will introduce **push-based synchronization**. The backend is designed to provide real-time updates – for instance, results of health data processing are saved to Firestore and intended to trigger real-time client updates. In a future iteration, the app could maintain a WebSocket connection or use Firebase/Firestore listeners to get immediate notifications when new data or insight results are available. Alternatively, **push notifications** (APNs) could inform the app that new insights have been generated (including perhaps a summary or an identifier in the payload), prompting the app to fetch updates in the background. For MVP, these real-time mechanisms are not yet implemented, but the architecture anticipates them. The synchronization code is structured so that it can easily switch from a pull model (polling every X seconds or on app open) to an event-driven model (update as soon as a push event is received). For example, the SyncManager might have a function `refreshInsights()` that is called both on a timer and in response to a notification, abstracting the trigger away from the data handling. This ensures the **transition to WebSockets/push** in the future will be seamless, improving consistency and reducing unnecessary network calls once implemented.

### Handling Network Failures and Retries

Robust sync must consider network unreliability. If a synchronization attempt fails (due to no connectivity, timeouts, or server errors), the app’s strategy is to **fail gracefully and retry** later, without corrupting local state. The SyncManager will catch networking errors (e.g. no internet) and schedule a retry after a short delay or when connectivity changes. For instance, using Apple’s Network framework, we can observe an `NWPathMonitor` for network status: when the device transitions from offline to online, any missed sync can be triggered immediately. Additionally, background refresh mechanisms (discussed below) will periodically attempt sync, so even if a foreground attempt fails, the data will sync eventually.

During a failure, the local SwiftData cache remains as is (showing possibly stale data but at least something). The UI can display an unobtrusive **offline or error indicator** (e.g. a banner saying “Unable to refresh – showing cached data”). Meanwhile, failed **upload** operations (if the user tries to submit data while offline) are queued for retry (see Offline Support below). The app avoids blocking the main thread during retries by using background tasks or `Task.detached` for reattempting network calls, and uses exponential backoff for repeated failures to be network-friendly. This way, data consistency is eventually restored once the network allows, aligning with an **at-least-once delivery** philosophy for syncing.

It’s worth noting that the backend expects unique metric IDs on uploads and will reject duplicates. The client leverages this by generating UUIDs for any new health metrics (ensuring idempotence – retrying an upload won’t create duplicate entries on the server, because the same metric\_id is used). This further aids consistency: if an upload was partially successful before a network drop, a retry with the same IDs will either be ignored or gently handled server-side, preventing double-processing.

### Background Synchronization (HealthKit Integration)

To keep health data updated, the app integrates with **HealthKit** to gather new samples (e.g. daily steps, heart rate, sleep data). Instead of relying only on foreground sync, CLARITY Pulse uses iOS background tasks to fetch and upload HealthKit data in the background. Specifically:

* A **BGAppRefreshTask** can be scheduled (with identifier e.g. `"com.clarity.refresh"`) to periodically wake the app (typically every few hours or as allowed by the system) and trigger a lightweight sync. In the refresh task, the app can call the backend for any updates (e.g. check if new insights are available, or refresh health metrics list). This keeps the cache warm without user intervention. The BGAppRefreshTask should be kept short (execution time is limited to a few seconds), e.g. just pulling small latest data or scheduling a heavier task if needed.

* A **BGProcessingTask** (identifier e.g. `"com.clarity.healthkit.upload"`) is used for longer, network-required operations like bulk HealthKit sample uploads. The app registers this task to run when the system deems appropriate (possibly when the device is charging or at least not in low-power mode). When invoked, the task handler will query HealthKit for any new samples that haven’t been uploaded yet (for example, using anchored queries or HKObserverQuery notifications that the app cached earlier), package them into the server’s expected JSON format (an array of metrics), and call the `POST /api/v1/health-data/upload` endpoint. Using the clarity-loop-backend’s API format, the request body contains a `user_id`, an array of `metrics` (each with type, value, unit, timestamp, etc.), and an `upload_source` (like `"ios_app"` or `"apple_health"`). The backend will respond with a processing status and a new `processing_id`. The background task can then either wait briefly for quick processing or schedule a follow-up (like a BGAppRefresh) to poll the processing status later. We set `requiresNetworkConnectivity = true` on this BGProcessingTask, so it will only run when the network is available (ensuring the upload can succeed). If the operation is large (e.g. syncing hundreds of samples), we might also set `requiresExternalPower = true` so it runs when the device is likely charging.

To implement background tasks, the app uses `BGTaskScheduler`. For example:

```swift
BGTaskScheduler.shared.register(forTaskWithIdentifier: "com.clarity.healthkit.upload", using: nil) { task in
    self.handleHealthKitUpload(task: task as! BGProcessingTask)
}
```

And scheduling it after each run (or after new data is added):

```swift
let request = BGProcessingTaskRequest(identifier: "com.clarity.healthkit.upload")
request.requiresNetworkConnectivity = true
request.requiresExternalPower = false // maybe true if large uploads
// Set earliest begin date if needed: e.g. 1 hour from now
request.earliestBeginDate = Date().addingTimeInterval(15 * 60)
try? BGTaskScheduler.shared.submit(request)
```

Within `handleHealthKitUpload`, we perform the HealthKit fetch and server upload using `async/await`, and call `task.setTaskCompleted(success:)` when done. We also schedule the next background task if needed, creating a continuous sync loop as long as there’s new data.

This strategy ensures **background synchronization** of data, complementing the foreground polling. It keeps local and remote data consistent even if the user doesn’t open the app frequently. It also aligns with Apple’s guidelines by doing heavy work in BGProcessingTasks and leaving quick checks to BGAppRefreshTasks.

## Offline Support

### Cached Data Availability

One of the core offline capabilities in the MVP is the ability to **view previously loaded data offline**. Thanks to SwiftData caching, any health summaries or insights the user has fetched will remain stored on-device. If the app launches without internet, the user will still see their last known metrics, charts, or insight summaries from the SwiftData store. SwiftUI views can bind to SwiftData models (using `@Query` property wrappers or by fetching manually in a view model), so the UI seamlessly loads cached objects. For instance, if the user had an insight with a summary “Your health metrics show consistent improvement...” from the last sync, that Insight entity remains in the database and can be displayed offline. The design ensures that read operations do not depend on network availability. All heavy analysis is done server-side, but the results of that analysis (the AI insights, health metric history) are persisted locally for reference. This provides a baseline **offline mode** for the app: data is **read-only** when offline, but it’s accessible.

### Offline State Detection and UI Indicators

The app will detect the offline/online status using iOS network monitoring. A common approach is using `NWPathMonitor` to observe changes in network connectivity. The app might maintain an `@Observable` (or `ObservableObject`) called `NetworkStatus` with a published property `isOnline`. The SwiftUI views or view models can observe this property to adjust UI accordingly. For example, if `isOnline` switches to false, the app could display a subtle banner or icon indicating “Offline Mode”. Certain UI elements may also become disabled or grayed out: for instance, a “Generate Insight” button could be disabled with a note that internet is required.

**User Feedback:** It’s important to inform users when they are seeing cached data. A banner at the top of the main screen can say “You are offline. Showing saved data.” and perhaps offer a “Retry” button that attempts to reconnect. If the user attempts an action that requires connectivity (like initiating a data sync or uploading new metrics), the app can immediately detect lack of connection and present an alert or toast (“No internet connection. We’ll try again when you’re online.”). This is more user-friendly than just failing silently or spinners that never resolve.

### Read-Only vs. Queued Actions Offline

In offline mode, CLARITY Pulse will **gracefully degrade** to primarily read-only functionality. Users can browse their health dashboard and past insights, since those are cached. Actions that normally involve the network will be queued rather than executed. For example:

* If a user tries to **upload new health data** (say, logging a mood or manually initiating a HealthKit sync) while offline, the app will not discard the data. Instead, it will save this data to SwiftData with a flag indicating “pending upload”. This could be implemented by a separate SwiftData entity or a status field. For instance, a `PendingUpload` entity might store a reference to the HealthMetric (or just the data if not yet saved remotely) plus a timestamp. Or the `HealthMetric` model itself could have a boolean property `needsUpload`. The UI would give immediate feedback (perhaps showing the new entry in a list with an “Uploading…” placeholder status), so the user sees it logged. The actual network upload is deferred.

* Similarly, if the user requests to **generate a new insight** offline, since that requires server-side AI, the app can either disable the action or queue the request. A possible strategy is to allow the user to “schedule” the insight generation: the request parameters (like date range, analysis type) are saved locally as a PendingInsight request. Then, as soon as connectivity is restored, the app will automatically send that request to the backend (and update the UI accordingly with the new insight status). Alternatively, the UI simply informs the user that the action cannot be completed offline and to retry later – this decision can be based on complexity. For MVP, it might be acceptable to **prevent** insight generation offline (with a friendly message), whereas simpler data logging could be queued transparently.

### Retry Queue and Sync on Reconnection

All queued offline actions are retried upon regaining connectivity. The `NetworkStatus` monitor can trigger a routine to process any pending operations. For example:

```swift
networkMonitor.pathUpdateHandler = { path in
    if path.status == .satisfied {
        Task { await self.flushPendingOperations() }
    }
}
```

The `flushPendingOperations()` might iterate through pending uploads in the SwiftData store. For each, it attempts the appropriate API call (e.g. upload health metric or submit insight generation). If the call succeeds, the pending record is marked as completed or removed. If it fails (e.g. server returns an error), the app can decide to either drop it or keep it for another retry, depending on error type. (For instance, a 400 validation error for a Health metric might mean the data was never acceptable – in that case it should not loop forever. But a 500 server error might be worth retrying later.)

This queue mechanism, combined with background tasks, means even if the user remains offline for an extended period, their actions accumulate and sync later without manual intervention. To avoid unlimited growth of the queue, we could impose a limit or expiration (e.g. if something has failed 5 times, flag it for manual review).

**Background upload on reconnect:** iOS does provide a way to attempt background transfers upon connectivity via `URLSession` background configuration. For instance, if using a `URLSessionConfiguration.background` session to upload files or large payloads, the system can automatically resume the transfer when network returns, even if the app is not foreground. In our design, health data uploads could leverage this for efficiency. However, simpler implementation might manage retry in-app with BGTasks as described.

### Background HealthKit Delivery

HealthKit provides a *push-style* mechanism called **background delivery** for certain data types, where the app can register to be notified when new health data is available (even if the app is not running). CLARITY Pulse can utilize this by registering HKObserverQuery for relevant sample types (like heart rate, steps, sleep). When HealthKit triggers an update (for example, new step count data was logged by the Apple Watch), iOS will wake the app in the background for a brief period. The app should then schedule a BGProcessingTask (as described earlier) to handle retrieving that data and uploading it. By chaining HealthKit’s background deliveries to our background upload tasks, we ensure that new health data is captured and sent to the server as soon as possible, even if the user doesn’t actively open the app. This contributes to both data freshness and a more complete dataset on the server.

In summary, offline support in MVP allows **reading cached data** and **queueing write operations**. The user experience is preserved by clear offline indicators and automatic retries. While full offline editing and conflict resolution is not heavily needed in a read-centric MVP, the groundwork is laid for future enhancements. Next, we address how conflicts or concurrent edits would be handled down the road:

* **Conflict Resolution (Future):** In the current model, the server is source of truth and clients mostly add new data. If in the future the app allows editing or deleting data locally (while offline) that also exists on the server, we’ll implement a strategy for conflict resolution. A common approach would be **last-write-wins** using timestamps or version numbers. For example, each SwiftData entity could store a `lastModified` date (from server), and any local edits get a pending flag. Upon sync, if the server version has a newer `lastModified` than the local edit, the local change might be discarded or flagged as a conflict for the user. Alternatively, using the server’s **sync\_token** mechanism (noted in the backend API) can help ensure the client only sends incremental changes that the server hasn’t seen, avoiding duplication. Since MVP avoids this scenario by limiting offline mutable actions, a simple approach (like always prefer server data and queue local changes as new entries) is acceptable for now. We note this here to highlight that the architecture is designed to evolve toward true bidirectional sync if needed.

## Comprehensive Error Handling

### Unified Error Categorization

Building a robust app requires a unified approach to error handling. We design a **central error type** (for example, an `AppError` enum) that covers the various error domains: network, decoding, persistence, HealthKit, etc. This allows us to handle errors in a consistent way in the UI and logging. For instance:

```swift
enum AppError: Error {
    case network(NetworkError)       // e.g. no connection, timeout, HTTP error codes
    case server(code: Int, message: String)  // HTTP 4xx/5xx with backend message
    case decoding(Error)            // JSON parsing issues
    case persistence(Error)         // SwiftData save/fetch errors
    case healthKit(Error)           // HealthKit authorization or query errors
    case unauthorized               // e.g. Firebase JWT expired
    case unknown(Error)             // fallback for uncategorized errors
}
```

The enum can be extended with convenience initializers to map system errors to these cases. For example, when a `URLError` occurs from URLSession, we map it to `.network(...)`. If the HTTP response has an error status, we can create a `.server` error with the status code and perhaps parse a message from the response JSON (the clarity backend often returns a structured error in JSON with an `"error"` field or message). SwiftData or HealthKit methods throw their own errors (which we wrap in `.persistence` or `.healthKit`). This categorization makes it easier to decide how to handle it: e.g. an `.unauthorized` error might trigger a login flow, a `.network` error might just show an offline banner, whereas a `.server` error for a failed upload might show a specific message to the user.

### Error Handling Patterns (try/catch & async)

All asynchronous operations (network calls, database writes, etc.) use Swift’s `async/await` and are wrapped in do/catch blocks to capture thrown errors. For example, a network fetch in a view model:

```swift
@MainActor
func loadInsights() async {
    do {
        let newInsights = try await api.fetchInsights()  // may throw
        try await database.saveInsights(newInsights)     // may throw
    } catch {
        // Convert to AppError if not already
        let appError = (error as? AppError) ?? AppError.unknown(error)
        self.error = appError  // publish the error to the UI
        logError(appError)
    }
}
```

By centralizing the catch logic, we ensure every failure goes through the same funnel. The example above sets a published `error` property on the view model (`self.error`) which the SwiftUI view observes to present an alert. The function is annotated `@MainActor` to ensure that UI state updates happen on the main thread.

### Logging and Monitoring

Internally, we log errors using **OSLog** for debugging and telemetry. We define log categories (subsystems and categories) for different modules. For instance, Network calls log to "Network" category, Database to "Persistence", HealthKit to "HealthKit". When an AppError is caught, we send a log entry:

```swift
import OSLog

let networkLog = Logger(subsystem: "com.clarity.app", category: "Network")
let persistenceLog = Logger(subsystem: "com.clarity.app", category: "Persistence")
// ...
func logError(_ error: AppError) {
    switch error {
    case .network(let netErr):
        networkLog.error("Network error: \(netErr.localizedDescription)")
    case .server(let code, let message):
        networkLog.error("Server error \(code): \(message)")
    case .decoding(let err):
        persistenceLog.error("Decoding error: \(err.localizedDescription)")
    case .persistence(let err):
        persistenceLog.error("Database error: \(err.localizedDescription)")
    case .healthKit(let err):
        Logger(subsystem: "com.clarity.app", category: "HealthKit")
            .error("HealthKit error: \(err.localizedDescription)")
    case .unauthorized:
        networkLog.warning("Auth error: unauthorized – token may have expired")
    case .unknown(let err):
        Logger(subsystem: "com.clarity.app", category: "General")
            .error("Unknown error: \(String(describing: err))")
    }
}
```

These logs use the unified logging system so they can be viewed in Console.app for debugging. In a production setting, we might also integrate a crash reporting or analytics tool to send critical error events (excluding sensitive data) for monitoring. The clarity backend itself is built with structured error responses, which our app can leverage – for example, if the server returns an error payload with specific code (like `"validation_error"` for a bad request), the client can decide if this is user-fixable or a system issue, and log accordingly.

We also persist certain logs if needed. For instance, if an upload repeatedly fails offline, we might log those events to a local file to diagnose later. However, care is taken to not log personal health data content, only metadata about failures, to maintain privacy.

### User-Friendly Error Messages and UI

Error handling doesn’t stop at logging; we present errors to the user in a clear and friendly manner, appropriate to the context. The app maintains a mapping from `AppError` cases to user-facing messages:

* **Network offline**: e.g. “It looks like you’re offline. Please check your internet connection.”
* **Server error**: e.g. for a 500 or unknown server error, “Oops, something went wrong on our end. Please try again later.”
* **Unauthorized**: “Your session has expired. Please log in again.”
* **Validation error** (if, for instance, the server rejects an upload because of bad data): “Some uploaded data was invalid. Please review and try again.” (We could even surface the specific field error if provided by backend.)

These messages are crafted to be non-technical. We use SwiftUI to display them. Common patterns include:

* **Alerts**: If an action the user took fails (like tapping “Generate Insight” and it errors), we show a SwiftUI `.alert`. For example, the view model’s `error` property (AppError?) can be bound to an alert presentation. Using the new observation API in SwiftUI, we could have something like:

  ```swift
  .alert(isPresented: $viewModel.hasError, error: viewModel.error) { error in
      Button("OK", role: .cancel) { }
  } message: { error in
      Text(error.userMessage)
  }
  ```

  Here, `error.userMessage` would be a computed String on AppError that returns the friendly text. SwiftUI in iOS 17 allows initializing an Alert with an `Error` that conforms to `LocalizedError` as well, so we could make AppError conform to `LocalizedError` and provide `errorDescription` accordingly.

* **Inline error views**: For less critical errors (like a non-essential background sync failing), we might not interrupt the user with a popup. Instead, we could show a small **Snackbar** or toast at the bottom of the screen. SwiftUI doesn’t have a built-in snackbar, but we can create one (for example, a view modifier that presents a transient overlay). This could say “Failed to refresh. Showing cached data.” and auto-dismiss after a few seconds.

* **Persistent banners**: In cases of prolonged issues (like no connectivity), a banner below the navigation bar can remain until the situation resolves. This ensures the user is aware of being offline or that certain data might be stale. We ensure these UI elements are updated on the main thread (using `DispatchQueue.main.async` or `@MainActor` for any state changes that drive the UI).

During long-running tasks, instead of freezing the UI, we use **progress indicators**. For instance, while waiting for a new insight to generate (which could take many seconds), the UI might show a progress view or at least a spinner with text “Analyzing your data…”. If the backend provides a progress percentage (the `/processing/{id}` endpoint returns a `progress` field 0–100%), the app can even display a progress bar reflecting that. This keeps the user engaged and less likely to think the app is stuck. Meanwhile, because the network call for status updates is done asynchronously, the main thread remains free to handle user interactions (the user could navigate away and the process would continue in background, with the data updating when ready).

### Ensuring Responsiveness with Structured Concurrency

We leverage Swift’s structured concurrency to keep the app responsive even in error scenarios. All heavy tasks run in background contexts (by default, an `async` call from a SwiftUI `.task` will run on a new task, not blocking UI). If a network call or database save is slow, the UI may show a loading state, but remains interactive (users can cancel or navigate away). We also use **Task cancellation** to avoid work that’s no longer needed: SwiftUI will cancel the task if the view disappears. For example, if the user initiates a sync and then leaves the screen, our `.task` can detect cancellation and stop processing further (perhaps using `Task.checkCancellation()` inside loops or `withTaskCancellationHandler` around important sections). This prevents unnecessary use of resources and avoids weird states (like updating UI for a screen that’s not visible).

In cases where multiple asynchronous operations are performed together (say, parallel network calls to fetch different data sets), we use `TaskGroup` or `async let` to perform them concurrently but still gather errors in an organized way. If one fails but others succeed, we can still update partial data and show an error for the portion that failed.

**Example – concurrent fetch with error handling:** Suppose on launch we want to fetch both latest health metrics and insights at the same time (to minimize wait). We can do:

```swift
do {
    async let metrics = api.fetchHealthData()
    async let insights = api.fetchInsights()
    let (metricsResult, insightsResult) = try await (metrics, insights)
    try await database.saveMetrics(metricsResult)
    try await database.saveInsights(insightsResult)
} catch {
    let appError = ... // map error
    self.error = appError
    logError(appError)
}
```

This way, network calls run in parallel. If either throws, the `catch` will capture it. We might inspect the error to see which one failed (for finer messaging like “Insights failed to load, but metrics updated” – though for simplicity, MVP could just show a generic “Some data failed to load”).

Finally, we consider **UI responsiveness** when errors occur. If an error is non-critical, we may choose not to block the UI at all. For example, if saving to the local database fails for a minor cache update, we log it but maybe don’t alert the user (since the next sync might fix it, and the user wouldn’t notice anything except perhaps stale data). We balance the user experience: show alerts for actions the user initiated or for major issues, but handle minor issues silently or with small indicators.

### Example Error Scenario and Handling

To illustrate the holistic error handling: imagine the user taps “Upload Now” to send recent HealthKit data to the server. The app calls the upload API, but the network drops midway. The flow would be:

1. The upload function throws a `URLError`. This is caught and mapped to `AppError.network` (with a sub-case like `.network(.notConnected)`).
2. Logging: We log via OSLog that the upload failed due to no connection.
3. UI update: The view model sets an `error` property. The UI, observing this, presents a non-intrusive message: “No internet connection. Your data will be uploaded when you’re back online.” Possibly also mark the upload item in the UI with a retry icon.
4. Queuing: Simultaneously, the function adds the unsent data to the pending queue (so it will retry automatically later).
5. The user sees the message and knows the app will handle it. They continue using the app without a crash or a hang.
6. Later, when connection is restored, the queued upload is sent and succeeds. The UI then clears the pending state and maybe shows a brief “✅ Synced!” confirmation.

Throughout this process, our structured error handling ensured the failure was anticipated and addressed gracefully, keeping the app stable and the user informed.

---

By focusing on these three pillars – a solid data sync mechanism (with future-proofing for real-time updates), robust offline support, and comprehensive error handling – the CLARITY Pulse iOS app will provide a reliable and smooth experience. It will remain **responsive** and useful even with spotty connectivity, keep the local and remote data consistent (eventually), and surface issues in a user-friendly manner. These engineering choices align with the backend’s capabilities (e.g. polling asynchronous results, using proper endpoints for data) and prepare the app for future scale and features as the CLARITY platform grows. All code is written with Swift 5.9 and iOS 17 frameworks, leveraging SwiftUI and SwiftData for maximum integration and performance on modern Apple devices.

