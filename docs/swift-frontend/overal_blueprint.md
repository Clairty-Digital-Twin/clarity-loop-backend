CLARITY Pulse iOS App Architecture and Implementation Strategy

Architecture Overview

The CLARITY Pulse app will adopt a Model-View-ViewModel (MVVM) architecture layered with clear separations for UI, ViewModels, Domain Services/Repositories, and Networking. This approach aligns well with SwiftUI’s reactive nature and promotes SOLID design (each layer has a single responsibility and can be extended or tested in isolation) ￼. The goal is a modular, maintainable, and testable codebase where UI is decoupled from business logic, and data handling is abstracted behind well-defined interfaces. We’ll also incorporate GoF patterns like Repository (for data access), Facade/Service (to orchestrate complex operations), and use Dependency Injection to invert dependencies for flexibility and testing.

Key Architectural Decisions:
    •    MVVM Pattern: SwiftUI Views bind to ViewModels (as @StateObject or @ObservedObject), which expose published state and handle actions. This separates presentation from business logic, making code easier to maintain and test ￼.
    •    Layered Structure: We define distinct layers:
    •    UI Layer (SwiftUI Views) – displays data and forwards user intents.
    •    ViewModel Layer – holds UI state, calls services/repositories, and applies presentation logic.
    •    Service/Repository Layer – contains domain logic and data fetching/saving (e.g., HealthKitService, AuthService, Data Repository).
    •    Networking Layer – handles HTTP calls, JSON serialization, and authentication headers.
    •    Persistence Layer (SwiftData) – manages local data models for caching and offline use.
    •    SwiftUI & SwiftData Integration: We avoid coupling SwiftUI views directly to SwiftData’s @Query property wrappers, which can tightly bind the UI to the persistence layer ￼. Instead, a data access layer (Repository) with protocols will abstract the database operations ￼. This ensures views remain UI-focused and our codebase is future-proof (e.g. if switching persistence frameworks or adding tests).
    •    SOLID & DRY Principles: Each component has a single responsibility (e.g., the HealthKit service only handles HealthKit interactions), and we use protocols/abstractions to enforce dependency inversion (ViewModels depend on interfaces, not concrete classes). Code re-use is maximized by centralizing shared logic (e.g., a generic networking client for all API calls, common error handling routines, etc.), avoiding duplication.

Below is an overview of the layer interactions in CLARITY Pulse (from user action down to data layers):

[SwiftUI View] -- (binds to) --> [ViewModel (ObservableObject)]
    |  (calls) -> loadData(), etc.
    v
[Service/Repository Layer] -- e.g., HealthDataRepository, AuthService
    |   (fetches/syncs data)
    v
[Networking Layer] -- APIClient (URLSession)
    |   (HTTP requests, JWT auth)    \
    |                                (remote FastAPI backend)
    v                                                 /
[Persistence Layer] -- SwiftData Model & context  <-'
    ^            (local read/write) 
    |            (HealthKitService reads from HealthKit framework)
    '---- (ViewModel updates UI state via published SwiftData or models) 

In summary, MVVM provides the backbone for a scalable architecture ￼, while repositories and services ensure that business logic and data management are cleanly separated from UI code. Next, we detail each layer and component in this architecture.

UI Layer – SwiftUI Views

SwiftUI will be used for all UI screens, leveraging its declarative and reactive UI updates. Each screen (or significant UI component) is implemented as a View struct that reflects the state of a corresponding ViewModel. Key considerations for the UI layer include:
    •    State Binding: Views observe their ViewModel using @StateObject (for a brand-new ViewModel instance) or @ObservedObject (if injected). The ViewModel’s @Published properties (or @Observable in new Swift macros) drive the UI. For example, a DashboardView might bind to dashboardVM.dailySummary or dashboardVM.isLoading to show content or a loading spinner. The SwiftUI view automatically updates when these values change.
    •    Composition & Reusable Components: Use SwiftUI’s composition to build reusable components for repeated UI patterns. For instance, create a HealthMetricCard view that displays a health statistic (like step count or heart rate) with a title and icon. This could be reused in the dashboard for various metrics. Similarly, an InsightCard view can show an AI insight string in a styled box. By encapsulating these in Views with configurable properties, we adhere to DRY (don’t repeat UI code for each metric/insight).
    •    Example – Health Summary Card Component:
To illustrate a reusable component, consider a SwiftUI view to display a health metric:

struct HealthMetricCard: View {
    let title: String
    let value: String
    let unit: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.headline)
            HStack {
                Text(value).font(.largeTitle).bold()
                if let unit = unit {
                    Text(unit).font(.title3).foregroundColor(.secondary)
                }
            }
        }
        .padding().frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(.systemGray6)))
        .shadow(radius: 1)
    }
}

This HealthMetricCard can be used in a dashboard:

struct DashboardView: View {
    @StateObject private var viewModel = DashboardViewModel()
    var body: some View {
        ScrollView {
            if let summary = viewModel.healthSummary {
                HealthMetricCard(title: "Steps (Today)", 
                                 value: "\(summary.todaySteps)", 
                                 unit: "steps")
                HealthMetricCard(title: "Resting Heart Rate", 
                                 value: "\(summary.restingHR)", 
                                 unit: "BPM")
                // ... other metrics
                InsightCard(text: viewModel.insightText)
            } 
            if viewModel.isLoading {
                ProgressView("Loading data...")  // loading indicator
                    .padding()
            }
        }
        .refreshable { await viewModel.refreshDashboard() }  // pull-to-refresh triggers data load
        .onAppear { Task { await viewModel.loadDashboard() } }  // initial load
        .alert(isPresented: $viewModel.hasError) {
            Alert(title: Text("Error"), message: Text(viewModel.errorMessage ?? "Unknown error"), dismissButton: .default(Text("OK")))
        }
    }
}

In this example, the view shows a loading indicator while data loads, uses .refreshable to support pull-to-refresh, and presents an alert on errors – all reflecting states from the ViewModel. The UI layer remains thin: it simply declares how the UI looks for each state (loading, loaded with data, or error) and delegates actions (like refresh) to the ViewModel.

    •    Navigation and Routing: SwiftUI’s navigation view and stack will manage screen transitions. For the MVP, screens might include: LoginView, DashboardView, and perhaps an InsightsDetailView. Navigation logic (e.g., moving from Login to Dashboard after a successful login) will be driven by observable state (for example, an @AppStorage or environment @State holding login status, or an AuthViewModel published property). Using SwiftUI’s sheet or NavigationLink bindings to bools published by ViewModel can present new screens when needed, keeping navigation declarative and testable.

Overall, the UI layer is designed with stateless Views that render given the state, enabling easily adding new UI features (like a chat screen) by creating new View + ViewModel pairs without breaking other components.

ViewModel Layer – State & Presentation Logic

Each major screen or feature has a corresponding ViewModel (an ObservableObject class) that acts as the “middleman” between the View and the data layers. The ViewModel is responsible for:
    •    Exposing State: It publishes properties the view needs (e.g., healthSummary, insightText, isLoading, errorMessage). These are often simple Swift model types or SwiftData models that the view can display. The ViewModel may also process or format raw data for display (e.g. formatting a date or number), but it does not handle heavy business logic or data fetching by itself – it delegates that to services or repositories.
    •    Handling User Actions: Functions in the ViewModel (often @MainActor async functions if they do async calls) are called by the view for events like refreshing data, logging in, or other button taps. The ViewModel will coordinate the appropriate service calls and update its published state accordingly. This keeps the UI declarative while imperative work happens in the ViewModel.
    •    Coordinating Data Flow: On initialization or on certain triggers, the ViewModel calls into the repository/service layer to fetch or submit data. For example, DashboardViewModel.loadDashboard() might:
    1.    Set isLoading = true.
    2.    In parallel, call AuthService to ensure the user token is valid, call HealthKitService to fetch recent health data, and call a HealthDataRepository to get cached summary or fetch new from backend.
    3.    Merge results (maybe the backend returns a summary including AI insights).
    4.    Update healthSummary and insightText on the main thread.
    5.    Set isLoading = false (or handle errors by setting an errorMessage).
    •    Example – ViewModel Implementation:

@MainActor
class DashboardViewModel: ObservableObject {
    @Published var healthSummary: HealthSummary? = nil
    @Published var insightText: String? = nil
    @Published var isLoading = false
    @Published var errorMessage: String? = nil
    var hasError: Bool { errorMessage != nil }
    
    private let healthRepo: HealthDataRepository
    private let healthKit: HealthKitService
    init(healthRepo: HealthDataRepository = RealHealthDataRepository(),
         healthKit: HealthKitService = HealthKitService.shared) {
        self.healthRepo = healthRepo
        self.healthKit = healthKit
    }
    
    func loadDashboard() async {
        isLoading = true
        do {
            // 1. Fetch latest health data from HealthKit (e.g., today’s steps)
            let newData = try await healthKit.fetchDailyMetrics()
            // 2. Upload to backend via repository
            try await healthRepo.uploadHealthData(newData)
            // 3. Fetch summary & insight from backend (or local cache if up-to-date)
            if let summary = try await healthRepo.fetchLatestSummary() {
                self.healthSummary = summary
                self.insightText = summary.aiInsightText
            }
        } catch {
            self.errorMessage = error.localizedDescription
        }
        isLoading = false
    }
    
    func refreshDashboard() async {
        // For pull-to-refresh, we can simply call loadDashboard
        await loadDashboard()
    }
}

In this pseudocode, DashboardViewModel uses two injected dependencies: a repository for health data and a HealthKit service. This adheres to dependency inversion (we could inject mock ones for testing). The ViewModel coordinates a full data flow: gets data from HealthKit, uploads it, then fetches an updated summary/insight from the backend. It updates published properties to inform the UI of new data or errors. All heavy work is done asynchronously off the main thread (thanks to Swift’s async/await), and UI updates are funneled back on the main actor.

    •    State Management: We use simple enum or booleans to track loading and error states. In the above example, isLoading and errorMessage/hasError represent the view state. Alternatively, we could use a State enum, for example:

enum LoadState { case idle, loading, loaded, error(String) }
@Published var state: LoadState = .idle

and update state accordingly (with associated error messages). The SwiftUI view then switches on this state to show appropriate UI (spinner, content, or error). The approach chosen should keep the UI logic straightforward. The goal is responsive UI – e.g., show a loading spinner immediately when a network call starts, and an error alert if something fails – which the above pattern achieves.

    •    Threading and MainActor: ViewModels that update @Published properties will do so on the main thread. By marking methods as @MainActor or by using DispatchQueue.main.async for final updates, we ensure UI updates happen safely. Swift’s structured concurrency makes this easier by isolating UI-bound state to the main actor.

By adhering to MVVM, we ensure Views remain passive (just reflecting state), while ViewModels handle logic in a way that’s testable (we can instantiate a ViewModel in a unit test with a mocked repository and assert that state changes correctly on certain method calls). This satisfies the separation-of-concerns needed for a robust app ￼.

Data Models and SwiftData Persistence

For local data storage and caching, we use SwiftData, Apple’s new persistence framework in Swift (which is built with a Core Data-like underneath). Data models will be defined as SwiftData @Model classes to integrate seamlessly with SwiftUI and allow observation. We will design these models to mirror the backend’s Pydantic models where appropriate, ensuring consistency between what the app stores locally and the data the FastAPI backend expects.

SwiftData Model Definitions:
Key entities might include User, HealthSample (or specific health metrics like StepCountSample, HeartRateSample), HealthSummary, and DailyInsight. For example, aligning with the backend’s user profile and daily insight models ￼ ￼, we could define:

import SwiftData
@Model
class UserProfile {
    @Attribute(.unique) var id: String  // e.g., Firebase UID
    var email: String
    var displayName: String
    var dateOfBirth: Date?
    var gender: String?
    // ... other profile fields like preferredUnits, etc.
    init(id: String, email: String, displayName: String) {
        self.id = id
        self.email = email
        self.displayName = displayName
    }
}

@Model
class DailyInsight {
    @Attribute(.unique) var insightId: String  // e.g., "insight_20240120_daily"
    var date: Date
    var type: String  // e.g., "daily_summary"
    var summaryText: String  // the narrative summary or main insight text
    var recommendations: [String] = []
    var user: UserProfile?  // relationship to user
    init(insightId: String, date: Date, summaryText: String) {
        self.insightId = insightId
        self.date = date
        self.summaryText = summaryText
    }
}

These are illustrative; actual models would include fields relevant to the app’s needs (e.g., breakdown of sleepAnalysis, activityAnalysis, etc., which could also be separate related models, or simply stored as JSON if not needed individually). We mark some fields unique (like user id, insightId) to avoid duplicates. SwiftData supports relationships (e.g., a UserProfile may have a to-many relationship to DailyInsight entries, useful for caching a history of insights).

Using SwiftData for Offline Caching: All data retrieved from the backend (user profile, health summaries, insights) should be saved to SwiftData. This allows the app to work offline or with intermittent connectivity. For example, after the app fetches the latest daily insight, it can store a DailyInsight model instance. The DashboardView could then use a SwiftData query (or a fetch via repository) to display the last known insight immediately on launch, while simultaneously fetching updates in the background.

Syncing with Backend: The Repository layer (discussed below) will coordinate syncing:
    •    On data fetch from network, insert/update the SwiftData models accordingly. SwiftData’s ModelContext.insert(_:) or updates will persist data locally. By configuring the ModelContainer at app launch (likely in ClarityPulseApp with .modelContainer(for: [UserProfile.self, DailyInsight.self, ...])), we ensure a persistent store is available app-wide.
    •    SwiftData can automatically persist changes on certain events (app going to background, etc.), but we can also explicitly call modelContext.save() if needed after heavy operations to guarantee writes to disk.
    •    If data is updated locally (not typical in this app, since most data originates from backend or HealthKit), we could mark it dirty and sync back to server when online (for MVP, most data flows from HealthKit -> backend -> app).

Best Practices for SwiftData with SwiftUI:
By default, SwiftData encourages using @Query in views to fetch data reactively. However, as noted, this couples views to the persistence implementation ￼. Our approach is to let repositories or ViewModels query SwiftData (via ModelContext.fetch(...) or by maintaining an in-memory copy of needed data) and then publish it. This manual approach requires a bit more code than @Query but yields greater testability and flexibility – we could swap SwiftData out if needed without changing UI code ￼. We can still take advantage of SwiftData’s reactivity: for example, if we keep an @Published var healthSummary: HealthSummary? in a ViewModel, and that HealthSummary is a SwiftData-managed object (i.e., an instance of a @Model class fetched from context), any changes to it (via context) will trigger SwiftUI updates.

Example – Fetching from SwiftData in Repository:

struct HealthDataRepository {
    let container: ModelContainer  // SwiftData container, injected
    func getLatestInsight(from date: Date) throws -> DailyInsight? {
        let context = container.mainContext
        let predicate = #Predicate<DailyInsight> { $0.date == date }
        let descriptor = FetchDescriptor<DailyInsight>(predicate: predicate)
        return try context.fetch(descriptor).first  // get today's insight if exists
    }
    func saveInsight(_ insight: DailyInsight, in context: ModelContext) throws {
        context.insert(insight)
        try context.save()
    }
}

In practice, the ModelContext is typically available via @Environment in a SwiftUI View or via the container. We might use a singleton or a SwiftUI environment object for the ModelContainer to pass it around. For testability, we can configure an in-memory container (as shown in AzamSharp’s example for previews ￼) to simulate the database during unit tests or SwiftUI previews.

The SwiftData models are kept relatively dumb data holders (with optional simple validation logic in model initializers or computed properties). Business logic mostly resides in the service/repo layer, except trivial model logic (e.g., a computed property to format a full name, or basic validations). This aligns with Clean Architecture principles and keeps models reusable. (It’s possible to embed more logic in the model classes as AzamSharp demonstrates for simple rules ￼ ￼, but for our app the heavy logic like “generate insights” or “sync data” doesn’t belong in model objects themselves; instead, we keep those in services).

Networking Layer – REST API Client Design

The networking layer will use Swift’s native URLSession with modern async/await for concise, readable asynchronous networking. We will build a lightweight networking client that handles all HTTP interactions with the FastAPI backend, including JWT authentication, request building, and JSON decoding into Swift models.

Key Elements of Networking Layer:
    •    API Client Singleton or Struct: We create an APIClient (could be a class or a struct with static shared instance) that offers functions corresponding to backend endpoints, for example:
    •    func login(email: String, password: String) async throws -> AuthToken
    •    func fetchHealthSummary(for date: Date) async throws -> HealthSummary
    •    func uploadHealthData(_ data: [HealthSample]) async throws -> Void
    •    etc.
Internally, these use URLSession to make requests.
    •    URLSession with async/await: We use URLSession.shared.data(for: URLRequest) inside our async functions. For example:

struct APIClient {
    let baseURL = URL(string: "https://api.claritypulse.com")!  // hypothetical base
    var authTokenProvider: () -> String?  // closure to get current JWT (injected)
    
    func getSummary(date: Date) async throws -> HealthSummary {
        var request = URLRequest(url: baseURL.appendingPathComponent("/summary/\(date.ISO8601Format())"))
        request.httpMethod = "GET"
        if let token = authTokenProvider() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw APIError.invalidResponse 
        }
        return try JSONDecoder().decode(HealthSummary.self, from: data)
    }
}

This snippet shows injecting the JWT token via a closure, ensuring the API client doesn’t directly depend on the Auth service (inversion of control). We check the HTTP status and decode JSON into a HealthSummary model (which would be a Swift struct or class matching the JSON from FastAPI). We’ll configure JSONDecoder with snakeCase key strategy if needed, to align with Pydantic’s field naming (unless the API returns keys in camelCase).

    •    Secure Token Handling: The Firebase JWT is the linchpin for authenticating with the backend. After a user logs in via Firebase, we obtain a JWT (Firebase ID Token) and store it securely (likely in the iOS Keychain or in-memory Keychain wrapper). The API client will attach this token on every request in the Authorization: Bearer header. To avoid scattering token logic, we use a single source (like the above authTokenProvider or a singleton AuthService that APIClient consults) ￼.
    •    We will also handle token expiration: Firebase tokens last ~1 hour. We can use Firebase’s SDK or REST API to refresh tokens using a refresh token if needed. The networking layer or AuthService should detect a 401 response from backend (which might indicate an expired token if not caught by backend’s own verification) and trigger a token refresh flow transparently, then retry the request. This can be achieved by checking response in APIClient and, on 401, calling a refresh function then retrying once.
    •    Error Handling: The network layer defines an APIError: Error enum for common error cases (e.g., .invalidResponse, .decodingFailed, .networkError(underlying: Error), .unauthorized). The API client functions throw these, which the calling ViewModel or repository will catch. Consistent error types make it easier to display user-friendly messages or take action (for instance, if we get .unauthorized, the app might prompt re-login).
    •    Serialization & Pydantic models: We ensure that our Swift structs/classes for decoding mirror the backend’s Pydantic models in structure and naming. For example, if Pydantic model DailyInsight has insight_id, sleep_analysis, etc., our Swift DailyInsightDTO (Data Transfer Object) struct would have matching coding keys. This reduces transformation code. We might maintain separate DTO structs for network responses and then map to SwiftData models for persistence. For instance, HealthSummaryResponse (Decodable) could be transformed into a HealthSummary SwiftData model object before storing. This mapping could be done in the repository layer. Keeping them separate ensures our app’s internal model can evolve somewhat independently (and decouples persistence from strict JSON structure), though for simplicity if the models align 1-to-1, we might use the same object.
    •    Testing Network Layer: By designing APIClient with injectable dependencies (like the token provider, or even injecting a URLSession or using URLProtocol for stubs), we can write unit tests that simulate network responses. For example, we could inject a custom URLProtocol that intercepts requests for certain endpoints and returns canned data to test our decoding logic, without hitting real network. This ties into our testing strategy later.
    •    Background Requests: If large data uploads are needed (e.g., uploading a large batch of HealthKit samples), we might use URLSessionConfiguration.background with an identifier so that the upload can continue if the app is backgrounded. For MVP, health data volumes are small (a few metrics), so standard requests suffice. But the architecture is ready to integrate background URLSession tasks if needed for efficiency (the UploadHealthData repository method could choose the appropriate session).

In summary, the networking layer is a thin, generic HTTP client plus some API-specific conveniences (like auto-adding auth headers). It will be used by the repository/service layer to communicate with the backend asynchronously.

Repository Layer – Data Access and Coordination

The Repository pattern is used to abstract data operations and provide a unified interface to the rest of the app for each type of data. Repositories serve as a facade to hide whether data comes from the network or local storage, and to implement caching or syncing policies. We will create repositories for major domains, for example:
    •    UserRepository – handles fetching/updating user profile data (perhaps from Firestore or from the backend’s user endpoints). In MVP, user data might mostly come via Firebase Auth (like email, name) and some through backend (like preferences), so this repo coordinates between Firebase and our backend if needed.
    •    HealthDataRepository – handles health metrics data. It will be responsible for sending HealthKit data to the backend (e.g., an uploadHealthData() method) and fetching processed summaries or insights (e.g., fetchLatestSummary() returns a HealthSummary which may include AI insight text).
    •    InsightsRepository – (could be combined with HealthDataRepo for MVP) fetches AI-driven insights or chat responses from the backend. Since MVP insights are static text entries, this might simply be part of the summary fetch. In future, if there’s a separate service or database for insights, an InsightsRepository could encapsulate that.
    •    (Potentially) AnalyticsRepository – if there are endpoints for trend analytics or aggregated data beyond daily summaries.

Responsibilities: Each repository encapsulates the logic for when to use cache vs network. For example, HealthDataRepository.fetchLatestSummary() might:
    •    Check SwiftData if a summary for today exists; if yes, use it (maybe still refresh in background).
    •    If not present or if a forced refresh is needed, call APIClient.getSummary(...) to retrieve from the server, then store it in SwiftData (via ModelContext.insert).
    •    Return the data (either from cache or fresh).

Similarly, UserRepository.getProfile() could first return the locally stored UserProfile (SwiftData) if available, while concurrently hitting the network for any updates (and then updating the cache).

By doing this, the app can show some data instantly (last known state) and then update when new data arrives, improving UX.

Implementation: Repositories can be classes or structs, but typically classes (especially if we want to inject them as reference types into VMs). They will have dependencies injected:
    •    Most will need the APIClient (to call backend) and the ModelContainer/ModelContext for SwiftData.
    •    Some may depend on other services (for example, HealthDataRepository might use HealthKitService for reading HK data).
    •    We define protocols for each repository (e.g., protocol HealthDataRepositoryProtocol) listing the operations (uploadData, fetchSummary, etc.) so that we can mock them in tests. The app will use concrete implementations (e.g., RealHealthDataRepository) that conform to these protocols.

GoF Patterns – Facade and Adapter: In some cases, a repository acts as a facade to multiple lower-level calls. For instance, an OverallDashboardRepository could have a method func loadDashboardData() which internally calls UserRepository, HealthDataRepository, and Insights (or AI) Service, then aggregates results into one DTO for the ViewModel. This isn’t strictly necessary for MVP but demonstrates how to shield the UI from needing to know about multiple calls. We might not implement a dedicated facade class, but the principle is used whenever a ViewModel calls one repository which in turn calls others.

If needed, the repository can also serve as an adapter between different data models – converting, say, Pydantic JSON data into SwiftData model objects, or mapping HealthKit’s data types into our own format for upload.

Example – HealthDataRepository (simplified):

protocol HealthDataRepositoryProtocol {
    func uploadHealthData(_ samples: [HealthSample]) async throws
    func fetchLatestSummary() async throws -> HealthSummary?
}

class RealHealthDataRepository: HealthDataRepositoryProtocol {
    private let api: APIClient
    private let container: ModelContainer  // for SwiftData
    init(api: APIClient = APIClient(), container: ModelContainer = .shared) {
        self.api = api
        self.container = container
    }
    func uploadHealthData(_ samples: [HealthSample]) async throws {
        // Convert samples to DTO for API (if needed) and send
        try await api.uploadHealthData(samples)
        // Perhaps store the raw samples locally if we want an archive
        let context = container.mainContext
        for sample in samples {
            context.insert(sample)  // assuming HealthSample is @Model too
        }
        try context.save()
    }
    func fetchLatestSummary() async throws -> HealthSummary? {
        // e.g., fetch today's summary from backend
        let summaryDTO = try await api.getSummary(date: Date())  
        // Map DTO to SwiftData model
        let context = container.mainContext
        // If HealthSummary is a @Model, we might update or create it
        let summary = HealthSummary(dto: summaryDTO, context: context)
        context.insert(summary)
        try context.save()
        return summary
    }
}

In this pseudocode, HealthSummary(dto:context:) would be an initializer on the SwiftData model to populate from a DTO struct. We show how uploading data might simply pass through to APIClient and then cache the data. The repository ensures all data altering goes through SwiftData for consistency. Also note, .shared ModelContainer implies we might make our ModelContainer accessible globally (perhaps via a static or a singleton PersistenceController). We have to be careful with singletons – better is to inject it – but using a static .shared can simplify usage in small scales. For test and modularity, injection is preferred (so we might instantiate RealHealthDataRepository with a specific container, e.g., a in-memory one for tests).

Offline Behavior: Repositories also define what happens when offline. Using SwiftData cache, the app can still show cached summaries or insights. If the user triggers an upload while offline, the repository could queue the data locally (perhaps marking samples as “to-upload” in SwiftData) and then attempt to sync when connectivity is restored. Implementing a full offline queue is complex for MVP, but our architecture allows adding it later (for example, an offline manager service or leveraging Combine to monitor network reachability and flush queued tasks). At minimum, we’ll ensure the app doesn’t crash and provides feedback if offline (e.g., repository throws a .networkError which ViewModel catches and sets an error message like “No internet connection”).

By using repositories, business logic is isolated: if later the data source changes (say, the backend introduces GraphQL or we move some data to CloudKit), we can adapt inside the repository without affecting UI or ViewModel code. This also aligns with the Open/Closed principle (we can extend data sources by adding new repository implementations or methods, without modifying callers).

HealthKit Integration Service

Integration with Apple’s HealthKit is crucial for CLARITY Pulse to gather health data. We will implement a dedicated HealthKitService (also could be called HealthService or HealthKitManager) to handle all interactions with HealthKit. This service will ensure separation of HealthKit-specific code (which involves permissions and Apple’s APIs) from the rest of the app logic. Key responsibilities include:
    •    Requesting Permissions: On first launch (or whenever needed), the HealthKitService will request the user’s authorization to read relevant health data types. For MVP, this likely includes read access to:
    •    Step count (e.g., HKQuantityTypeIdentifier.stepCount)
    •    Heart rate (HKQuantityTypeIdentifier.heartRate)
    •    Sleep analysis (HKCategoryTypeIdentifier.sleepAnalysis)
    •    Possibly active energy burned, workout data, etc., as required by the backend’s analysis.
The service will use HKHealthStore.requestAuthorization(toShare: nil, read: desiredTypes) to get permission. This call is typically triggered via the onboarding or first dashboard load. The result (granted/denied) will be stored or passed back so the app can handle denial (maybe by showing an alert guiding the user to enable permissions from Settings).
    •    Data Fetching: HealthKitService will provide methods to fetch health data, abstracting the HealthKit queries. For example:
    •    func fetchDailyMetrics() async throws -> HealthDataBatch – which might gather today’s step count, today’s average/resting heart rate, last night’s sleep duration, etc., and package them into a custom struct HealthDataBatch ready to send to the backend. Internally, it might use HKStatisticsQuery for summing steps, HKSampleQuery for heart rate samples, etc.
    •    func fetchHeartRateSamples(range: DateInterval) -> [HeartRateSample] – if needed for more detailed data. For MVP, however, we likely send aggregated data (the backend’s Pydantic models show that they expect sessions with possibly detailed samples ￼, but MVP could simplify to summary stats).
    •    Background Delivery & Low Battery Impact: To minimize battery usage, we will:
    •    Leverage HKObserverQuery and background deliveries: HealthKit can notify the app when new data is available (e.g., new day’s data or new workout). We can register for updates and then perform a background fetch to upload data. This way, we avoid polling. iOS can wake the app in the background for HealthKit updates if configured.
    •    Use BGTaskScheduler for periodic refresh: If real-time sync is not required, schedule a background task (e.g., once a day or when charging) to fetch and upload health data. This ensures tasks run at optimal times (e.g., when the device is charging or at scheduled intervals).
    •    Only fetch necessary data: e.g., for steps, use statistics query for daily total instead of all samples which is efficient. For heart rate, perhaps only fetch resting HR or average if full series isn’t needed.
    •    Monitor battery or power mode: optionally, HealthKitService could skip non-essential background sync if Low Power Mode is on.
    •    Data Upload Pipeline: HealthKitService works closely with the repository: after fetching data, it doesn’t itself send to backend, but hands off to, say, HealthDataRepository.uploadHealthData(). This separation means HealthKitService is focused on retrieving data from HealthKit in a convenient format, and the repository (or a higher-level coordinator) decides what to do with it (upload, store, etc.).
    •    Example – HealthKitService usage in ViewModel:
In DashboardViewModel.loadDashboard() (as shown earlier), we call try await healthKit.fetchDailyMetrics(). Let’s say this returns a struct like:

struct HealthDataBatch {
    let steps: Int
    let stepGoal: Int?
    let restingHeartRate: Double
    let sleepHours: Double
    // ...other metrics
}

The HealthKitService would produce this by querying HealthKit. For instance:

class HealthKitService {
    private let healthStore = HKHealthStore()
    // Define the quantity types we care about
    private let stepType = HKQuantityType.quantityType(forIdentifier: .stepCount)!
    private let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
    // ... other types
    func requestPermission() async throws {
        try await healthStore.requestAuthorization(toShare: [], read: [stepType, heartRateType, ...])
    }
    func fetchDailyMetrics() async throws -> HealthDataBatch {
        let today = Date()  // assume we measure from start of day to now
        let startOfDay = Calendar.current.startOfDay(for: today)
        let endOfDay = Calendar.current.date(byAdding: .day, value: 1, to: startOfDay)!
        // Step count sum query
        let stepQuery = HKStatisticsQuery(quantityType: stepType, quantitySamplePredicate: HKQuery.predicateForSamples(withStart: startOfDay, end: endOfDay), options: .cumulativeSum) { _, stats, error in
            // ... (we'd use continuation to return result)
        }
        // (Simplified: in practice, use HKStatisticsCollectionQuery for steps or async HK query API if available)
        // Heart rate example: fetch resting HR (could use ClinicalVital or just average of samples in resting state if available)
        // ...
        // Compile results into HealthDataBatch
        return HealthDataBatch(steps: totalSteps, restingHeartRate: avgHR, sleepHours: calculatedSleep, stepGoal: maybeFrom user profile or HK?)
    }
}

Note: HealthKit’s API isn’t fully async/await yet (though iOS 17+ might have some async initializers for statistics queries); we might wrap HK queries with async using continuations. The specifics can get complex – the key is the service encapsulates this complexity.

    •    Testing HealthKitService: We will create a protocol (e.g. HealthKitServiceProtocol with the same methods) and use a concrete implementation that talks to HKHealthStore. For unit tests, we can implement a mock version that returns preset data without hitting HealthKit. This allows testing the ViewModel or repository logic without needing actual HealthKit access. Because HealthKit cannot run in unit test environment, this abstraction is necessary for testability.

The HealthKitService ensures our architecture respects SOLID: Health data collection is a separate concern, and other parts of the app don’t need to know how steps or heart rate are fetched. They just ask for data from the service. The service in turn is mindful of UX and performance – it will only operate when authorized and in efficient ways (batching queries, etc.). By designing for background fetch and minimal battery impact, we prepare the app to continuously and quietly sync health data in the future (possibly uploading data overnight or when the user isn’t actively using the app).

Firebase Authentication Flow

User authentication in CLARITY Pulse relies on Firebase Auth (JWT-based), so we need to integrate Firebase SDK (or REST calls) in a secure and user-friendly way. The architecture will include an AuthService and corresponding AuthViewModel to handle login state, token management, and protected API calls.

Authentication Strategy:
    •    Firebase SDK for iOS: We can use Firebase’s official SDK to handle sign-in (with email/password, Apple Sign-In, or other providers as needed). The SDK will manage user credentials and provide us an ID Token (JWT) for the logged-in user. This token is what we’ll pass to our FastAPI backend for authentication (the backend likely verifies it using Firebase Admin or a public key, as implied by the JWT usage).
    •    Alternatively, one could use Firebase’s REST API (/verifyPassword for email login, etc.) to get tokens without the full SDK, but using the SDK is simpler for features like token refresh and easy support of Google/Apple sign-in.
    •    AuthService: We create an AuthService class responsible for:
    •    Initiating sign-in flows (e.g., signIn(email, password) or signInWithGoogle() which under the hood calls Firebase).
    •    Caching the auth state and tokens. Firebase SDK by default caches the user and refresh token in keychain. We will still extract the ID token when needed (Firebase currentUser.getIDToken() gives a fresh token, possibly auto-refreshing if expired).
    •    Providing the current JWT to other parts of the app (perhaps via a AuthService.currentToken computed property or a closure as shown to APIClient).
    •    Signing out (clearing any stored data, and informing the app to go back to a login screen if needed).
    •    Secure Storage: The JWT and refresh token should be kept securely. Relying on Firebase’s internal storage is fine; if doing manually, use Keychain. The app should never expose the JWT in logs or to unintended destinations. Communication with backend is all over HTTPS so the JWT is protected in transit.
    •    Auth State in UI:
    •    We’ll use an AuthViewModel (or even just rely on Firebase’s Combine publisher or state listener) to track if the user is logged in. For example, AuthViewModel could publish a @Published var isAuthenticated and maybe the UserProfile of the logged-in user.
    •    On app launch, we check if Firebase has a cached user session. If yes, fetch an ID token and proceed to main app; if not, show the LoginView.
    •    After login, the token is obtained and stored, and the app transitions to the main interface.
    •    The ClarityPulseApp (SwiftUI App struct) can observe this via an @StateObject for AuthViewModel, switching the root view accordingly:

@main
struct ClarityPulseApp: App {
    @StateObject var authVM = AuthViewModel()
    var body: some Scene {
        WindowGroup {
            if authVM.isAuthenticated {
                DashboardView().environmentObject(authVM)  // main app view
            } else {
                LoginView().environmentObject(authVM)
            }
        }
    }
}

In this setup, the login screen and main app share the same AuthViewModel via environment, which can provide user info and allow logout etc.

    •    Login UI & Flow: The LoginView SwiftUI screen will likely have fields for email/password (for MVP), and a button. It ties to a LoginViewModel which uses AuthService. For example, LoginViewModel.login(email, pass) calls AuthService.signIn(email, pass):

class AuthService {
    func signIn(email: String, password: String) async throws {
        // e.g., using Firebase Auth SDK
        let result = try await Auth.auth().signIn(withEmail: email, password: password)
        // result.user is now logged in, token will be available via getIDToken
    }
    func getCurrentToken() async throws -> String {
        guard let user = Auth.auth().currentUser else { throw AuthError.notAuthenticated }
        return try await user.getIDToken()  // refreshes if needed
    }
}

The AuthService simplifies the Firebase calls and errors. The LoginViewModel would catch errors (e.g., wrong password) and set a published loginError for the view to display.

    •    JWT Passing to Network: As described in the Networking section, we inject the token into requests. We might do this by giving APIClient a reference to AuthService or, better, a closure authTokenProvider so that APIClient remains decoupled. For example, APIClient(authTokenProvider: { await Auth.auth().currentUser?.getIDToken() }) – using Firebase directly in the closure. But more testable is to have AuthService implement a protocol that APIClient can use. E.g., protocol AuthTokenProviding { func validToken() async throws -> String } which AuthService conforms to. Then APIClient can call try await authProvider.validToken() internally.
    •    Token Refresh: Firebase handles refresh under the hood if using their SDK (the getIDToken() call will automatically use the refresh token if needed). We just need to ensure to handle failures (if refresh fails, perhaps due to revoked credentials, we force logout and show login). The backend might also have an expiration check for tokens; the app should pro-actively refresh via Firebase rather than hitting an expired token error from backend. A strategy could be to always fetch a fresh ID token before a critical API call (Firebase tokens are typically valid for 1h; if our sync frequency is less, we might always get a new one to be safe).
    •    Logout and Data Clearing: On logout, AuthService will sign out from Firebase (Auth.auth().signOut()), and we should clear any sensitive cached data in SwiftData (or mark it for deletion). Possibly, we might wipe the SwiftData store or at least remove user-specific entries so that a new login starts fresh. Given SwiftData is local and user-specific, best practice is to segregate data per user (for instance, include userId in model and filter by it, or reset container on logout). We’ll incorporate a method in AuthService or a separate DataResetService to handle that.

By handling authentication as a separate module, we adhere to Separation of Concerns: the rest of the app (like repositories) does not need to know how login happened, just that they can get a token to use. Also, using Firebase ensures security – we rely on a proven system for authentication, and by using JWTs on our API, we benefit from Firebase’s user management while our backend remains stateless JWT auth.

Finally, for testing, we can abstract AuthService as well. For instance, a AuthServiceProtocol can have a isAuthenticated property and a getCurrentToken() method. In tests, we could bypass Firebase and have a dummy that returns a preset token. This way, ViewModels and Repositories can be tested without requiring real Firebase calls (which would be out of scope in unit tests).

Dependency Injection and Modular Design

To keep the app modular and testable, we employ dependency injection (DI) extensively:
    •    Use of Protocols: Most services (AuthService, HealthKitService, Repositories, API client) will have protocols defining their interface. For example, HealthDataRepositoryProtocol as shown, or AuthServiceProtocol. ViewModels will be typed to depend on the protocol, not the concrete class. This means we can inject a mock or stub in tests (fulfilling the same protocol) without changing the ViewModel code. It also allows swapping implementations (e.g., a new version of a service) without altering callers, following the Open/Closed principle.
    •    Initializer Injection: The primary way to inject dependencies in Swift is via initializers. As seen in the DashboardViewModel snippet, we pass healthRepo and healthKit with default arguments. In production, the defaults can be actual singletons or real instances, but in tests, we can supply a dummy. Other ViewModels and services will follow suit. For example, LoginViewModel might accept an AuthServiceProtocol in its init.
    •    Environment Objects and Singletons: For app-wide dependencies like the ModelContainer (SwiftData stack) or AuthService, we might use SwiftUI’s @EnvironmentObject or singletons. For instance, a PersistenceController singleton could hold the ModelContainer and context. We might set ModelContainer.shared = … after initializing in the App struct. Another approach: use SwiftUI’s environment to pass down the ModelContext. Actually, SwiftData requires us to set up a container via .modelContainer(for: [ModelType]) modifier on a view (usually at the App’s root). We will do that on the root view so that SwiftUI’s environment has a modelContext available. Then, if needed, we can obtain it in our repository by having an environment injection or by capturing it via a view initializer (less ideal).
    •    One way is to initialize repositories inside the SwiftUI App and inject them as environment objects too. For example:

let container = try ModelContainer(for: [UserProfile.self, DailyInsight.self])
let apiClient = APIClient(authTokenProvider: { await authVM.token }) 
let healthRepo = RealHealthDataRepository(api: apiClient, container: container)
let healthKitService = HealthKitService()
WindowGroup {
   DashboardView()
     .environmentObject(authVM)
     .environment(\.modelContext, container.mainContext)
     .environmentObject(healthRepo)
     .environmentObject(healthKitService)
}

In the above pseudo-configuration, we set up all dependencies in one place (the composition root). This makes it clear how the app is wired together. ViewModels then can fetch their needed objects via the environment if we choose (for example, DashboardViewModel could use @EnvironmentObject var healthRepo: RealHealthDataRepository instead of injection in init). However, environment injection can hide dependencies and complicate reuse, so often initializer injection is preferred for non-global objects. We might use a mix: environment for truly global singletons (Auth state, container) and explicit for feature-specific ones.

    •    SwiftUI Environment vs DI: We should balance using SwiftUI’s convenience (like @Environment(\.modelContext) which gives the persistence context in any view) and keeping our architecture decoupled. A good compromise is to use environment objects for very central services and use factory/initializers for others. Also, if we use environment to inject, say, a repository into child views, those views could pass it into their ViewModel’s initializer. This way the ViewModel remains testable (you can still init it with a different repository in tests).
    •    Dependency Inversion Principle (SOLID): Our architecture is designed such that high-level layers (like ViewModels) do not depend on low-level implementation classes directly. For example, a ViewModel knows only about HealthDataRepositoryProtocol and not that it’s using URLSession under the hood. This decoupling is classic DIP: it makes components interchangeable and the code more maintainable ￼.
    •    Modular Structure: We could organize code into separate Swift packages or groups: e.g., Networking module (with API client and models), Services module, etc. For an initial MVP, using project groups is fine, but as it grows, considering Swift Package Manager for self-contained modules (like a package for the Data layer) can enforce boundaries. This is an optional refinement.

Example – Dependency Injection in Tests:
Suppose we want to test the DashboardViewModel’s logic. We can create a fake repository:

class FakeHealthDataRepo: HealthDataRepositoryProtocol {
    var didUpload = false
    var testSummary: HealthSummary?
    func uploadHealthData(_ samples: [HealthSample]) async throws {
        didUpload = true
        // do nothing or record the samples
    }
    func fetchLatestSummary() async throws -> HealthSummary? {
        return testSummary  // return a pre-set summary
    }
}

And a fake health kit:

class FakeHealthKitService: HealthKitServiceProtocol {
    var testData: HealthDataBatch
    init(testData: HealthDataBatch) { self.testData = testData }
    func fetchDailyMetrics() async throws -> HealthDataBatch {
        return testData
    }
}

Now in the test:

func testDashboardLoadsData() async {
    let fakeRepo = FakeHealthDataRepo()
    fakeRepo.testSummary = HealthSummary(todaySteps: 1000, restingHR: 60, ... , aiInsightText: "Test insight")
    let fakeHK = FakeHealthKitService(testData: HealthDataBatch(steps: 1000, restingHeartRate: 60, sleepHours: 8))
    let vm = DashboardViewModel(healthRepo: fakeRepo, healthKit: fakeHK)
    await vm.loadDashboard()
    XCTAssertFalse(vm.isLoading)
    XCTAssertNotNil(vm.healthSummary)
    XCTAssertEqual(vm.healthSummary?.todaySteps, 1000)
    XCTAssertEqual(vm.insightText, "Test insight")
    XCTAssertTrue(fakeRepo.didUpload)  // ensure it attempted upload
}

This unit test verifies the ViewModel logic with no real network or health data needed. This demonstrates how our DI approach yields high testability: each piece can be isolated and verified.

Overall, by consciously injecting dependencies and abstracting with protocols, we ensure the app’s components are loosely coupled and adherent to SOLID principles – especially Dependency Inversion and Single Responsibility (each class has one job and relies on abstractions for anything else).

Testing Strategy

Building a production-grade app requires a strong testing strategy. We will employ multiple testing methods:
    •    Unit Tests (Business Logic): We will write unit tests for ViewModels, services, and repositories:
    •    ViewModel tests: using fake/mocked services as shown above, we verify that for given inputs or simulated service responses, the ViewModel updates its published properties and state correctly. For example, test that LoginViewModel.login sets an error message on wrong credentials (by injecting an AuthService that throws an auth error), or that DashboardViewModel.loadDashboard populates summary and insight as expected.
    •    Service/Repository tests: test HealthKitService in isolation (this might be tricky without actual HealthKit; we can structure it to have parts that can be tested with sample data). Alternatively, abstract out the HKHealthStore behind a protocol so it can be simulated. For the networking layer, we can use URLProtocol stubs or inject a mock APIClient to test the repository’s behavior (e.g., repository correctly saves data to SwiftData when API returns certain JSON).
    •    Model tests: If we have any significant logic in SwiftData models (like computed properties or validation methods), we test those by creating instances in an in-memory ModelContainer and checking their behavior. SwiftData’s inMemory container is useful here ￼.
    •    UI Tests (Integration): Using Xcode’s UI Testing suite, we will simulate user interactions in the app to catch any integration issues:
    •    For example, a UI test can launch the app, enter sample credentials on LoginView, tap login, wait for the dashboard, and verify that a certain element (like the steps count Text) appears. We can use Accessibility IDs to easily find UI elements.
    •    UI tests will rely on either using a real backend (not ideal for CI) or a stubbed environment. We can launch the app with launch arguments/environment variables that tell the app to use a Mock APIClient implementation that returns fixed data, ensuring predictable results (this might be achieved via dependency injection at runtime or using Swift’s @testable to override certain behaviors).
    •    SwiftUI views are highly testable via UI tests because of their deterministic rendering given state. Additionally, snapshot testing (comparing rendered UI to reference images) could be used for key components (though maintenance of snapshots can be heavy).
    •    Automated Testing for HealthKit & Permissions: We might create unit tests for the logic that formats HealthKit data, but actual HealthKit queries can’t run in a test environment (no entitlement). Instead, we rely on integration testing on device for that – or we abstract the data source as mentioned. We will manually test that permission prompts appear and data flows correctly, then capture that logic in unit tests by simulating the outcomes.
    •    Continuous Testing: As the project grows, setting up CI to run unit tests on each commit will help maintain quality. For UI tests, perhaps run them on a nightly build or a real device lab if available.
    •    Testable Architecture Benefits: Because our architecture separates concerns, we can test each part in isolation:
    •    Auth: Use Firebase emulator or a separate test project if needed, or bypass by injecting a fake token provider to test flows that require auth.
    •    ViewModels: no UI involved, fast logic tests.
    •    Repositories: can test with an in-memory SwiftData (so no persistent side effects) and a stub network.
    •    Error scenarios: easily simulated by having a mock service throw errors, verifying the app responds gracefully (e.g., error message shown).
    •    SOLID in Testing: Thanks to dependency inversion, our tests don’t have to deal with concrete FirebaseAuth or HKHealthStore or URLSession if we provide interfaces. This dramatically reduces flakiness and complexity in tests, fulfilling our goal of a testable codebase.

In summary, testing will be a mix of unit tests for logic, and UI tests for end-to-end flows. The first MVP aim is to have core logic covered by unit tests (especially around the critical data syncing pipeline). We will also use SwiftUI Previews during development for rapid feedback on UI layouts and maybe even to exercise some states (SwiftUI previews can inject sample data and are very useful for visual validation, though not a substitute for programmatic tests). By baking testing considerations into the architecture from the start, we ensure higher code quality and easier maintenance.

MVP Scope – Full Data Flow Slice

For the Minimum Viable Product, we will implement a vertical slice that covers one complete data cycle: from HealthKit, through the backend, to the UI. Focusing on a single cohesive scenario ensures we wire up all layers end-to-end. The proposed MVP flow is:
    1.    User Authentication: A simple Email/Password login using Firebase Auth. The MVP will include a Login screen where the user enters credentials (or uses an existing test account), and upon success the app obtains a Firebase JWT and transitions to the main dashboard. (User registration could be handled either via a separate screen or assumed pre-created accounts to keep MVP small.)
    2.    Health Data Permission & Fetch: After login (or on first dashboard load), the app will prompt for HealthKit permission to read the required data. Once granted, the app (via HealthKitService) immediately fetches the latest health metrics. For MVP, let’s pick a concrete example: daily step count (and possibly one other metric like resting heart rate). This gives us numeric data to send and display.
    3.    Upload to Backend: The fetched HealthKit data (e.g., today’s step count total) is sent via the HealthDataRepository.uploadHealthData() call to a FastAPI endpoint (say, POST /api/v1/healthkit/upload with JSON containing step count and maybe date). On the backend, this would likely trigger data processing (which for MVP might be simplified or instantaneous).
    •    We have a hint from the repository that there is an endpoint for healthkit upload. The app will call it and expect a response (maybe just 200 OK or perhaps an immediate summary).
    4.    Fetch Summary & Insights: After uploading (or in parallel, depending on backend design), the app fetches the health summary for the day, which includes any AI-generated insight. For MVP we assume the insight is pre-computed or quickly computed. The app calls, for example, GET /api/v1/summary/today or GET /api/v1/insights/daily/{date}. The response could be a JSON like:

{
   "date": "2024-01-21",
   "steps": 10000,
   "restingHeartRate": 60,
   "insight": "You reached your step goal today! 👍 Keep it up."
}

(This is a simplification of the detailed Pydantic models, but sufficient for MVP UI.)

    5.    Store and Display Data: The app receives the summary/insight, saves it to SwiftData (creating or updating a DailyInsight or HealthSummary record in the local store), and updates the ViewModel’s published state. The DashboardView then automatically reflects the new data: e.g., showing “10,000 steps” and the insight text. If the network call fails, the ViewModel sets an error message, and the UI presents an alert – ensuring the user is informed (e.g., “Could not sync data. Pull to retry.”).
    6.    Persistence & Offline use: The downloaded summary and insight are now cached locally. If the user closes and reopens the app (without a new day starting), the Dashboard can load the cached summary immediately from SwiftData and show it, even without hitting the network. This meets a basic offline requirement. The architecture for MVP will focus on this one-day summary; trend graphs or longer history might not be implemented yet (we can show just today’s metrics and one or two insight sentences).

By implementing this slice, we cover authentication, permission handling, data fetch, data upload, backend interaction, local persistence, and UI update – essentially every layer in action:
    •    LoginView -> AuthViewModel -> AuthService/Firebase (auth part)
    •    DashboardView -> DashboardViewModel -> HealthKitService & HealthDataRepository -> APIClient & SwiftData (data part)

MVP Development Steps:
To achieve this, the development can be broken down as:
    1.    Setup Project: Configure SwiftUI lifecycle app and add Firebase SDK. Set up SwiftData container.
    2.    Auth Implementation: Create AuthService, AuthViewModel, LoginView UI. Test logging in (perhaps with a dummy Firebase project initially).
    3.    HealthKit Integration: Implement HealthKitService with permission request and a basic fetch (e.g., steps). Write a simple test to ensure it returns plausible data (or at least handle the async).
    4.    Networking: Implement APIClient with a placeholder endpoint for upload and summary fetch. If the backend isn’t ready, simulate the response (we could stand up a dummy FastAPI for local testing or use Postman Mock server). Ensure JWT from Firebase is attached.
    5.    Repository & Data models: Create SwiftData models for what we need (maybe HealthSummary with just fields for steps and HR and insight text). Implement HealthDataRepository’s upload and fetch using the APIClient, and saving into SwiftData.
    6.    Dashboard ViewModel & UI: Put it all together in DashboardViewModel, calling HealthKitService and repository as described. Build DashboardView to show the data nicely (using our cards for metrics and insight).
    7.    Testing: Write unit tests for DashboardViewModel (with a fake repository as above), and for LoginViewModel (with fake AuthService). Verify basic logic.
    8.    UX Polish: Ensure the loading states show correctly (e.g., use .overlay for spinner perhaps), and error handling (maybe simulate no internet scenario to see the alert). Also, handle the case of permission denied: if HealthKit permission is denied, perhaps show an informative message in the dashboard instructing the user to enable it (and skip the upload attempt).
    9.    Background Mode (if time permits): Try scheduling a background fetch (BGTask) after login, which will call HealthKitService and repository periodically. If this is too much, leave as a to-do for post-MVP.

By focusing on this vertical slice, we ensure the architecture is proven out. Once MVP is done, adding more metrics (e.g., weekly trends or additional health parameters) will mostly involve extending the HealthKitService queries and updating the UI – the underlying infrastructure (network, auth, persistence) will already be in place.

UX and Performance Considerations

The architecture is designed not only for code quality but also to support a smooth user experience:
    •    Responsive Loading: We make liberal use of SwiftUI’s ability to show interim UI states. Each network or lengthy operation is accompanied by state toggles to show activity indicators. The user should rarely be left wondering if the app is doing something. For example, when logging in, we can show a ProgressView over the login form or disable the button to indicate progress. The DashboardViewModel’s isLoading drives a spinner for data refreshes. Using async/await avoids blocking the main thread, keeping the UI fluid.
    •    Error Presentation & Handling: All errors are caught in ViewModels and translated to user-friendly messages. The architecture centralizes error strings (possibly in the repository or service, e.g., mapping HTTP 500 to “Server error, please try later”). The UI will use SwiftUI’s .alert or a dedicated error view modifier to inform the user. Also, by structuring network and service calls with throws, we ensure errors bubble up clearly to be handled in one place (rather than causing unpredictable issues). For persistent issues (like no internet), we could implement a retry mechanism or instruct the user to check connectivity, ensuring the app fails gracefully.
    •    Background Sync: Although MVP might require the user to open the app to sync, the architecture anticipates background sync:
    •    By having HealthKitService and repository methods that can be called from a background AppDelegate or BGTask, we separate UI from data sync. We can register a BGTask that calls a sync function in HealthDataRepository (which in turn uses HealthKitService and APIClient). Thanks to DI, this sync function can be the same one used by the ViewModel. SwiftData can be accessed in background tasks as long as you create a ModelContainer on a background thread (or perhaps use the same container if threadsafe via the model context’s actor).
    •    When the app launches or comes to foreground, the ViewModel could check if a background sync happened (maybe by a timestamp stored in UserDefaults or by checking SwiftData for new records) and update the UI immediately. This way, if the app updated insights overnight, the user sees fresh data on launch without waiting.
    •    Modular Feature Extensibility: The clear-cut boundaries mean we can add features without cluttering existing code. For instance, adding a Chat interface with the AI down the line:
    •    We’d introduce a new View + ViewModel (for chat), and maybe an AIChatService that opens a WebSocket or uses a streaming endpoint. This service can reuse the same APIClient (if using HTTP) or a new network stack for WebSockets. It would be separate from the HealthDataRepository to follow single-responsibility.
    •    The App’s environment could include this new service, and the navigation could present a ChatView when the user taps something like “Ask AI” on the insights screen. The rest of the app remains unaffected by this addition, confirming that our modular design works.
    •    Efficiency: We pay attention to not doing duplicate work:
    •    If multiple views need the same data (e.g., profile info on settings and dashboard), we can have one source (UserRepository or an @EnvironmentObject for current User) so that fetching happens once and is shared.
    •    SwiftUI views are lightweight, and the heavy lifting is done in background threads. We avoid blocking the UI. Also, SwiftData automates a lot of caching in memory, so accessing frequently read objects (like user profile) is cheap after initial fetch.
    •    The use of asynchronous pipelines ensures that, for example, we can fetch HealthKit data and network data concurrently if needed.
    •    We also consider memory: SwiftData ensures only needed objects are loaded; large health datasets (if any) might be better fetched in summary form to not overwhelm the app.

By building these UX considerations into the architecture (loading states, error flows, background capabilities), we ensure the app feels robust and responsive to the end user, even in the initial version. Good architecture enables good UX by making it easy to implement these feedback loops and not get bogged down in spaghetti code to handle every little state.

Extensibility and Roadmap (Post-MVP)

The chosen architecture sets up CLARITY Pulse for future growth. After the MVP is validated, the following features can be added relatively easily due to the modular design:
    •    AI Chat Interface: To support a conversational interface with the AI (Gemini 2.5 Pro), we can introduce a new ChatViewModel and ChatView. This ViewModel might use a new ChatService that manages a WebSocket connection to the backend (if the backend provides a real-time chat endpoint) or uses long-polling/streaming HTTP responses.
    •    The ChatService could implement sending user queries and receiving AI responses. Using Combine or AsyncStream, it can publish incoming messages to the ViewModel.
    •    Because our networking layer was designed with JWT auth and possibly can be extended to WebSockets, adding a WebSocket client that uses the same token for authentication is straightforward. We’d just ensure the AuthService can provide a token for the connection handshake (perhaps in the URL or headers).
    •    The UI for chat would be a new SwiftUI view showing a scrollable list of messages and a text field to send new messages. It would be another layer but isolated – it may reuse some models (like maybe the DailyInsight or context from recent data to send to AI).
    •    Importantly, adding this doesn’t tangle with the health data flow; it’s a separate module. We can even package it as a separate SwiftUI view module if needed. The only integration point might be that the ChatView could be launched from an Insight (“Ask for more details” button, for example).
    •    Real-time Updates: Beyond chat, if we want the app to get real-time updates (say the backend completes processing new insights while the user is in the app and pushes them), we could integrate WebSockets or Push Notifications:
    •    If using WebSockets, an InsightsWebSocketService could maintain a connection and notify the InsightsRepository or directly the ViewModel of new insight data. Our architecture can accommodate this by updating the SwiftData store on a new message, which in turn would update UI (since SwiftUI would see the data change).
    •    If using APNs (push notifications) for something like “New insight available!”, we’d handle that in AppDelegate/SceneDelegate and route it to update the relevant repository (e.g., trigger a fetch of new insights when a push is received). This again would leverage the existing data flow – essentially just another trigger for the repository to fetch data.
    •    Additional Health Metrics and Features: With the base in place, adding more HealthKit data types (nutrition, mindfulness, etc.) is mainly extending HealthKitService to request those permissions and fetch data, then updating the data models and UI to present them. The modular design ensures these additions don’t require rewriting core logic, just augmenting:
    •    E.g., to add weekly trend charts, we might create a TrendRepository or extend HealthDataRepository with methods to fetch weekly aggregates (perhaps from backend’s Cloud SQL). The UI would have a TrendsViewModel and view with SwiftUI’s Charts framework to plot data.
    •    The SwiftData models might need to store historical data (e.g., a list of step counts for past 7 days). We can add a new @Model or extend existing ones. The repository would manage filling those from backend or computing from raw data. Because our data layer is abstracted, these changes localize to that layer and the new UI; other parts (like login or chat) remain untouched.
    •    Multi-platform or Watch Support: Although not asked, a well-architected SwiftUI app can potentially share code with a watchOS app or macOS (via Mac Catalyst or SwiftUI on Mac). Our separation of UI and logic means we could, for example, reuse the HealthKitService and repositories on watchOS (except watchOS has its own HKHealthStore, but we could abstract the differences). If that were a goal, we’d further ensure not to use any API that’s iOS-only in shared code or guard it appropriately.
    •    Analytics and Logging: As the app grows, having a logging mechanism or analytics (to Firebase Analytics or another service) is easier to insert when using a centralized architecture. For instance, one could add logging in the repository for each network call success/failure or in the ViewModel for user actions, without cluttering UI code. Or use a Publisher in the AuthService to listen for events (like sign-in) and send analytics. The clean separation helps identify where to put such cross-cutting concerns.
    •    Code Maintenance: With SOLID principles, adding new developers to the project or performing refactors is more manageable. For example, if a new requirement says “use a GraphQL API instead of REST”, we could swap out the Networking layer implementation (APIClient) and perhaps repository internals, but the ViewModels and Views remain unchanged (since they call the same repository interface). Likewise, if Apple updates SwiftData or if we needed to move to CoreData/Realm, our data access is abstracted enough to accommodate that with minimal changes to higher layers (as AzamSharp notes, abstracting persistence can future-proof the app ￼ ￼).
    •    SOLID and Future Features: We continue to honor SOLID as we extend: new features get their own classes (Single Responsibility), we favor composition/DI over adding if-else in existing code (Open-Closed), we rely on protocol interfaces (Liskov substitution: e.g., new repository impl can replace old), we design small, focused interfaces for new services (Interface segregation), and we wire new dependencies via injection (Dependency inversion).

In conclusion, the architecture described provides a strong foundation for CLARITY Pulse. MVVM ensures a clear separation of UI and logic, SwiftData gives local persistence integrated with SwiftUI, and the additional layers (services, repositories, networking) enforce a clean structure that is easy to test and evolve. The production-grade considerations (error handling, background sync, caching, DI) built into this plan will help avoid common pitfalls and make the app robust from day one. With this blueprint, the development team can confidently implement the MVP and iteratively expand the app’s capabilities (such as AI chat and real-time updates) without needing to rethink the core architecture. Each piece has its defined role, and together they form a coherent, scalable system ready to deliver a quality mobile experience for the CLARITY Digital Twin ecosystem.

Sources:
    •    Darren Thiores, “The Ultimate Guide to SwiftData in MVVM: Achieves Separation of Concerns” – notes on using SwiftData outside of SwiftUI and benefits of MVVM for testability ￼ ￼.
    •    Azam Sharp, “SwiftData Architecture – Patterns and Practices” – guidance on abstracting persistence with protocols to avoid coupling UI to SwiftData and to ease future changes ￼ ￼.
    •    CLARITY Backend Pydantic Models – used to align data models (e.g., DailyInsight structure) between app and server ￼ ￼.
