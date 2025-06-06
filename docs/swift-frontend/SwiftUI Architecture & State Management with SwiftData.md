# Proposed Architecture for the CLARITY Pulse iOS App (SwiftUI + SwiftData)

## 1. **Architecture Pattern Selection and Rationale**

For CLARITY Pulse’s SwiftUI app, we recommend an **MVVM-based architecture augmented with Clean Architecture principles**, rather than a purely “Model-View” approach or an overly complex framework. This choice aligns with SwiftUI’s design and keeps the code scalable and testable:

* **MVVM (Model-View-ViewModel)** is a natural fit for SwiftUI because views can bind to observable view models for state. It separates UI from business logic, improving maintainability and testability. SwiftUI’s **new Observation framework (iOS 17)** further streamlines MVVM by allowing observable objects to drive UI updates with minimal boilerplate.
* **Clean Architecture layering** (or “modular MVVM”) complements MVVM by introducing **Use Cases/Interactors** and **Repository/Service** layers. This ensures complex asynchronous flows (HealthKit queries, AI calls, Firebase sync) are handled in dedicated layers, keeping view models simple. Separating concerns like this prevents the **“massive view” or “massive view-model”** problem and makes the app easier to evolve and test.
* We considered **The Composable Architecture (TCA)** as it excels at managing state and side-effects in large apps. TCA could be a valid choice for a very complex state management scenario, but for an MVP it may introduce needless complexity. Instead, our MVVM+Clean approach achieves a similar separation of state and logic in a more lightweight, SwiftUI-native way. It leverages Swift’s structured concurrency (async/await) and the Observation API to handle asynchronous data flows without a heavy external framework.
* This hybrid approach provides a **clear structure for modular growth**. New features (e.g. additional dashboards or AI tools) can be added as new use cases and view models, with minimal impact on existing code. **Testability** is also well-supported: business logic lives in view models or use case classes that can be unit-tested in isolation (with SwiftUI views being simple to preview or test via UI tests).

**Why not a simplistic SwiftUI MV pattern?** While SwiftUI allows putting state and logic directly in views (“Model-View”), that can lead to large, untestable view code in non-trivial apps. MVVM moves logic to view models, and Clean Architecture ensures even those view models stay lean by delegating data access to repositories. This division of responsibilities is crucial given CLARITY Pulse’s features (health data sync, AI chat, etc.), which involve multiple frameworks and asynchronous tasks.

## 2. **Layered Architecture Structure**

To enforce modularity, we define clear layers and modules (each potentially as separate groups or Swift packages):

* **UI Layer (View + ViewModel):** SwiftUI Views are declarative and subscribe to state from ViewModels. Each feature (Authentication, Health Dashboard, Chat, etc.) has:

  * **SwiftUI View:** Responsible for layout and presenting data. Contains minimal logic – primarily uses SwiftUI dataflow (e.g. `.onAppear` to trigger data load via the ViewModel). The view observes the ViewModel (using `@StateObject` or the new `@Observable` macro) and updates automatically. User interactions (button taps, etc.) are forwarded to the ViewModel.
  * **ViewModel:** An `ObservableObject` (or `@Observable` class) that holds the **state** for the view (published properties for data, loading flags, error messages, etc.) and the **logic** to handle user intents. It calls Domain layer use cases or services to perform work. ViewModels transform raw data into view-ready form (e.g. formatting dates or combining HealthKit records for a chart). They do *not* do direct networking or database calls – those are delegated to the lower layers. This separation keeps UI logic testable outside of SwiftUI.

* **Domain Layer (Use Cases & Domain Models):** Encapsulates business logic and complex operations, decoupled from UI and low-level data details.

  * **Use Cases / Interactors:** Each represents a specific functionality or business process (e.g. `FetchDailyHealthSummary`, `SyncHealthData`, `GenerateInsightMessage`). These are plain Swift classes or `actor`s that orchestrate calls to repositories/services. A use case may fetch data from multiple sources (e.g. HealthKit + SwiftData cache) and apply any necessary business rules (e.g. computing trends or thresholds for insights). By centralizing this logic, we avoid duplicating it in multiple view models and make it easy to adapt or optimize in one place.
  * **Domain Models:** High-level data structures used within the app’s logic, distinct from database or API schemas. For example, a `DailyHealthSummary` struct might combine steps count, sleep hours, etc., regardless of how those are stored or retrieved. Domain models ensure the Domain layer doesn’t depend on SwiftData or HealthKit types, aiding testability and potential future platform sharing.
  * This layer defines **contracts** via protocols for what needs to be done (e.g. an `HealthDataRepositoryProtocol` with methods like `fetchDailyMetrics(date:)` or `saveWorkout(_:)`). The actual implementations live in the next layer.

* **Data Layer (Repositories & Services):** Responsible for data operations – whether from device APIs, persistence, or network. Each major data source has a service or repository:

  * **Firebase Auth Service** (e.g. `AuthService`): Handles user authentication, sign-in/out, and user account data, using Firebase SDK. Exposes methods like `login(email,password)` or publishers/async functions for auth state. The rest of the app relies on an `AuthRepositoryProtocol`, not on Firebase directly, allowing replacement or mocking.
  * **HealthKit Service** (e.g. `HealthKitRepository`): Encapsulates all HealthKit interactions. For example, it knows how to request permissions, query health data (steps, heart rate, etc.), and maybe handle background deliveries. It provides async APIs like `readDailyStats()` that return domain models (or raw data that the Domain layer will process). This isolation means the Health feature of the app can be developed and tested against a consistent API, and if Apple HealthKit changes or we add a new data source, the impact is limited to this component.
  * **AI Service** (e.g. `InsightAIService`): Manages calls to the AI models (PAT model, Gemini AI). For instance, a method `queryInsight(for question: String)` that calls a remote API or local model and returns an answer or a stream of messages. It could internally call a cloud function or use URLSession – the ViewModel doesn’t care, it just awaits a result. Abstracting AI behind a service interface also allows using different models or switching from a dummy implementation to a real one easily.
  * **Persistence Repository** (e.g. `DataRepository` or specific ones like `InsightsRepository`): Handles local data via SwiftData. For example, saving retrieved health summaries or caching AI chat history in SwiftData. It might also coordinate syncing with cloud (e.g. saving a summary to Firebase Firestore if needed). In our case, SwiftData is the local store framework – the repository hides its details from the rest of the app. This layer may also include any networking (if not covered by the above services) or file storage as needed.
  * **Integration of Local & Remote:** Some repository implementations may combine sources. For example, a `HealthDataRepository` might fetch from HealthKit and also update SwiftData cache, or a `UserProfileRepository` might read from SwiftData cache first, then refresh from Firebase. The goal is to present a simplified interface to the domain layer (e.g. “give me user’s profile data”) while handling the complexity of syncing data behind the scenes.

**Roles & Responsibilities:** The view layer deals only with *presenting* data and minor UI state; the view model handles *interaction logic* and state for the UI; the domain use cases encapsulate *business rules* or complex workflows; and the data layer provides *clean data access* (whether from device APIs, network, or database). This separation ensures each part of the app has a single responsibility, making the app easier to extend. For example, adding a new “Weekly Trends” screen might involve creating a new view + view model, a new use case (e.g. `ComputeWeeklyTrends`) and possibly a new repository method. Each can be developed and tested in isolation and plugged into the architecture without monolithic changes.

## 3. **SwiftData Integration and Persistence Management**

SwiftData (Apple’s new persistence framework in iOS 17) will be leveraged for local data persistence (e.g. storing health insights, caching HealthKit readings, saving user settings or chat transcripts). **Integrating SwiftData** requires careful handling of its ModelContext in our layered architecture:

* **Model Setup:** Define SwiftData `@Model` classes for the data we need to persist (for example, a `HealthRecord` model or `Insight` model). In the App startup (`@main` App struct), we can configure a `ModelContainer` with these models and attach it to the SwiftUI environment using `.modelContainer(for: [ModelType])`. This makes a `ModelContext` available throughout SwiftUI via `@Environment(\.modelContext)` if needed. For instance, if we have `Insight` as a model, we include it in the container so SwiftData knows about it. SwiftData automatically persists changes on certain app lifecycle events (e.g. moving to background) and handles saving, which simplifies persistence logic.

* **Use of a Data Controller:** To ensure our architecture remains UI-agnostic and testable, we **abstract SwiftData’s context management out of the views**. We introduce a **Persistence Controller** (e.g. `SwiftDataManager`) as a singleton or injectable object. This controller creates and owns the `ModelContainer` and `ModelContext` on launch. By centralizing this, we avoid scattered `@Environment(\.modelContext)` usage and can even use SwiftData in non-UI code (or in future, on other platforms) easily. For example, `SwiftDataManager.shared` will initialize the container (with our model types) and provide a `mainContext` (of type `ModelContext`).

* **Repository usage of SwiftData:** Repositories that need persistence will use the `SwiftDataManager` to access the context. For instance, a `HealthDataRepository` might have:

  ```swift
  struct HealthDataRepository: HealthDataRepositoryProtocol {
      private let context = SwiftDataManager.shared.mainContext
      func saveDailySummary(_ summary: DailyHealthSummary) throws {
          let record = HealthRecord(from: summary)
          context.insert(record)
          try context.save()  // if manual save is needed
      }
      func fetchDailySummary(for date: Date) -> DailyHealthSummary? {
          // Use a FetchDescriptor on context to retrieve HealthRecord for the date
      }
  }
  ```

  By confining these calls in the repository, we ensure that **SwiftUI views or view models do not directly perform database operations**. This improves testability – we can swap out the repository with a mock without dealing with SwiftData in tests. It also means if we needed to replace SwiftData with another persistence layer (say Realm or CloudKit) in the future, we’d only change this layer.

* **Observing data changes:** SwiftData provides the `@Query` property wrapper for SwiftUI views to automatically fetch and track changes in data (similar to Core Data’s FetchedResultsController). In our architecture, we have two approaches:

  * *Simple scenarios:* For straightforward lists or content (e.g. list of insight messages or records), we might use `@Query` in SwiftUI views for quick binding – it will auto-update the UI when data changes on disk. This is convenient and reduces code, leveraging SwiftData’s integration with SwiftUI.
  * *Advanced scenarios:* When more control is needed (complex filtering, combining with remote data, etc.), the ViewModel will explicitly fetch data via the repository. The repository can use SwiftData’s `FetchDescriptor` on the ModelContext to query objects and return them. The ViewModel then updates a `@Published` property with the results. This manual approach means we handle refresh: for example, after inserting a new record, the ViewModel can re-fetch or simply append to its in-memory list. The **Observation framework** will still propagate these changes to the view, since the properties are published.

  In general, a mix of both can be used. For MVP, using `@Query` for something like displaying all saved insights might be perfectly fine (less code). But for something like the Health dashboard (where data might be combined from live HealthKit queries and local cache), the ViewModel approach is preferable.

* **Concurrency and MainActor:** SwiftData requires that mutation operations occur on the main thread (its contexts are tied to the UI thread by default). We mark any SwiftData access that changes data with `@MainActor` or ensure calls happen on the main queue. For example, a repository’s save functions will likely be called from the main actor context (since UI initiated), or if called from a background task, we’ll use `await MainActor.run { ... }` to perform the actual context insertion. This prevents threading issues with the persistence layer. Reading data can be done on the main thread as well (reads are usually fast). If we anticipate heavy data processing (say, migrating a lot of records), we could use background contexts (SwiftData might allow multiple ModelContext instances), but for MVP it’s acceptable to use the main context with careful asynchronous design.

* **Syncing local and remote data:** The architecture allows for a **“sync loop”** between SwiftData and remote sources (Firebase, HealthKit):

  * For **HealthKit**, since it’s read-only for Apple Health data, the app might periodically fetch new health samples (via the HealthKit service), then store summary or detail data in SwiftData for quick access and offline use. A background task (perhaps using `BackgroundTasks` or simply on app launch) could reconcile recent data. The `HealthDataRepository` would encapsulate this: e.g. a method `syncLatestHealthData()` that fetches from HealthKit (maybe only new data since last sync, timestamp stored in SwiftData) and then inserts or updates SwiftData records. SwiftUI views showing this data (via repository or @Query) will then reflect updates.
  * For **Firebase or any cloud user data**, consider a user’s AI chat history or personalized insights: the repository could upload SwiftData-stored insights to Firestore, or conversely fetch server-provided insights and store them locally. Because this might be complex, a **Use Case** can orchestrate it: e.g. `SyncInsightsUseCase` calls `insightsRepo.fetchNewRemoteInsights()` and `insightsRepo.save(insight:)` for each, etc. All such operations remain outside the view layer. SwiftData’s local store then becomes the source for the UI (with the benefit of offline access), and the remote store ensures persistence across devices.
  * We will also utilize SwiftData’s ability to mark data as **shared or private** if needed. (For example, certain ModelContainer configurations might allow iCloud sync via CloudKit in future – though that’s beyond MVP scope, the architecture is ready for it.)

* **ModelContext management best practices:** Only one instance of the main ModelContext (from our `SwiftDataManager`) will be used for UI-related operations to avoid conflicts. For any long-running background import (if any), we could create a separate ModelContext on a background thread (SwiftData likely allows an in-memory context or background context). In general, however, the **volume of data for MVP (daily health summaries, user profile, chat logs)** is not massive, so a single context is fine. We also ensure to save context at appropriate times (though SwiftData auto-saves on app background, calling `try context.save()` explicitly after critical operations (insert/delete) in the repository is a good practice to handle errors).

In summary, SwiftData is integrated via a centralized context and used through repository methods or SwiftUI queries. This approach balances SwiftData’s seamless SwiftUI integration with the need for decoupling and testability, as recommended by recent SwiftData best-practice guides.

## 4. **State Management with the Observation Framework (iOS 17) and Beyond**

Robust state management is crucial given the asynchronous, data-heavy nature of CLARITY Pulse. We employ a combination of SwiftUI’s state tools and the new iOS 17 Observation API to manage view state, handle loading/error conditions, and share data between views.

* **SwiftUI Observation API:** We will make heavy use of Swift’s new `@Observable` and `@Bindable` attributes (from the Observation framework) for our ViewModel classes. By marking a ViewModel as `@Observable`, SwiftUI can automatically subscribe to its properties without needing the traditional `@Published` on each property (though we can still use `@Published` for clarity on certain fields). This greatly reduces boilerplate. For instance:

  ```swift
  @Observable 
  class DashboardViewModel {
      var summary: DailyHealthSummary?    // Changes to this will update the view
      var isLoading: Bool = false
      var error: Error? = nil
      // ...
  }
  ```

  SwiftUI will treat these as published for us, and we can even use `@Bindable` in the view to two-way bind if needed (useful for forms or editable state). Using the Observation API keeps our state management lean and SwiftUI-idiomatic.

* **View-specific vs Global State:**

  * *View-specific state:* Each View (screen) has its own ViewModel owning the state for that screen. This state is instantiated when the view appears (using `@StateObject` to keep it alive across view redraws). For example, the **Health Dashboard screen** has a `DashboardViewModel` with properties like `dailySummary`, `weeklyTrend`, `isLoadingData`, etc. These are relevant only to that view. We use `@State` for very local transient UI state (like a toggle in that view, or the selected date in a date picker), and `@StateObject` or `@ObservedObject` for the ViewModel.
  * *Shared/global state:* Some state needs to be accessed by multiple parts of the app – for example, the current user’s profile or authentication status, or a global setting like a units preference. For such state, we utilize **environment objects or singletons**. We can create an **AppViewModel (or AppState)** as a central observable object provided at the App level. For instance, an `AppState` object (marked with `@Observable`) might hold `@Published var currentUser: User?` and `@Published var authStatus: AuthStatus`. We inject it using `@EnvironmentObject` in any view that needs it (e.g., many screens might need to know if the user is logged in or to trigger login flow if not).
  * Additionally, some *feature-specific* shared state might use smaller environment objects. For example, if multiple views need read/write access to a HealthKit data store or a common settings object, we can place an instance of that repository or a wrapper as an `@EnvironmentObject` high in the view hierarchy. iOS 17’s environment can even hold observable objects that aren’t `ObservableObject` but use the Observation API.
  * We’ll be mindful of not overusing global state (to avoid making everything singleton). The rule is: if a piece of state truly spans multiple screens or the whole app (like user login), it goes in a shared object. Otherwise keep it localized to the view or feature.

* **Loading, Error, and Empty States:** Each ViewModel will include properties to represent these states, ensuring the UI can react accordingly:

  * Typically, an **enum** can neatly capture state. For example:

    ```swift
    enum ViewStatus {
       case idle, loading, loaded, error(String), empty
    }
    ```

    and the ViewModel has `@Published var status: ViewStatus = .idle`. The SwiftUI view then switches on this to show a loading spinner, an error message, or the main content. This pattern cleanly centralizes UI state representation.
  * Alternatively, boolean flags like `isLoading: Bool` and an `errorMessage: String?` (or `Error?`) can be used. For the chat view, perhaps `isSending` for when a query is in progress. We will adopt whichever makes the UI logic simplest. For instance, the **Insights Chat screen** ViewModel might use:

    * `isLoadingResponse: Bool` – to show a typing indicator or spinner while AI is thinking.
    * `inputText: String` – two-way bound to the TextField for user query (using `@Bindable`).
    * `messages: [ChatMessage]` – the conversation so far.
    * `error: String?` – any error from the AI service.
  * SwiftUI makes it easy to conditionally show views based on these states (e.g. `if viewModel.isLoading { ProgressView() }` or overlay an error view if `error != nil`). We will create some **reusable subviews** for common states, like a full-screen “Empty State” view or an error message view, to maintain consistency across features.
  * **Observation & updates:** Because our ViewModels are observable, when they update these state properties (on the main thread), the corresponding view will re-render. For example, when `DashboardViewModel` sets `isLoading = true` at the start of data fetch, the SwiftUI view shows a loading indicator. When data comes in, `summary` gets set and `isLoading` goes false, causing the view to switch to showing content. This reactive update cycle is the heart of SwiftUI state management.

* **Complex asynchronous flows:** The app’s asynchronous operations (fetching HealthKit data, calling AI APIs, syncing with Firebase) will be handled with Swift’s `async/await` concurrency, often within the ViewModel or Use Case. To avoid blocking UI:

  * We might use `Task { … }` in the ViewModel initializers or onAppear to kick off background work. For example, in `DashboardView.onAppear`, call `viewModel.loadDashboardData()` which is an async function in the ViewModel. That function could:

    ```swift
    func loadDashboardData() {
       Task {
         do {
            await MainActor.run { self.isLoading = true }
            let summary = try await fetchDailySummaryUseCase.execute()
            await MainActor.run { 
               self.dailySummary = summary
               self.isLoading = false
            }
         } catch {
            await MainActor.run { 
               self.isLoading = false
               self.errorMessage = "Failed to load data"
            }
         }
       }
    }
    ```

    We ensure to update SwiftUI state on the main thread (using `await MainActor.run` or marking the whole function `@MainActor` if appropriate).
  * The Observation framework will detect those state changes and update the UI accordingly. This pattern cleanly handles loading and error transitions for asynchronous tasks.
  * For **AI chat**, which might involve streaming partial responses, we can integrate with Combine or AsyncSequence. For instance, if our `InsightAIService` provides an `AsyncThrowingStream` of `ChatMessage` for a query, the ViewModel can iterate over it, appending to `messages` as new chunks arrive, creating a live-updating conversation.
  * We will also handle **cancellation** of tasks if needed. SwiftUI can cancel tasks automatically when a view disappears if launched with `.task` modifiers, but for tasks in ViewModels, we might store references to them and cancel in `deinit` or when not needed (like if user navigates away mid-request). This ensures we don’t do unnecessary work or update UI that’s gone.

* **Using @State vs @StateObject vs @Binding:** We will follow best practices for state ownership:

  * Use `@State` for simple value types owned by a View (e.g. a TextField’s text in a small view).
  * Use `@StateObject` for a ViewModel, so it's created once and preserved even if the view reloads (preventing multiple initializations).
  * Use `@ObservedObject` if a parent creates the ViewModel and passes it in (less common in our design; we usually let each view create its own VM or use environment singletons).
  * Use `@Binding` to pass state down to subviews if they need to edit parent state.
  * Leverage `@EnvironmentObject` for truly global objects like AppState.
  * Leverage the new `@Environment(\.myValue)` for injecting dependencies (discussed below in DI).

* **Example – Auth State Global:** The Firebase auth status is a good example of state management:

  * We might have an `AuthViewModel` that monitors Firebase’s auth publisher (Firebase provides a Combine publisher or callback for auth state changes). This ViewModel updates an `@Published var user: User?` and `@Published var isLoggedIn: Bool`.
  * We provide this AuthViewModel (or just its state) as an `@EnvironmentObject` throughout the app. Top-level views then decide to show either the login screen or the main content based on `isLoggedIn`.
  * The login screen ViewModel will call `AuthService.login()` and handle the loading/error of that. Upon success, the global auth state changes, and SwiftUI can transition screens (perhaps using a simple `if auth.isLoggedIn { MainView() } else { LoginView() }` in the App view).
  * **Loading states** for auth (like an activity indicator on the login button) are confined to the LoginViewModel and LoginView.

By leveraging **iOS 17’s state tools**, our architecture cleanly differentiates transient view state, long-lived shared state, and loading/error UI states. This ensures each screen manages its own concerns while the app can still coordinate overall status (such as a global loading spinner if we ever need to indicate something app-wide). The Observation framework and SwiftUI’s data flow do the heavy lifting of propagating changes, resulting in a responsive UI with minimal manual UI update code.

## 5. **Navigation and Deep Linking Strategy**

Navigation in SwiftUI  - specifically using **NavigationStack** (iOS 16+ API) - will be structured to handle both in-app navigation flows and external deep links gracefully. Our approach:

* **NavigationStack & NavigationPath:** We will use a **single source of truth for navigation state** where appropriate. For most simple cases, a `NavigationStack` with programmatic navigation is achieved by binding it to a `NavigationPath` (or even a simpler `@State var [Screen]`). For example, the App’s main view might look like:

  ```swift
  @State private var navPath = NavigationPath()
  NavigationStack(path: $navPath) {
      ContentView()  // or some entry view
      .navigationDestination(for: Route.self) { route in 
          switch route {
            case .dashboard: DashboardView()
            case .insightsChat: InsightsChatView()
            // other routes...
          }
      }
  }
  ```

  Here `Route` could be an enum or struct that conforms to `Hashable` representing different navigation targets (with associated values for parameters). This **data-driven NavigationStack** means we can push new destinations by appending to `navPath`.

* **Programmatic Navigation:** In MVVM, we often let ViewModels trigger navigation by publishing some navigation event. There are a few patterns:

  * Use a **Coordinator** object that listens to navigation intents from VMs and updates the NavigationPath. For example, a `NavigationCoordinator` (could be part of AppState) that has methods like `goToDashboard()` which does `navPath.append(Route.dashboard)`. ViewModels would call these coordinator methods instead of directly manipulating navigation.
  * Alternatively, since SwiftUI allows programmatic navigation via bindings, a ViewModel could have an `@Published var navigateTo: Route?` which the View observes. When non-nil, a hidden `NavigationLink` triggers navigation. However, with NavigationStack and path, it’s cleaner to operate on the path.
  * For MVP, a simple approach is to pass a binding or closure into the ViewModel for navigation. E.g., the LoginViewModel on successful login could call a closure provided by the view that appends the next route. But an even simpler is using environment: since the Auth status is global, the view might just switch based on it (no explicit nav call needed for login -> main).
  * We will choose a pragmatic approach for each flow. A dedicated coordinator pattern might be overkill for MVP, but we ensure navigation logic is not duplicated. Critical flows (like onboarding or multi-step forms) can use a small coordinator.

* **Deep Linking Support:** Deep linking (via URLs or universal links) will be supported by translating an external URL or user activity into our internal Route and pushing it on the NavigationStack. SwiftUI makes this convenient:

  * Use the `.onOpenURL` modifier on the NavigationStack or top-level view to handle incoming URLs. We parse the URL and determine the appropriate Route. For example, an `myapp://dashboard?date=2023-12-01` might map to `Route.dashboard(date: Date)`. Once we create that Route value, we simply do `navPath.append(route)`, and the NavigationStack will navigate to the correct screen, thanks to the matching `.navigationDestination(for: Route.self)` that we set up.
  * Similarly, `.onContinueUserActivity` can handle Handoff or Siri intents if needed (like a specific Health insight view).
  * Our Route type and NavigationDestination definitions need to cover all deep-linkable screens. We will maintain a mapping in a centralized place (possibly in App or in a Router object) to avoid scattering URL handling logic.
  * **Example:** If a deep link is supposed to open the chat and ask a question, e.g., `myapp://insightChat?question=XYZ`, the app onOpenURL will decode that and perhaps directly call the AI service or set a state in the ChatViewModel with that question. Alternatively, we navigate to the chat screen and pass the question as part of the route, and in `InsightsChatView.onAppear` the ViewModel sees a preset question and handles it.

* **NavigationStack best practices:** We will prefer **NavigationLink and NavigationDestination** for in-app links (which keeps type-safe navigation). For instance, tapping a health metric in the Dashboard might navigate to a detailed trend view:

  ```swift
  NavigationLink(value: Route.trendDetail(metric: .heartRate)) {
      TrendCardView(...)
  }
  ```

  And a corresponding `navigationDestination(for: Route.self)` for `.trendDetail` presents `TrendDetailView`.
  This approach is more robust than using deprecated programmatic triggers, and it integrates well with state restoration.

* **Back and Presentation:** The NavigationStack approach inherently supports back navigation (the stack pop). For modal presentations (like a sheet for editing profile or a share sheet), we will use `.sheet` and manage the boolean state for showing it in the view or view model.

  * For example, `@Published var showingProfileEditor = false` in a view model can drive a `.sheet(isPresented: $viewModel.showingProfileEditor) { ProfileEditorView() }`.

* **Deep Link Testing:** As part of design, we’ll ensure that if the app is launched via a deep link, the NavigationStack can jump to the right screen even if intermediate views weren’t shown. This is where having a single NavigationStack at the root helps: we can set the path to contain the necessary hierarchy immediately. If some views require data (like the detail needs an ID), the deep link should provide it or we fetch it as needed.

* **State Restoration:** Although not a primary ask, our use of NavigationPath is compatible with state restoration. We can persist the NavigationPath (it’s `Codable` if our Route is codable) using `sceneStorage` or in AppState, and restore it on launch. This means the user returns to where they left off (especially nice for multi-step flows or if they had drilled into a detail from the dashboard).

**Summary:** Navigation will be declarative and data-driven. By centralizing routes, supporting programmatic navigation via state, and using NavigationStack’s features, we cover everything from simple in-app links to deep links from external sources. This approach scales to complex flows (just add new Route cases and destinations) and makes the app URL-addressable for future integration (e.g., notifications tapping into a specific insight).

## 6. **Dependency Injection (DI) Strategy**

We aim for **lightweight dependency injection** that fits SwiftUI’s paradigm, avoiding heavy libraries. The strategy combines SwiftUI’s Environment and protocol-based injections for maximum flexibility and testability:

* **Protocol-Oriented Design:** For each service or repository in the data layer, we define a protocol that describes its interface (methods/properties). For example, `protocol HealthDataRepositoryProtocol { func fetchDailyMetrics() async throws -> DailyMetrics }`. Our ViewModels and UseCases will **depend on these protocols** rather than concrete types. This abstraction is crucial for testing (we can make a mock that conforms to the protocol) and for future-proofing (could swap implementations, such as a new AI provider, without changing call sites).

* **Environment Values:** SwiftUI’s `@Environment` is a powerful way to inject dependencies into views (and thus into view models). We will extend `EnvironmentValues` with entries for our services, allowing us to provide dependencies app-wide. For example:

  ```swift
  struct HealthDataRepositoryKey: EnvironmentKey {
      static let defaultValue: HealthDataRepositoryProtocol = HealthDataRepository()  // default impl
  }
  extension EnvironmentValues {
      var healthRepo: HealthDataRepositoryProtocol {
          get { self[HealthDataRepositoryKey.self] }
          set { self[HealthDataRepositoryKey.self] = newValue }
      }
  }
  ```

  Now any view or view model can declare `@Environment(\.healthRepo) var healthRepo` to get the repository. In the SwiftUI App initializer, we can inject the real instances, e.g. `ContentView().environment(\.healthRepo, HealthDataRepository())` and similarly for others (Auth, AI, etc.). **This is a clean DI approach**: it externalizes the creation of services and makes them swappable (for example, in tests or previews we can inject a `MockHealthDataRepository` easily).

  SwiftUI environment injection is **lightweight** – it doesn’t require third-party frameworks and works naturally with SwiftUI’s view hierarchy. It also lines up with Apple’s own use (e.g., `.modelContext` is just an environment value under the hood).

* **Direct Initialization / Factory:** In some cases, we might inject dependencies via initializer parameters or factory methods instead:

  * For instance, a ViewModel might be initialized with its needed services:

    ```swift
    class InsightsChatViewModel: ObservableObject {
        init(aiService: AIServiceProtocol = Environment(\.aiService).wrappedValue) { ... }
    }
    ```

    Here we give a default that pulls from environment, so in production it auto-resolves. In tests, we can call `InsightsChatViewModel(aiService: MockAIService())` to override. Another approach is to have a central factory or use case that builds the ViewModel with proper dependencies resolved.
  * We will avoid directly instantiating singletons inside view models (like `InsightsChatViewModel()` doing `AIService()` internally) because that hides dependencies and makes testing hard. Instead, pass them in or use environment as above.

* **Singletons where appropriate:** Some subsystems might naturally be singletons (like our `SwiftDataManager.shared` or a `HealthKitManager.shared` if it needs to observe background deliveries continuously). We’ll use singletons for these low-level utilities, but still expose them via protocols if the app code interacts with them. For example, `AuthService` could be a singleton internally, but conform to `AuthServiceProtocol` and we inject that protocol. This pattern is pragmatic: it limits to one instance (since e.g. Firebase or HealthKit typically should be single), but we still abstract it for test and DI purposes.

* **Testing and Mocking:** With the above approach, writing tests is straightforward. We can construct a ViewModel with a mock repository (conforming to the protocol) that returns predictable data. The ViewModel doesn’t care that it’s a mock – it just calls the protocol methods. Similarly, for UI tests or SwiftUI previews, we can inject dummy data via environment:

  * SwiftUI **Previews**: We can do `MyView().environment(\.healthRepo, PreviewHealthRepo())` where `PreviewHealthRepo` provides static sample data. This way, our previews show realistic content without hitting real databases or APIs.
  * Unit Tests: For example, test `DashboardViewModel` by injecting a `FakeHealthDataRepository` that returns a known set of metrics, then verify the ViewModel’s `summary` is computed correctly after `loadDashboardData()`.

* **Minimal Overhead DI:** We specifically choose not to introduce heavy DI containers (like Swinject or Resolver) to keep things understandable. SwiftUI’s environment and initializer injection cover our needs for a moderate-sized app. Also, we avoid global variables; environment values provide a scoped and controlled way of passing objects.

* **Example – Injecting AI Service:**

  * Define `protocol InsightAIServiceProtocol { func getInsight(for query: String) async throws -> String }`.
  * Provide a concrete `GeminiAIService: InsightAIServiceProtocol` that calls the Gemini API.
  * Add `InsightAIServiceProtocol` to EnvironmentValues:

    ```swift
    struct AIServiceKey: EnvironmentKey {
       static let defaultValue: InsightAIServiceProtocol = GeminiAIService()
    }
    extension EnvironmentValues { var aiService: InsightAIServiceProtocol { ... } }
    ```
  * Use in a ViewModel: `@Environment(\.aiService) private var aiService`.
  * In `InsightsChatViewModel.sendQuestion()` do `let answer = try await aiService.getInsight(for: currentQuestion)`.
  * For testing, inject `environment(\.aiService, MockAIService())` where `MockAIService` returns a canned response.
  * This makes the AI integration completely swappable (e.g., if PAT model is local and we have a `LocalAIService`, we could toggle which implementation is injected via some config).

* **Dependency Graph:** We ensure that there are no cyclic dependencies and manage object lifetimes appropriately. Environment-injected services effectively behave like app-singletons (since we set them at app launch environment and they live as long as the view hierarchy). We will be careful that long-lived services (Auth, HealthKit) don’t retain view models strongly (they should use callbacks or Combine without retain cycles), to avoid memory leaks.

By using **protocols for abstraction and environment for injection**, we adhere to the principle of **inversion of control** in a Swifty way. Our views and view models are agnostic to concrete implementations, which means we can test with ease and swap modules (even adapt the app for a different backend or data source down the line) with minimal fuss.

## 7. **Reusable SwiftUI Components and Consistency**

To ensure a consistent and maintainable UI, we will build reusable SwiftUI components and follow a design system. Key strategies include:

* **Style Guide & Theming:** Establish a centralized set of UI constants and modifiers for the app’s look and feel. For example, define common color palette, font styles, and spacing in a struct or using SwiftUI’s theming:

  * Use `@Environment(\.colorScheme)` and extend EnvironmentValues with our custom `Theme` if needed, so all views can use the same fonts, corner radius, etc.
  * For example, a `PrimaryButtonStyle: ButtonStyle` can be created to give all primary buttons a consistent appearance. We then use `.buttonStyle(PrimaryButtonStyle())` everywhere for actions like "Sync" or "Login".

* **Reusable Views:** Identify UI patterns in the app that occur in multiple places and create custom views for them:

  * e.g. **Data Card View:** The Health Insights Dashboard might show cards for daily steps, heart rate, etc. We can make a generic `HealthMetricCard` view that takes parameters like title, value, unit, and maybe a trend indicator. This view can be used for any metric by passing in the data. It simplifies the Dashboard layout and ensures if we update the card style, all metrics update.
  * e.g. **Chat Message Bubble:** For the AI chat interface, define a `MessageBubbleView` that renders a message (from user or AI) in a styled bubble, handling alignment (trailing for user, leading for AI) and colors. The Chat screen just uses a `ForEach` on messages to generate a list of `MessageBubbleView(message: msg)`.
  * **Empty/Error Views:** As mentioned, create a reusable `EmptyStateView` and `ErrorStateView`. These could be simple SwiftUI Views that take in a message and maybe an image. By reusing them, we get visual consistency whenever a section is empty or an error occurs.
  * **Loading indicator overlay:** A small reusable view or view modifier that overlays a dimmed background with a ProgressView (spinner) can be made, to easily show loading over an entire screen.

* **View Composition:** SwiftUI encourages breaking UI into smaller pieces. We’ll follow the guideline that if a view body exceeds a certain number of lines or has multiple `VStack/HStack` sections, it might be a candidate to split into subviews. By composing views:

  * We avoid “massive SwiftUI views” and instead get modular, testable pieces.
  * Each subview can have its own Preview and even its own small ViewModel if needed.
  * For example, the Dashboard might consist of `DailySummaryView`, `WeeklyTrendsChart`, etc., each as a subview. Those might further use `MetricCard` subviews. This hierarchy of components ensures clarity.
  * SwiftUI has low overhead for multiple small views, so there’s little performance penalty for splitting components.

* **Consistency via Modifiers:** Create custom View modifiers for common things. If multiple screens use a specific background gradient or padding style, define a modifier like:

  ```swift
  extension View {
    func clarityBackground() -> some View {
        self.padding().background(
            LinearGradient(colors: [...], startPoint: .top, endPoint: .bottom)
        )
    }
  }
  ```

  Then use `.clarityBackground()` everywhere needed. This way, a style change is in one place. It also makes the view code more declarative and readable.

* **Accessibility & Scaling:** Reusable components also help ensure we handle accessibility uniformly. For instance, our `PrimaryButtonStyle` can automatically adjust for bold text or high contrast. We should use SwiftUI’s dynamic type and such across components. Testing components in isolation (with different content sizes in Previews) will be part of our workflow.

* **Testability of Components:** Since our components are mostly pure SwiftUI views (without internal state, or with simple state), testing them can be done via:

  * **SwiftUI Previews** (to visually verify and even use the new Xcode preview interactions).
  * **Snapshot tests** (render the view and compare to a reference image) if we integrate a testing tool for that.
  * **Unit testing view logic:** If a view has any logic (like a computed property or a custom initializer that processes data), we can unit test those portions. But largely, our logic is in view models, which we already covered testing for.

* **Multi-platform considerations:** By using SwiftUI, many of our reusable components can directly be used on iPad or Mac Catalyst. We might adopt responsive layout techniques (like using `NavigationSplitView` on iPad for a master-detail if needed). If, in the future, a watchOS app is made, we’d reuse as much domain and even UI component logic as possible (maybe the same models and formatting code, with watch-specific views). Ensuring our components are not iPhone-screen specific (e.g., avoiding fixed sizes) will ease this adaptation.

* **Example – Applying Reusable Component:** Suppose the daily summary needs a ring progress view for activity completion. We create a `RingProgressView(value: Double, max: Double)` component. This can then be used in multiple places (maybe one for daily steps, one for sleep goal, etc.) with different parameters. All ring visuals will be identical in style because they use the same component. If we decide to change the ring thickness or color scheme, we do it in `RingProgressView` and all usages update.

By designing a **component library within the app**, we ensure visual and functional consistency. This also speeds up development: adding a new screen often becomes a matter of assembling existing components rather than writing everything from scratch. In code reviews, having canonical components reduces mistakes (developers are not reinventing a button style each time, for instance). And if a component is complex (like a chart view), it’s encapsulated and can be optimized or fixed in one place. This approach aligns with SwiftUI’s strength in composition and leads to a cleaner, more maintainable codebase.

## 8. **Multiplatform Adaptability (Future-Proofing)**  *(Optional)*

While focusing on iOS 17+ for MVP, the proposed architecture naturally accommodates future expansion to watchOS or iPadOS:

* **Shared Domain & Data Layers:** The Domain logic (use cases, models) and Data services (repositories) contain no UI-specific code, so they can be moved to a **shared Swift Package** and imported into a watchOS app target or an iPadOS app. For example, the HealthKit service and SwiftData models can be reused on watchOS (note: HealthKit is available on watchOS for certain data; our service might need conditional compilation for features not on watch). The AI service, being network-based, is also platform-neutral. This means the heavy lifting of health data crunching or chat generation does not need rewriting for other platforms.
* **SwiftUI Views Reuse:** SwiftUI is cross-platform by design. Many of our Views and ViewModels could be reused on iPad or Mac with minimal changes – typically just adapting navigation. We can use conditional modifiers if needed (e.g. a navigationSplitView for iPad vs navigationStack for iPhone, but SwiftUI can often use the same code for both). For watchOS, the UI would be different (watch has very small screen and different controls), so we might build separate watch-specific views. But those views can still use the same ViewModels and observe the same state. For instance, a Watch app could have a simplified `DailySummaryView` that uses `DashboardViewModel` from the phone, communicating via Connectivity or shared cloud data.
* **Consistent Architecture:** Having a clear separation of concerns means each platform’s UI layer is the only thing that needs tailoring. The overall architecture (MVVM + use cases + repositories) can be mirrored on each platform, providing consistency in development approach. Developers working on iPad or watch will be familiar with the patterns established.
* **Platform Services:** Some services may have platform-specific implementations (e.g., watchOS might not support Firebase Auth directly, or might handle HealthKit differently). Using protocols again helps – we could have an `AuthServiceProtocol` and provide a different implementation on watch (maybe the watch just uses the phone’s auth via connectivity). The app can determine at runtime or compile-time which to use. Since this is beyond MVP, we simply note that the abstraction via protocols makes it feasible.

Overall, the architecture is designed with **future scale** in mind – both in features and platforms. By adhering to these patterns, CLARITY Pulse’s codebase will remain **scalable, maintainable, and extensible**, ready to incorporate new technologies or requirements down the road.

---

## **Summary Table: Components and Responsibilities**

| **Component / Layer**                | **Responsibilities**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SwiftUI View (UI Layer)**          | - Declares the UI layout and appearance using SwiftUI.<br/>- Binds to ViewModel state (using `@StateObject` or `@ObservedObject`) and updates automatically on changes.<br/>- Forwards user input (button taps, etc.) to the ViewModel via closure or @Bindable actions.<br/>- **No business logic** beyond minor view-only formatting.                                                                                                                                                                                                                                                                                                |
| **ViewModel (Presentation)**         | - Acts as the **mediator between View and Domain**.<br/>- Holds UI state (published properties such as data to display, loading flags, error messages).<br/>- Executes async tasks and calls Use Cases / Services to fetch or modify data.<br/>- Transforms raw data into view-ready format (e.g., combine multiple repo results, format strings).<br/>- Notifies the view of state changes via SwiftUI’s observation (Observation API / ObservableObject).                                                                                                                                                                            |
| **Use Case / Interactor (Domain)**   | - Encapsulates a **specific unit of business logic** or workflow (e.g., “Fetch daily health summary”, “Generate AI insight”).<br/>- Orchestrates calls to one or multiple repositories/services. Contains decision-making logic (e.g., if data is stale, fetch remote; combine health + AI data if needed for an insight).<br/>- Returns results in terms of Domain Models to ViewModel, or updates repositories. Ensures complex logic is tested once here rather than in every ViewModel.                                                                                                                                            |
| **Domain Models**                    | - Pure Swift structs or simple classes representing core app data (e.g., `DailyHealthSummary`, `InsightMessage`).<br/>- Used across app layers to avoid passing around low-level data formats (like HealthKit types or raw JSON).<br/>- May include computed properties or simple validation, but no heavy dependencies.                                                                                                                                                                                                                                                                                                               |
| **Repository / Service (Data)**      | - **Abstract data source**; each focuses on one area: e.g., HealthKit, Firebase, AI API, Local SwiftData.<br/>- Implements a protocol interface (e.g., `HealthDataRepositoryProtocol`) to allow substitution and testing.<br/>- Performs the actual data operations: network calls, database fetches, HealthKit queries, etc., possibly using framework SDKs (HealthKit, URLSession, Firebase SDK, SwiftData).<br/>- May cache or sync data (e.g., store fetched health data into SwiftData). Ensures on completion to call back or return data for use cases to consume. All heavy I/O or framework interaction is concentrated here. |
| **Persistence Layer (SwiftData)**    | - Stores app data locally in a persistence store (using SwiftData’s ModelContainer and ModelContext).<br/>- `@Model` entities represent tables. SwiftDataManager sets up the container and context at app launch and handles context life cycle.<br/>- Repositories use this to save or load data. Also provides change notifications (through `@Query` or manual fetch) to update the UI with latest persisted data.                                                                                                                                                                                                                  |
| **App State / Environment (Global)** | - Singleton or environment-driven state available app-wide (e.g., current user session, global settings, NavigationPath).<br/>- Provided as `@EnvironmentObject` or Environment values, so multiple views can read/update if necessary.<br/>- Helps coordinate things like global navigation (deep link handling), theme, or user login status across the app.                                                                                                                                                                                                                                                                         |
| **Dependency Injection**             | - *Not a layer, but a cross-cutting concern:* Uses SwiftUI Environment and protocols for injecting services into ViewModels and Views.<br/>- Ensures loose coupling between components – e.g., ViewModel relies on an `AuthServiceProtocol`, and we inject a concrete `FirebaseAuthService` in production or a mock in tests.<br/>- Provides flexibility to swap implementations (useful for testing and future changes, like switching persistence or AI provider).                                                                                                                                                                   |
| **Reusable UI Components**           | - Custom SwiftUI views (or view modifiers) used in multiple places (buttons, cards, charts, etc.).<br/>- Encapsulate a specific UI pattern for consistency (following design system).<br/>- Improve development speed and ensure that styling and behavior remain uniform across the app. Each is typically stateless or internally manages its small state, taking inputs via init parameters or bindings.                                                                                                                                                                                                                            |

Each of these pieces works together in the proposed architecture. The **flow** for a given feature, say “display today’s health stats and an AI insight”, would be:

1. The **View** appears, its ViewModel is created and perhaps calls a **UseCase** like `GetTodayDashboardData`.
2. That UseCase asks the **HealthDataRepository** for health metrics and the **AIService** for an insight. The repositories fetch from **HealthKit** and **SwiftData** (for cached data) and from the AI API (network) respectively.
3. The UseCase combines results into a **Domain model** (e.g., a `DashboardData` struct containing a summary and a tip).
4. The ViewModel gets this and updates its **state** (published properties).
5. SwiftUI observes the changes via the **Observation** system and re-renders the **View**, which displays the new data (or error state if something failed).
6. If the user triggers an action (e.g., refresh or asks a new question), the ViewModel handles it (possibly invoking another UseCase or repository call), and the cycle continues.

This architecture ensures each feature is built in a consistent, modular way, addressing the MVP requirements and setting a strong foundation for the app’s growth. It embraces SwiftUI and SwiftData’s modern capabilities while enforcing separation of concerns for long-term **scalability, maintainability, and testability**.

