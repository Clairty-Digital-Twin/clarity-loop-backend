# 🗄️ SwiftData Backend Wrapper Guide

*Transform your backend into a gorgeous local-first experience with SwiftData*

## 🎯 Local-First Principles

- **📱 Local storage is primary** - App works offline
- **🔄 Sync is secondary** - Background sync when online  
- **⚡ Immediate response** - No loading spinners
- **🧠 Smart conflict resolution** - Handle data conflicts gracefully

---

## 📊 SwiftData Models

```swift
import SwiftData

@Model
final class HealthMetric {
    @Attribute(.unique) var id: UUID
    @Attribute(.indexed) var metricType: MetricType
    var value: Double
    var unit: String
    @Attribute(.indexed) var timestamp: Date
    @Attribute(.indexed) var userId: String
    
    // Sync tracking
    var syncStatus: SyncStatus = .pending
    var serverVersion: Int = 0
    var localVersion: Int = 1
    
    // Relationships
    @Relationship(deleteRule: .cascade) 
    var insights: [HealthInsight] = []
    
    init(type: MetricType, value: Double, unit: String, userId: String) {
        self.id = UUID()
        self.metricType = type
        self.value = value
        self.unit = unit
        self.userId = userId
        self.timestamp = Date()
    }
}

@Model
final class HealthInsight {
    @Attribute(.unique) var id: UUID
    var title: String
    var content: String
    var confidence: Double
    @Attribute(.indexed) var category: InsightCategory
    @Attribute(.indexed) var generatedDate: Date
    
    // Sync tracking
    var syncStatus: SyncStatus = .pending
    
    @Relationship(inverse: \HealthMetric.insights)
    var healthMetric: HealthMetric?
    
    init(title: String, content: String, confidence: Double, category: InsightCategory) {
        self.id = UUID()
        self.title = title
        self.content = content
        self.confidence = confidence
        self.category = category
        self.generatedDate = Date()
    }
}

enum MetricType: String, Codable, CaseIterable {
    case heartRate = "HEART_RATE"
    case heartRateVariability = "HEART_RATE_VARIABILITY"
    case sleepAnalysis = "SLEEP_ANALYSIS"
    case activityLevel = "ACTIVITY_LEVEL"
    case stressIndicators = "STRESS_INDICATORS"
    
    var displayName: String {
        switch self {
        case .heartRate: return "Heart Rate"
        case .heartRateVariability: return "HRV"
        case .sleepAnalysis: return "Sleep"
        case .activityLevel: return "Activity"
        case .stressIndicators: return "Stress"
        }
    }
    
    var iconName: String {
        switch self {
        case .heartRate: return "heart.fill"
        case .heartRateVariability: return "waveform.path.ecg"
        case .sleepAnalysis: return "bed.double.fill"
        case .activityLevel: return "figure.walk"
        case .stressIndicators: return "brain.head.profile"
        }
    }
}

enum SyncStatus: String, Codable {
    case pending = "pending"
    case syncing = "syncing"
    case synced = "synced"
    case failed = "failed"
    case conflict = "conflict"
}
```

---

## 🔄 Repository Pattern

```swift
@Observable
final class HealthRepository {
    private let modelContext: ModelContext
    private let syncService: HealthSyncService
    private let apiClient: ClarityAPIClient
    
    var isLoading = false
    var isSyncing = false
    var syncError: Error?
    
    init(modelContext: ModelContext) {
        self.modelContext = modelContext
        self.apiClient = ClarityAPIClient()
        self.syncService = HealthSyncService(
            modelContext: modelContext,
            apiClient: apiClient
        )
    }
    
    // Local-first operations
    func fetchMetrics(for type: MetricType? = nil) -> [HealthMetric] {
        var descriptor = FetchDescriptor<HealthMetric>(
            sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
        )
        
        if let type {
            descriptor.predicate = #Predicate { $0.metricType == type }
        }
        
        return (try? modelContext.fetch(descriptor)) ?? []
    }
    
    func addMetric(_ metric: HealthMetric) async throws {
        // Store locally first
        modelContext.insert(metric)
        try modelContext.save()
        
        // Queue for background sync
        await syncService.queueForUpload(metric)
    }
    
    func syncWithBackend() async throws {
        guard !isSyncing else { return }
        
        isSyncing = true
        defer { isSyncing = false }
        
        try await syncService.performFullSync()
    }
}
```

---

## 🔄 Sync Service

```swift
@Observable
final class HealthSyncService {
    private let modelContext: ModelContext
    private let apiClient: ClarityAPIClient
    
    init(modelContext: ModelContext, apiClient: ClarityAPIClient) {
        self.modelContext = modelContext
        self.apiClient = apiClient
    }
    
    func performFullSync() async throws {
        // Upload pending changes
        await uploadPendingMetrics()
        
        // Download new data
        await downloadLatestData()
        
        // Resolve conflicts
        await resolveConflicts()
    }
    
    func queueForUpload(_ metric: HealthMetric) async {
        metric.syncStatus = .pending
        try? modelContext.save()
        
        // Process immediately if online
        if NetworkMonitor.shared.isConnected {
            await uploadMetric(metric)
        }
    }
    
    private func uploadMetric(_ metric: HealthMetric) async {
        do {
            metric.syncStatus = .syncing
            try modelContext.save()
            
            let response = try await apiClient.uploadHealthMetric(metric.toDTO())
            
            metric.syncStatus = .synced
            metric.serverVersion = response.version
            try modelContext.save()
            
        } catch {
            metric.syncStatus = .failed
            try? modelContext.save()
        }
    }
    
    private func downloadLatestData() async {
        do {
            let latestMetrics = try await apiClient.fetchLatestMetrics()
            
            for serverMetric in latestMetrics {
                let existingMetric = findLocalMetric(id: serverMetric.id)
                
                if let existing = existingMetric {
                    if existing.localVersion > existing.serverVersion {
                        // Conflict detected
                        existing.syncStatus = .conflict
                    } else {
                        // Update from server
                        existing.updateFromServer(serverMetric)
                        existing.syncStatus = .synced
                    }
                } else {
                    // New metric from server
                    let newMetric = HealthMetric.fromServer(serverMetric)
                    newMetric.syncStatus = .synced
                    modelContext.insert(newMetric)
                }
            }
            
            try modelContext.save()
        } catch {
            print("Download failed: \(error)")
        }
    }
}
```

---

## 🔌 API Client

```swift
final class ClarityAPIClient {
    private let baseURL = URL(string: "http://localhost:8080/api/v1")!
    private let session = URLSession.shared
    
    func uploadHealthMetric(_ metric: HealthMetricDTO) async throws -> HealthMetricResponse {
        let url = baseURL.appendingPathComponent("health-data/upload")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(AuthManager.shared.token)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONEncoder().encode(metric)
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw APIError.uploadFailed
        }
        
        return try JSONDecoder().decode(HealthMetricResponse.self, from: data)
    }
    
    func fetchLatestMetrics() async throws -> [HealthMetricDTO] {
        let url = baseURL.appendingPathComponent("health-data")
        var request = URLRequest(url: url)
        request.setValue("Bearer \(AuthManager.shared.token)", forHTTPHeaderField: "Authorization")
        
        let (data, _) = try await session.data(for: request)
        
        let response = try JSONDecoder().decode(HealthDataResponse.self, from: data)
        return response.data
    }
}

struct HealthMetricDTO: Codable {
    let id: UUID
    let metricType: MetricType
    let value: Double
    let unit: String
    let timestamp: Date
    let userId: String
    let version: Int
}

struct HealthMetricResponse: Codable {
    let id: UUID
    let version: Int
    let status: String
}

struct HealthDataResponse: Codable {
    let data: [HealthMetricDTO]
}

enum APIError: Error {
    case uploadFailed
    case downloadFailed
    case unauthorized
}
```

---

## 📱 SwiftUI Integration

```swift
struct HealthDashboardView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(HealthRepository.self) private var repository
    
    @Query(
        filter: #Predicate<HealthMetric> { metric in
            metric.timestamp >= Calendar.current.startOfDay(for: Date())
        },
        sort: \HealthMetric.timestamp,
        order: .reverse
    ) var todaysMetrics: [HealthMetric]
    
    @Query(
        filter: #Predicate<HealthMetric> { metric in
            metric.syncStatus != .synced
        }
    ) var unsyncedMetrics: [HealthMetric]
    
    var body: some View {
        NavigationStack {
            List {
                if !unsyncedMetrics.isEmpty {
                    Section("Sync Status") {
                        SyncStatusRow(
                            pendingCount: unsyncedMetrics.count,
                            isLoading: repository.isSyncing
                        ) {
                            Task {
                                try? await repository.syncWithBackend()
                            }
                        }
                    }
                }
                
                ForEach(MetricType.allCases, id: \.self) { type in
                    let typeMetrics = todaysMetrics.filter { $0.metricType == type }
                    
                    if !typeMetrics.isEmpty {
                        Section(type.displayName) {
                            ForEach(typeMetrics) { metric in
                                MetricRowView(metric: metric)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Health Data")
            .refreshable {
                try? await repository.syncWithBackend()
            }
        }
    }
}

struct MetricRowView: View {
    let metric: HealthMetric
    
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("\(metric.value, specifier: "%.1f") \(metric.unit)")
                    .font(.headline)
                
                Text(metric.timestamp, style: .relative)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            Spacer()
            
            SyncStatusIndicator(status: metric.syncStatus)
        }
    }
}

struct SyncStatusIndicator: View {
    let status: SyncStatus
    
    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(status.color)
                .frame(width: 8, height: 8)
            
            Text(status.rawValue.capitalized)
                .font(.caption2)
        }
    }
}

extension SyncStatus {
    var color: Color {
        switch self {
        case .pending: return .orange
        case .syncing: return .blue
        case .synced: return .green
        case .failed: return .red
        case .conflict: return .purple
        }
    }
}
```

---

## 🚀 App Setup

```swift
@main
struct ClarityLoopApp: App {
    let modelContainer: ModelContainer
    
    init() {
        do {
            modelContainer = try ModelContainer(
                for: HealthMetric.self, HealthInsight.self
            )
        } catch {
            fatalError("Failed to create ModelContainer: \(error)")
        }
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .modelContainer(modelContainer)
                .environment(HealthRepository(
                    modelContext: modelContainer.mainContext
                ))
        }
    }
}
```

---

## ✅ Implementation Steps

1. **Setup Models**: Create SwiftData models with sync tracking
2. **Build Repository**: Implement local-first data access
3. **Add Sync Service**: Handle background synchronization
4. **Create API Client**: Interface with Clarity Loop backend
5. **Build UI**: SwiftUI views with @Query
6. **Test Offline**: Verify app works without network

---

*Your backend is now perfectly wrapped with SwiftData! 🎉* 