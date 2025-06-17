# 🚀 Swift Frontend Integration Guide for Clarity Loop Backend

*The Ultimate Guide to Modern SwiftUI & SwiftData Patterns*

## 🎯 Executive Summary

Transform your Clarity Loop Backend into a gorgeous SwiftUI frontend using:
- **SwiftUI 5.0 + iOS 17** patterns  
- **SwiftData local-first** architecture
- **Apple Intelligence** integration
- **Real-time WebSocket** connections

---

## 📊 Backend API Overview

Your backend provides these endpoints:

### Authentication
```
POST /api/v1/auth/register
POST /api/v1/auth/login  
GET  /api/v1/auth/me
```

### Health Data
```
POST /api/v1/health-data
GET  /api/v1/health-data/
```

### AI Analysis  
```
POST /api/v1/pat/step-analysis
POST /api/v1/insights
```

### WebSocket
```
ws://localhost:8080/api/v1/ws/health-analysis/{user_id}
```

---

## 🏗️ Modern SwiftUI Architecture

### State Management (iOS 17+)

```swift
// ✅ NEW: Observable Pattern
@Observable 
class HealthViewModel {
    var healthData: [HealthMetric] = []
    var isLoading = false
    var insights: [Insight] = []
}

struct HealthView: View {
    @State private var viewModel = HealthViewModel()
}
```

### SwiftData Models

```swift
import SwiftData

@Model
final class HealthMetric {
    @Attribute(.unique) var id: UUID
    var metricType: MetricType
    var value: Double
    var timestamp: Date
    var syncStatus: SyncStatus = .pending
    
    @Relationship(deleteRule: .cascade) 
    var insights: [HealthInsight]?
    
    init(type: MetricType, value: Double) {
        self.id = UUID()
        self.metricType = type
        self.value = value
        self.timestamp = Date()
    }
}

@Model
final class HealthInsight {
    @Attribute(.unique) var id: UUID
    var title: String
    var content: String
    var confidence: Double
    var category: InsightCategory
}
```

---

## 🎨 Beautiful UI Components

### Dashboard View

```swift
struct HealthDashboardView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \HealthMetric.timestamp, order: .reverse) 
    var metrics: [HealthMetric]
    
    @State private var viewModel = HealthDashboardViewModel()
    
    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVStack(spacing: 20) {
                    MetricsOverviewSection(metrics: todaysMetrics)
                    HealthChartsSection(metrics: weeklyMetrics)
                    InsightsSection(insights: viewModel.insights)
                }
                .padding()
            }
            .navigationTitle("Health Dashboard")
        }
        .task {
            await viewModel.loadData()
        }
    }
}
```

### Metric Cards

```swift
struct MetricCard: View {
    let metric: HealthMetric
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: metric.metricType.iconName)
                    .foregroundStyle(metric.metricType.color)
                
                Spacer()
                
                SyncStatusIndicator(status: metric.syncStatus)
            }
            
            VStack(alignment: .leading) {
                Text(metric.formattedValue)
                    .font(.largeTitle.bold())
                
                Text(metric.metricType.displayName)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}
```

---

## 🔄 Repository Pattern

```swift
@Observable
final class HealthRepository {
    private let modelContext: ModelContext
    private let apiClient: ClarityAPIClient
    private let syncService: HealthSyncService
    
    var isLoading = false
    var syncError: Error?
    
    init(modelContext: ModelContext) {
        self.modelContext = modelContext
        self.apiClient = ClarityAPIClient()
        self.syncService = HealthSyncService(
            apiClient: apiClient,
            modelContext: modelContext
        )
    }
    
    func fetchMetrics(for type: MetricType? = nil) -> [HealthMetric] {
        var descriptor = FetchDescriptor<HealthMetric>(
            sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
        )
        
        if let type {
            descriptor.predicate = #Predicate { $0.metricType == type }
        }
        
        return (try? modelContext.fetch(descriptor)) ?? []
    }
    
    func addMetric(_ metric: HealthMetric) async {
        modelContext.insert(metric)
        try? modelContext.save()
        
        await syncService.queueForUpload(metric)
    }
    
    func syncWithBackend() async {
        isLoading = true
        defer { isLoading = false }
        
        await syncService.performFullSync()
    }
}
```

---

## 🤖 Apple Intelligence Integration

```swift
@Observable
final class HealthInsightsService {
    private let repository: HealthRepository
    
    var insights: [HealthInsight] = []
    var isGenerating = false
    
    func generateInsights() async {
        isGenerating = true
        defer { isGenerating = false }
        
        let recentMetrics = repository.fetchMetrics()
        let insights = await analyzeWithAppleIntelligence(recentMetrics)
        
        await MainActor.run {
            self.insights = insights
        }
    }
    
    private func analyzeWithAppleIntelligence(_ metrics: [HealthMetric]) async -> [HealthInsight] {
        // On-device AI analysis
        var insights: [HealthInsight] = []
        
        // Heart rate analysis
        let heartRateMetrics = metrics.filter { $0.metricType == .heartRate }
        if let insight = await analyzeHeartRate(heartRateMetrics) {
            insights.append(insight)
        }
        
        return insights
    }
}
```

---

## 🌊 WebSocket Integration

```swift
@Observable
final class HealthWebSocketService {
    private var webSocketTask: URLSessionWebSocketTask?
    private let userId: String
    private let repository: HealthRepository
    
    var isConnected = false
    var lastMessage: String?
    
    init(userId: String, repository: HealthRepository) {
        self.userId = userId
        self.repository = repository
    }
    
    func connect() {
        let url = URL(string: "ws://localhost:8080/api/v1/ws/health-analysis/\(userId)")!
        var request = URLRequest(url: url)
        request.setValue("Bearer \(AuthManager.shared.token)", forHTTPHeaderField: "Authorization")
        
        webSocketTask = URLSession.shared.webSocketTask(with: request)
        webSocketTask?.resume()
        
        isConnected = true
        receiveMessage()
    }
    
    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                self?.handleMessage(message)
                self?.receiveMessage()
            case .failure(let error):
                print("WebSocket error: \(error)")
                self?.isConnected = false
            }
        }
    }
    
    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        if case .string(let text) = message,
           let data = text.data(using: .utf8),
           let update = try? JSONDecoder().decode(HealthUpdate.self, from: data) {
            
            Task { @MainActor in
                await processUpdate(update)
            }
        }
    }
}
```

---

## 🚀 Complete App Structure

```swift
@main
struct ClarityLoopApp: App {
    let modelContainer: ModelContainer
    
    init() {
        do {
            modelContainer = try ModelContainer(for: HealthMetric.self, HealthInsight.self)
        } catch {
            fatalError("Failed to create ModelContainer: \(error)")
        }
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .modelContainer(modelContainer)
        }
    }
}

struct ContentView: View {
    @Environment(AuthManager.self) private var authManager
    
    var body: some View {
        Group {
            if authManager.isAuthenticated {
                MainTabView()
            } else {
                AuthenticationView()
            }
        }
    }
}
```

---

## ✅ Getting Started

1. **Setup Project**: iOS 17+, SwiftUI
2. **Add Models**: Copy SwiftData models
3. **Implement Repository**: Local-first pattern
4. **Build UI**: Dashboard + Charts
5. **Add WebSocket**: Real-time updates
6. **Test**: Offline + Sync behavior

---

*Your backend is now wrapped in gorgeous SwiftUI! 🎉* 