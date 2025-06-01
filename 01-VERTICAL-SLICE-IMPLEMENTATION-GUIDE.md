# Vertical Slice Implementation Guide

**Canonical reference for implementing complete user features from API to storage**

## Core Philosophy

**Vertical Slice = Complete User Journey**: Each slice delivers end-to-end functionality that users can actually experience, from iOS app interaction through backend processing to data storage and insights generation.

## Vertical Slice Structure

### Slice Anatomy
```
┌─────────────────────────────────────────────────────────┐
│ iOS/watchOS Client (SwiftUI + HealthKit)                │
├─────────────────────────────────────────────────────────┤
│ API Gateway (FastAPI + Firebase Auth)                   │
├─────────────────────────────────────────────────────────┤
│ Business Logic (AsyncIO + Pydantic)                     │
├─────────────────────────────────────────────────────────┤
│ ML Processing (PyTorch + Gemini 2.5 Pro)                │
├─────────────────────────────────────────────────────────┤
│ Storage Layer (Firestore + Cloud Storage)               │
└─────────────────────────────────────────────────────────┘
```

## Implementation Order

### Phase 1: Health Data Upload & Storage
**Goal**: User can sync Apple Watch data and see it stored

#### Checklist:
- [ ] Create `src/api/v1/health_data.py` - Upload endpoint
- [ ] Create `src/models/health_data.py` - Pydantic models
- [ ] Create `src/services/health_data_service.py` - Business logic
- [ ] Create `src/storage/firestore_client.py` - Database operations
- [ ] Create `tests/integration/test_health_data_flow.py` - E2E test
- [ ] Verify: POST /api/v1/health-data/upload returns 202 with processing_id

#### Files Created:
```
src/
├── api/v1/health_data.py          # FastAPI endpoints
├── models/health_data.py          # Pydantic validation
├── services/health_data_service.py # Business logic
├── storage/firestore_client.py    # Database layer
└── tests/integration/test_health_data_flow.py
```

#### Implementation Pattern:
```python
# 1. Define the API contract first (what users can do)
@router.post("/health-data/upload", response_model=UploadResponse)
async def upload_health_data(
    data: HealthDataUpload,
    current_user: User = Depends(get_current_user)
):
    processing_id = await health_data_service.process_upload(data, current_user.id)
    return UploadResponse(processing_id=processing_id, status="accepted")

# 2. Create Pydantic models for validation
class HealthDataUpload(BaseModel):
    data_type: Literal["heart_rate", "steps", "sleep", "activity"]
    values: List[HealthDataPoint]
    source: str = "apple_watch"
    timestamp: datetime
    
    class Config:
        extra = "forbid"

# 3. Implement business logic
async def process_upload(data: HealthDataUpload, user_id: str) -> str:
    # Validate data quality
    quality_score = await assess_data_quality(data.values)
    
    # Store raw data
    doc_ref = await firestore_client.store_health_data(user_id, data)
    
    # Trigger async processing
    await pubsub_client.publish("health-data-processing", {
        "user_id": user_id,
        "data_type": data.data_type,
        "document_id": doc_ref.id
    })
    
    return doc_ref.id

# 4. Test the complete flow
async def test_health_data_upload_flow():
    # Upload data
    response = await client.post("/api/v1/health-data/upload", 
        json=sample_heart_rate_data)
    
    # Verify immediate response
    assert response.status_code == 202
    processing_id = response.json()["processing_id"]
    
    # Verify data stored
    stored_data = await firestore_client.get_document(processing_id)
    assert stored_data["data_type"] == "heart_rate"
```

### Phase 2: AI Insights Generation
**Goal**: Uploaded data triggers AI analysis and generates insights

#### Checklist:
- [ ] Create `src/ml/gemini_client.py` - AI integration (Gemini 2.5 Pro) [See Vertex AI documentation](https://cloud.google.com/vertex-ai/docs)
- [ ] Create `src/services/insights_service.py` - Insight generation
- [ ] Create `src/api/v1/insights.py` - Insights endpoints
- [ ] Create `src/models/insights.py` - Insight models
- [ ] Create background processing with Pub/Sub
- [ ] Verify: GET /api/v1/insights/daily returns AI-generated analysis

#### Implementation Pattern:
```python
# Background processor triggered by Pub/Sub
async def process_health_data_insights(message: dict):
    user_id = message["user_id"]
    data_type = message["data_type"]
    
    # Retrieve user's recent data
    health_data = await get_user_health_data(user_id, days=7)
    
    # Generate insights with Gemini 2.5 Pro
    insights = await gemini_client.generate_health_insights(
        data=health_data,
        user_context=await get_user_context(user_id)
    )
    
    # Store insights
    await store_insights(user_id, insights)
    
    # Notify user if significant findings
    if insights.significance_score > 0.8:
        await notification_service.send_insight_notification(user_id, insights)
```

### Phase 3: Real-time Chat Interface
**Goal**: User can chat with AI about their health data

#### Checklist:
- [ ] Create `src/api/v1/chat.py` - WebSocket endpoints
- [ ] Create `src/services/chat_service.py` - Conversation logic
- [ ] Create `src/ml/conversation_context.py` - Context management
- [ ] Implement real-time streaming responses
- [ ] Verify: WebSocket connection enables health data Q&A

#### Implementation Pattern:
```python
@router.websocket("/chat/{user_id}")
async def chat_websocket(websocket: WebSocket, user_id: str):
    await websocket.accept()
    conversation_context = ConversationContext(user_id)
    
    try:
        while True:
            # Receive user message
            message = await websocket.receive_text()
            
            # Get relevant health data context
            health_context = await get_relevant_health_data(
                user_id, message, days=30
            )
            
            # Generate streaming response
            async for response_chunk in gemini_client.stream_chat_response(
                message=message,
                health_context=health_context,
                conversation_history=conversation_context.history
            ):
                await websocket.send_text(response_chunk)
            
            # Update conversation context
            conversation_context.add_exchange(message, response_chunk)
            
    except WebSocketDisconnect:
        await conversation_context.save()
```

### Phase 4: Actigraphy ML Pipeline
**Goal**: Advanced sleep/activity analysis using PAT models

#### Checklist:
- [ ] Create `src/ml/actigraphy_transformer.py` - PAT integration
- [ ] Create `src/ml/preprocessing.py` - Data normalization
- [ ] Create `src/services/actigraphy_service.py` - ML orchestration
- [ ] Implement z-score normalization pipeline
- [ ] Verify: Sleep stage classification and circadian analysis

#### Implementation Pattern:
```python
class ActigraphyProcessor:
    def __init__(self):
        self.pat_model = ActigraphyTransformer.load_pretrained()
        self.preprocessor = HealthDataPreprocessor()
    
    async def process_sleep_analysis(self, user_id: str, days: int = 7):
        # Get raw actigraphy data
        raw_data = await get_actigraphy_data(user_id, days)
        
        # Apply PAT preprocessing (per-year z-scaling)
        processed_data = self.preprocessor.apply_pat_preprocessing(
            raw_data, 
            user_baseline=await get_user_baseline(user_id)
        )
        
        # Run PAT model inference
        with torch.no_grad():
            sleep_stages = self.pat_model.predict_sleep_stages(processed_data)
            circadian_features = self.pat_model.extract_circadian_features(processed_data)
        
        # Generate clinical insights
        insights = await self.generate_sleep_insights(
            sleep_stages, circadian_features, user_id
        )
        
        return {
            "sleep_stages": sleep_stages.tolist(),
            "circadian_rhythm": circadian_features,
            "clinical_insights": insights,
            "depression_risk": self.assess_depression_risk(sleep_stages)
        }
```

## Vertical Slice Development Rules

### 1. Always Start with API Contract
Define the user interaction first, then build backward:
```python
# This is what users can do - define this FIRST
@router.post("/health-data/upload", response_model=UploadResponse)
async def upload_health_data(
    data: HealthDataUpload,
    current_user: User = Depends(get_current_user)
):
    # Implementation comes after contract is clear
    pass
```

### 2. Build Backwards from User Value
- Start with the endpoint that delivers user value
- Work backward through business logic → data models → storage
- Always maintain the connection to user experience

### 3. Make It Work, Then Make It Right
- Get the simplest possible implementation working first
- Add error handling, validation, and optimization incrementally
- Each commit should maintain a working vertical slice

### 4. Test the Complete Flow
Every vertical slice must have an integration test that verifies the complete user journey:
```python
async def test_complete_health_data_journey():
    # 1. User uploads data
    upload_response = await client.post("/api/v1/health-data/upload", 
        json=sample_data)
    processing_id = upload_response.json()["processing_id"]
    
    # 2. System processes data
    await asyncio.sleep(2)  # Wait for async processing
    
    # 3. User gets insights
    insights = await client.get(f"/api/v1/insights/daily?processing_id={processing_id}")
    assert insights.status_code == 200
    assert "recommendations" in insights.json()
    
    # 4. User can chat about data
    async with client.websocket_connect(f"/chat/{user_id}") as websocket:
        await websocket.send_text("What does my heart rate data show?")
        response = await websocket.receive_text()
        assert "heart rate" in response.lower()
```

## File Organization per Slice

### Standard Structure:
```
src/
├── api/v1/              # FastAPI routers
│   ├── __init__.py
│   ├── health_data.py   # Health data endpoints
│   ├── insights.py      # Insights endpoints
│   ├── chat.py          # Chat/WebSocket endpoints
│   └── auth.py          # Authentication endpoints
├── models/              # Pydantic models
│   ├── __init__.py
│   ├── health_data.py   # Health data models
│   ├── insights.py      # Insight models
│   ├── chat.py          # Chat models
│   └── user.py          # User models
├── services/            # Business logic
│   ├── __init__.py
│   ├── health_data_service.py
│   ├── insights_service.py
│   ├── chat_service.py
│   └── auth_service.py
├── ml/                  # AI/ML components
│   ├── __init__.py
│   ├── gemini_client.py         # Vertex AI Gemini 2.5 Pro client [See Vertex AI documentation](https://cloud.google.com/vertex-ai/docs)
│   ├── actigraphy_transformer.py
│   ├── preprocessing.py
│   └── conversation_context.py
├── storage/             # Database/storage
│   ├── __init__.py
│   ├── firestore_client.py
│   ├── cloud_storage.py
│   └── repositories/
└── config/              # Configuration
    ├── __init__.py
    ├── settings.py
    └── dependencies.py
```

### Slice-Specific Files:
Each vertical slice creates files in ALL layers:
- `api/v1/{feature}.py` - User interface
- `models/{feature}.py` - Data validation
- `services/{feature}_service.py` - Business logic
- `storage/{feature}_repository.py` - Data persistence
- `tests/integration/test_{feature}_flow.py` - E2E verification

## Quality Gates per Slice

### 1. Functionality Gate
- [ ] API endpoint responds correctly
- [ ] Data is validated and stored
- [ ] Business logic executes without errors
- [ ] Integration test passes

### 2. Performance Gate
- [ ] Response time < 2 seconds for sync operations
- [ ] Async operations complete < 30 seconds
- [ ] Memory usage stays within bounds
- [ ] Database queries are optimized

### 3. Security Gate
- [ ] Authentication required for protected endpoints
- [ ] Input validation prevents injection attacks
- [ ] HIPAA-compliant data handling
- [ ] Audit logging implemented

### 4. Observability Gate
- [ ] Comprehensive logging at all levels
- [ ] Metrics collection for monitoring
- [ ] Error tracking and alerting
- [ ] Health checks implemented

## Next Steps

1. **Choose Your First Slice**: Start with health data upload (most foundational)
2. **Create the API Contract**: Define exactly what users can do
3. **Build Backwards**: Implement storage → logic → validation → endpoint
4. **Test End-to-End**: Verify the complete user journey works
5. **Deploy and Validate**: Get it running in the real environment
6. **Move to Next Slice**: Apply same pattern to insights generation

## Common Pitfalls to Avoid

### 1. Horizontal Layer Development
❌ **Don't do this**: Build all APIs, then all services, then all storage
✅ **Do this**: Build one complete user journey at a time

### 2. Over-Engineering Early
❌ **Don't do this**: Design perfect abstractions before understanding requirements
✅ **Do this**: Get basic functionality working, then refactor

### 3. Skipping Integration Tests
❌ **Don't do this**: Only test individual components
✅ **Do this**: Test complete user journeys end-to-end

### 4. Ignoring Real Data
❌ **Don't do this**: Only test with perfect mock data
✅ **Do this**: Test with realistic, messy health data from the start
