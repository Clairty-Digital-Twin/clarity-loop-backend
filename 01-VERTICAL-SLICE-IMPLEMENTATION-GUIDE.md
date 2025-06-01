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
│ API Gateway (FastAPI + Firebase Auth)                  │
├─────────────────────────────────────────────────────────┤
│ Business Logic (AsyncIO + Pydantic)                    │
├─────────────────────────────────────────────────────────┤
│ ML Processing (PyTorch + Gemini 2.5)                   │
├─────────────────────────────────────────────────────────┤
│ Storage Layer (Firestore + Cloud Storage)              │
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
- [ ] Create `src/ml/gemini_client.py` - AI integration
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
    
    # Generate insights with Gemini
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
