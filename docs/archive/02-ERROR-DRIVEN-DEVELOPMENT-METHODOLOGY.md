# Error-Driven Development Methodology

**Canonical approach for aggressive, fast implementation that learns from failures**

## Core Philosophy

**Embrace Errors as Teachers**: Every error is information about the system requirements. Instead of trying to prevent all errors upfront, we run into them quickly and let them guide our implementation.

**Speed Through Failure**: The fastest path to working software is through intentional collision with edge cases, constraint violations, and integration challenges.

## The EDD Cycle

### 1. Intentional Collision

**Goal**: Hit errors as fast as possible to understand real requirements

```python
# Example: Start with the simplest possible implementation
@router.post("/health-data/upload")
async def upload_health_data(data: dict):
    # This WILL fail - and that's the point
    return {"status": "uploaded", "id": "placeholder"}
```

**Expected Errors** (and what they teach us):

- `422 Unprocessable Entity` → Need data validation
- `500 Internal Server Error` → Need error handling
- `401 Unauthorized` → Need authentication
- `Database connection failed` → Need database setup
- `Schema validation error` → Need proper data models

### 2. Error-Guided Implementation

**Goal**: Let each error tell us exactly what to build next

#### Error-Response Pattern

```python
# Error: ValidationError - no data validation
# Response: Add Pydantic model
class HealthDataUpload(BaseModel):
    data_type: Literal["heart_rate", "steps", "sleep", "activity"]
    values: List[HealthDataPoint]
    timestamp: datetime
    source: str = "apple_watch"

    class Config:
        extra = "forbid"  # Reject unknown fields

# Error: Database not found
# Response: Add database connection
async def get_database():
    if not hasattr(get_database, "_client"):
        get_database._client = firestore.AsyncClient()
    return get_database._client

# Error: User not authenticated
# Response: Add auth dependency
async def get_current_user(token: str = Depends(get_token)):
    try:
        return await validate_firebase_token(token)
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication")

# Error: Data processing timeout
# Response: Add async background processing
async def process_health_data_async(data: HealthDataUpload, user_id: str):
    # Move heavy processing to background task
    await pubsub_client.publish("health-data-processing", {
        "user_id": user_id,
        "data": data.dict(),
        "timestamp": datetime.utcnow().isoformat()
    })
```

### 3. Progressive Hardening

**Goal**: Build resilience incrementally based on real failure modes

#### Hardening Sequence

1. **Basic functionality** - Make it work once
2. **Input validation** - Handle bad data
3. **Error boundaries** - Graceful failure
4. **Performance optimization** - Handle load
5. **Security hardening** - Handle attacks

## EDD Implementation Checklist

### Phase 1: Collision Course

- [ ] Write the minimal API endpoint that compiles
- [ ] Make a real request and collect ALL errors
- [ ] Document every failure mode encountered
- [ ] Prioritize errors by user impact severity

### Phase 2: Targeted Fixes

- [ ] Fix the highest-impact error first
- [ ] Add only the minimal code to resolve that specific error
- [ ] Test that the fix works
- [ ] Move to the next highest-impact error
- [ ] Repeat until basic functionality works

### Phase 3: Edge Case Discovery

- [ ] Send malformed data intentionally
- [ ] Test with missing authentication
- [ ] Try invalid user inputs
- [ ] Simulate network failures
- [ ] Document all new error modes

### Phase 4: Resilience Building

- [ ] Add comprehensive error handling
- [ ] Implement retry logic where appropriate
- [ ] Add logging for debugging
- [ ] Create monitoring and alerts

## Error-Driven Patterns

### 1. The Collision Template

Start every new feature with intentional failure:

```python
# Step 1: Minimal broken implementation
@router.post("/new-feature")
async def new_feature(data: dict):
    # Intentionally simple - will break
    return {"message": "not implemented"}

# Step 2: Make first request and observe failures
# - 422: Need data validation
# - 500: Need business logic
# - 401: Need authentication

# Step 3: Fix errors one by one
@router.post("/new-feature")
async def new_feature(
    data: NewFeatureRequest,  # Fix: Add validation
    user: User = Depends(get_current_user)  # Fix: Add auth
):
    try:
        result = await business_logic(data, user)  # Fix: Add logic
        return {"result": result}
    except BusinessLogicError as e:  # Fix: Add error handling
        raise HTTPException(status_code=400, detail=str(e))
```

### 2. Error-First Testing

Write tests that expect failures:

```python
# Test the errors first, then the success cases
async def test_health_data_upload_errors():
    # Test missing authentication
    response = await client.post("/health-data/upload", json={})
    assert response.status_code == 401

    # Test invalid data
    response = await client.post("/health-data/upload",
        json={"invalid": "data"},
        headers=auth_headers
    )
    assert response.status_code == 422

    # Test valid data (only after errors are handled)
    response = await client.post("/health-data/upload",
        json=valid_health_data,
        headers=auth_headers
    )
    assert response.status_code == 202
```

### 3. Error Logging and Learning

Capture error information for continuous learning:

```python
import structlog
logger = structlog.get_logger()

class ErrorDrivenMiddleware:
    async def __call__(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Log error with context for learning
            await logger.error(
                "request_failed",
                path=request.url.path,
                method=request.method,
                error_type=type(e).__name__,
                error_message=str(e),
                user_agent=request.headers.get("user-agent"),
                body_size=len(await request.body()) if request.body else 0
            )

            # Convert error to appropriate HTTP response
            if isinstance(e, ValidationError):
                return JSONResponse(
                    status_code=422,
                    content={"detail": "Invalid input data"}
                )
            elif isinstance(e, AuthenticationError):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required"}
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Internal server error"}
                )
```

## EDD for Different Layers

### API Layer EDD

```python
# Start broken, fix systematically
@router.post("/insights/generate")
async def generate_insights(request: dict):
    # Will fail immediately - good!
    return {"insights": "placeholder"}

# Error progression:
# 1. 422 → Add request validation
# 2. 401 → Add authentication
# 3. 500 → Add business logic
# 4. 503 → Add external service handling
# 5. 429 → Add rate limiting
```

### Database Layer EDD

```python
# Start with direct database calls
async def store_health_data(data: dict):
    # Will fail - connection, schema, validation
    db = firestore.client()
    doc_ref = db.collection("health_data").add(data)
    return doc_ref.id

# Error progression:
# 1. ConnectionError → Add connection management
# 2. PermissionError → Add authentication
# 3. ValidationError → Add schema validation
# 4. TimeoutError → Add retry logic
```

### ML Model Integration EDD

```python
# Start with direct model calls
async def run_actigraphy_analysis(data: list):
    # Will fail - model loading, preprocessing, memory
    model = torch.load("pat_model.pth")
    result = model(data)
    return result

# Error progression:
# 1. FileNotFoundError → Add model management
# 2. OutOfMemoryError → Add batch processing
# 3. RuntimeError → Add input validation
# 4. TimeoutError → Add async processing
```

## EDD Workflow Steps

### Daily EDD Routine

#### Morning Setup (5 minutes)

1. **Collision Planning**: Choose one new feature to break
2. **Error Budget**: Decide how many errors you want to encounter
3. **Learning Goals**: What do you want the errors to teach you?

#### Implementation Session (25 minutes)

1. **Write broken code** (5 minutes)
2. **Run and collect errors** (5 minutes)
3. **Fix highest priority error** (10 minutes)
4. **Test fix and find next error** (5 minutes)

#### Review Session (10 minutes)

1. **Document what errors taught you**
2. **Update error patterns library**
3. **Plan next collision for tomorrow**

### Error Pattern Library

Maintain a living document of error patterns:

```markdown
# Error Pattern: Missing Authentication
**Symptoms**: 500 errors, "user not found" exceptions
**Root Cause**: No authentication middleware
**Solution Pattern**: Add `Depends(get_current_user)` to endpoints
**Time to Fix**: ~10 minutes
**Prevention**: Start with auth dependency from day 1

# Error Pattern: Database Connection Pool Exhaustion
**Symptoms**: "too many connections" errors during load testing
**Root Cause**: Not using connection pooling properly
**Solution Pattern**: Configure asyncpg pool with max_size=20
**Time to Fix**: ~30 minutes
**Prevention**: Load test early and often
```

## Advanced EDD Techniques

### 1. Chaos-Driven Development

Intentionally inject failures to discover edge cases:

```python
# Chaos middleware for development
class ChaosMiddleware:
    def __init__(self, failure_rate: float = 0.1):
        self.failure_rate = failure_rate

    async def __call__(self, request: Request, call_next):
        # Randomly fail requests to discover error handling gaps
        if random.random() < self.failure_rate:
            failure_type = random.choice([
                TimeoutError("Simulated timeout"),
                ConnectionError("Simulated connection failure"),
                ValueError("Simulated data corruption")
            ])
            raise failure_type

        return await call_next(request)
```

### 2. Error-Driven Load Testing

Use errors to guide performance optimization:

```python
# Start with load test that will fail
async def load_test_health_upload():
    tasks = []
    for i in range(1000):  # This will overwhelm the system
        task = asyncio.create_task(
            upload_health_data(sample_data)
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Analyze failure patterns
    errors = [r for r in results if isinstance(r, Exception)]
    error_types = Counter(type(e).__name__ for e in errors)

    print(f"Error distribution: {error_types}")
    # Use this to prioritize optimizations
```

### 3. Error-Driven Security Testing

Let security failures guide hardening:

```python
# Intentionally vulnerable endpoint for testing
@router.post("/vulnerable-upload")
async def vulnerable_upload(data: str):  # Raw string - will break
    # This will fail with various injection attempts
    eval(data)  # Obviously dangerous - will teach us about input validation
    return {"status": "processed"}

# Test with malicious inputs to discover security needs:
# - SQL injection attempts
# - Script injection attempts
# - Oversized payloads
# - Malformed JSON
```

## EDD Metrics and Success Criteria

### Velocity Metrics

- **Time to First Error**: Should be < 5 minutes
- **Error Resolution Rate**: Should fix 1 error every 10-15 minutes
- **Feature Completion Time**: Should decrease with each error cycle

### Quality Metrics

- **Error Pattern Recognition**: Reuse previous error solutions
- **Defensive Code Coverage**: % of code with error handling
- **Recovery Time**: How fast the system recovers from errors

### Learning Metrics

- **Error Pattern Library Growth**: New patterns discovered weekly
- **Error Prediction Accuracy**: Can you predict the next 3 errors?
- **Knowledge Transfer**: Can team members predict error patterns?

## Common EDD Anti-Patterns

### ❌ Error Avoidance

```python
# DON'T: Try to prevent all errors upfront
def overly_cautious_upload(data):
    if not data:
        raise ValueError("No data")
    if not isinstance(data, dict):
        raise ValueError("Not dict")
    if "timestamp" not in data:
        raise ValueError("No timestamp")
    # ... 50 more validations before doing anything
```

### ✅ Error Embrace

```python
# DO: Let errors guide you to the real requirements
def error_driven_upload(data):
    # Start simple, add validations as errors reveal needs
    return process_data(data)
```

### ❌ Error Hiding

```python
# DON'T: Catch and hide errors
try:
    result = risky_operation()
except Exception:
    return {"status": "success"}  # Lying about failure
```

### ✅ Error Learning

```python
# DO: Let errors teach you
try:
    result = risky_operation()
except SpecificError as e:
    logger.error("learned_something", error=str(e))
    # Add specific handling for this error type
    raise HTTPException(status_code=400, detail="Specific guidance")
```

## Integration with Other Methodologies

### EDD + TDD

1. **Error-Driven Test Writing**: Write tests for expected errors first
2. **Red-Green-Refactor**: Red = errors, Green = fixes, Refactor = optimization
3. **Test-Guided Error Handling**: Let test failures guide error handling

### EDD + Vertical Slice Development

1. **Slice-Specific Error Patterns**: Each slice has predictable error types
2. **End-to-End Error Testing**: Test complete user journeys for errors
3. **Layer-by-Layer Error Handling**: Errors propagate up the stack

### EDD + Continuous Integration

1. **Error-Driven Pipeline Design**: Let failures guide pipeline improvements
2. **Failure-Fast Feedback**: Optimize for quick error feedback
3. **Error Pattern Automation**: Automate handling of known error patterns

## Next Steps

1. **Start Small**: Pick one endpoint and intentionally break it
2. **Document Errors**: Keep a log of every error encountered
3. **Build Error Library**: Create reusable error handling patterns
4. **Share Learning**: Teach the team about error patterns discovered
5. **Iterate**: Apply EDD to progressively larger features

Remember: **The goal isn't to avoid errors, it's to learn from them faster than anyone else.**
