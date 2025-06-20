# Files Requiring TYPE_CHECKING Import Fixes

This document lists all files where types are imported inside `if TYPE_CHECKING:` blocks but are used outside those blocks in runtime code (function signatures, class definitions, or variable annotations).

## Files to Fix

### 1. **src/clarity/core/config_provider.py**

- **Imported in TYPE_CHECKING**: `MiddlewareConfig`, `Settings`
- **Used in**:
  - Line 26: `settings: Settings` parameter
  - Line 170: `MiddlewareConfig` return type
  - Line 205: `Settings` return type
- **Fix**: Move imports outside TYPE_CHECKING block

### 2. **src/clarity/auth/decorators.py**

- **Imported in TYPE_CHECKING**: `User`
- **Used in**:
  - Line 27: `cast("User | None", kwargs.get("current_user"))`
  - Line 62: `cast("User | None", kwargs.get("current_user"))`
  - Line 87: `cast("User | None", kwargs.get("current_user"))`
- **Fix**: Import `User` outside TYPE_CHECKING and remove string literals from cast

### 3. **src/clarity/middleware/security_headers.py**

- **Imported in TYPE_CHECKING**: `Receive`, `Scope`, `Send`
- **Used in**: Likely in type annotations (need to verify)
- **Fix**: Move imports outside TYPE_CHECKING if actually used

### 4. **src/clarity/auth/aws_cognito_provider.py**

- **Imported in TYPE_CHECKING**: `AttributeTypeTypeDef`
- **Used in**: Type annotations (2 occurrences)
- **Fix**: Move import outside TYPE_CHECKING block

### 5. **src/clarity/integrations/apple_watch.py**

- **Imported in TYPE_CHECKING**: `HealthDataPoint`
- **Used in**: Type annotations (10 occurrences)
- **Fix**: Move import outside TYPE_CHECKING block

### 6. **src/clarity/integrations/HealthKit.py**

- **Imported in TYPE_CHECKING**: `types` module
- **Used in**: Type annotations (8 occurrences)
- **Fix**: Move import outside TYPE_CHECKING block

### 7. **src/clarity/ML/pat_optimization.py**

- **Imported in TYPE_CHECKING**: `PATModelService`
- **Used in**: Type annotations (10 occurrences)
- **Fix**: Move import outside TYPE_CHECKING block

### 8. **src/clarity/ML/processors/activity_processor.py**

- **Imported in TYPE_CHECKING**: `Sequence`
- **Used in**: Type annotations (2 occurrences)
- **Fix**: Move import outside TYPE_CHECKING block

### 9. **src/clarity/ML/analysis_pipeline.py**

- **Imported in TYPE_CHECKING**: `SleepFeatures`
- **Used in**: Type annotations (6 occurrences)
- **Fix**: Move import outside TYPE_CHECKING block

### 10. **src/clarity/API/v1/metrics.py**

- **Imported in TYPE_CHECKING**: `types` module
- **Used in**: Type annotations (2 occurrences)
- **Fix**: Move import outside TYPE_CHECKING block

### 11. **src/clarity/main_bulletproof.py**

- **Imported in TYPE_CHECKING**: `AsyncGenerator`
- **Used in**: Type annotations (2 occurrences)
- **Fix**: Move import outside TYPE_CHECKING block

### 12. **src/clarity/ports/config_ports.py**

- **Imported in TYPE_CHECKING**: `MiddlewareConfig`
- **Used in**: Type annotations (4 occurrences)
- **Fix**: Move import outside TYPE_CHECKING block

### 13. **src/clarity/services/s3_storage_service.py**

- **Imported in TYPE_CHECKING**: `S3Client`
- **Used in**: Type annotations (2 occurrences)
- **Fix**: Move import outside TYPE_CHECKING block

### 14. **src/clarity/services/dynamodb_service.py**

- **Imported in TYPE_CHECKING**: `DynamoDBServiceResource`
- **Used in**: Type annotations (2 occurrences)
- **Fix**: Move import outside TYPE_CHECKING block

## Files Already Fixed (for reference)

- src/clarity/ML/processors/cardio_processor.py
- src/clarity/ML/preprocessing.py
- src/clarity/storage/dynamodb_client.py
- src/clarity/auth/aws_auth_provider.py

## Files with Empty TYPE_CHECKING Blocks (No Action Needed)

- src/clarity/middleware/request_logger.py
- src/clarity/middleware/request_size_limiter.py
- src/clarity/middleware/auth_middleware.py
- src/clarity/core/cloud.py
- src/clarity/auth/modal_auth_fix.py
- src/clarity/API/v1/websocket/app_example.py
- src/clarity/ports/middleware_ports.py
- src/clarity/services/messaging/insight_subscriber.py

## Special Cases

### src/clarity/core/container.py

- Has unusual import pattern with `app as clarity_app` that may need special handling

### src/clarity/ML/inference_engine.py

- Shows high usage count but may be a false positive due to parentheses in code

### src/clarity/ML/processors/respiration_processor.py

- Shows `datetime` usage but this might be the standard library datetime, not a TYPE_CHECKING import

## Summary

Total files needing fixes: **14 files**

These files have types that are only imported inside TYPE_CHECKING blocks but are used in runtime code, which will cause NameError exceptions when the code runs in production.
