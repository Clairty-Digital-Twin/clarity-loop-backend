# PAT Model Path Configuration Fix

## 🔍 Issue Identified

The external audit revealed that the PAT service is looking for models in the wrong directory:
- **Expected location**: `/usr/local/lib/python3.11/models/pat/`
- **Actual location**: `/app/models/pat/`

### Root Cause

When the application is installed as a Python wheel in the Docker container:
1. The Python files are installed to `/usr/local/lib/python3.11/site-packages/clarity/`
2. The `_PROJECT_ROOT` calculation using `Path(__file__).parent.parent.parent.parent` results in `/usr/local/lib/python3.11/`
3. Therefore, models are expected at `/usr/local/lib/python3.11/models/pat/`
4. But `download_models.sh` downloads them to `/app/models/pat/`

## 🛠️ Solution Implemented

### 1. Enhanced Path Detection Logic

Updated `src/clarity/ml/pat_service.py` with a smarter path detection algorithm:

```python
# Priority order for model paths:
# 1. Environment variable PAT_MODEL_BASE_DIR (if set)
# 2. /app/models/pat/ - Docker production (models downloaded here)
# 3. /models/pat/ - Alternative Docker mount point
# 4. Local development - relative from source

if os.environ.get("PAT_MODEL_BASE_DIR"):
    _PROJECT_ROOT = Path(os.environ["PAT_MODEL_BASE_DIR"]).parent.parent
elif os.path.exists("/app/models/pat"):
    _PROJECT_ROOT = Path("/app")
elif os.path.exists("/models/pat"):
    _PROJECT_ROOT = Path("/")
else:
    _PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
```

### 2. Added Environment Variable Support

Added `PAT_MODEL_BASE_DIR` to the ECS task definition:
```json
{
  "name": "PAT_MODEL_BASE_DIR",
  "value": "/app/models/pat"
}
```

This provides explicit control over where models are located, preventing any ambiguity.

### 3. Updated Allowed Directories

Extended the whitelist in `_sanitize_model_path` to include:
- `/app/models` - Docker production models
- `/models` - Alternative Docker mount

## 📝 Testing the Fix

### Local Testing
```bash
# Run the debug script
python3 debug_pat_paths_production.py

# Test the PAT endpoint
python3 debug_pat_endpoint.py
```

### Production Testing
```bash
# SSH into ECS container (if enabled) or run as ECS task
python3 debug_pat_paths_production.py

# Check logs
aws logs tail /ecs/clarity-backend --follow --region us-east-1
```

## 🚀 Deployment Steps

1. **Build and push new Docker image**:
   ```bash
   docker build -t clarity-backend .
   docker tag clarity-backend:latest 124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend:latest
   docker push 124355672559.dkr.ecr.us-east-1.amazonaws.com/clarity-backend:latest
   ```

2. **Update ECS task definition**:
   ```bash
   aws ecs register-task-definition --cli-input-json file://ops/deployment/ecs-task-definition.json --region us-east-1
   ```

3. **Update ECS service**:
   ```bash
   aws ecs update-service --cluster clarity-backend-cluster --service clarity-backend-service --task-definition clarity-backend --force-new-deployment --region us-east-1
   ```

4. **Monitor deployment**:
   ```bash
   aws ecs wait services-stable --cluster clarity-backend-cluster --services clarity-backend-service --region us-east-1
   ```

## ✅ Verification

The fix is successful when:
1. No more "Model file not found" errors in logs
2. Logs show: "Using Docker production path: /app/models/pat/" or "Using PAT_MODEL_BASE_DIR: /app/models/pat"
3. PAT analysis endpoint returns 200 OK with valid predictions
4. `debug_pat_paths_production.py` confirms models are found at `/app/models/pat/`

## 🔄 Backward Compatibility

The solution maintains backward compatibility by:
1. Checking multiple possible paths in priority order
2. Supporting the existing symlinks created by `create_model_symlinks.sh`
3. Falling back to relative paths for local development
4. Allowing explicit override via environment variable

## 🎯 Summary

This fix ensures the PAT service can reliably find its models regardless of:
- How the Python package is installed (wheel vs source)
- Where the container filesystem places the models
- Whether EFS or local storage is used
- Development vs production environments

The explicit `PAT_MODEL_BASE_DIR` environment variable provides a definitive solution that eliminates all ambiguity.