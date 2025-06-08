# Modal Environment Setup

## Current Deployment Configuration

Your CLARITY backend is now deployed in two separate Modal environments:

### Development Environment
- **URL**: https://crave-trinity-dev--clarity-backend-fastapi-app.modal.run
- **Health Check**: https://crave-trinity-dev--clarity-backend-fastapi-app.modal.run/health
- **Environment**: development
- **Purpose**: Testing, development, and debugging
- **Behavior**: Uses mock services when Firebase/GCP credentials fail

### Production Environment
- **URL**: https://crave-trinity-prod--clarity-backend-fastapi-app.modal.run
- **Health Check**: https://crave-trinity-prod--clarity-backend-fastapi-app.modal.run/health
- **Environment**: production (Note: currently showing development due to config issue - will be fixed)
- **Purpose**: Live application for your SwiftUI frontend
- **Behavior**: Requires real credentials, no mock fallbacks

## Deployment Commands

```bash
# Deploy to development
modal deploy --env dev modal_deploy_optimized.py

# Deploy to production
modal deploy --env prod modal_deploy_optimized.py

# Test deployments
modal run --env dev modal_deploy_optimized.py::health_check
modal run --env prod modal_deploy_optimized.py::health_check
```

## SwiftUI Frontend Configuration

For your SwiftUI app, use these URLs:

```swift
// Development
let devBaseURL = "https://crave-trinity-dev--clarity-backend-fastapi-app.modal.run"

// Production
let prodBaseURL = "https://crave-trinity-prod--clarity-backend-fastapi-app.modal.run"
```

## Environment Detection

The Modal deployment automatically sets the `MODAL_ENVIRONMENT` variable:
- `dev` environment → `ENVIRONMENT=development`
- `prod` environment → `ENVIRONMENT=production`

## Notes

1. Both environments use the same codebase but different configurations
2. Secrets are configured separately for each environment
3. Old deployments in the "main" environment are stopped and will be auto-cleaned by Modal
4. Deploy times are ~20 seconds thanks to layered caching

## Next Steps

1. Update your SwiftUI app to use the production URL
2. Monitor logs with `modal app logs --env prod clarity-backend`
3. Scale as needed using Modal's auto-scaling features