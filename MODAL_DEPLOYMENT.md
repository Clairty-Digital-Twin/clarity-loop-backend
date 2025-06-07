# Modal Deployment Guide

This project includes optimized Modal deployment scripts for different use cases.

## Deployment Scripts

### 🚀 Production Deployment: `modal_deploy_optimized.py`

**Use this for production and regular development.**

- **Layered dependency caching** for lightning-fast deploys
- **Complete ML stack** (PyTorch, transformers, scikit-learn)
- **Production scaling** configuration
- **Comprehensive health checks**

```bash
# Deploy to production
modal deploy modal_deploy_optimized.py

# Test health check
modal run modal_deploy_optimized.py::health_check

# Test credentials
modal run modal_deploy_optimized.py::credentials_test
```

**First deploy**: ~90 seconds (installs all dependencies)  
**Subsequent deploys**: ~20 seconds (uses cached layers)

### 🧪 Development Deployment: `modal_deploy_simple.py`

**Use this for quick testing and credential verification.**

- **Minimal dependencies** for fast iteration
- **Basic functionality** testing
- **Credential validation** without heavy ML libs

```bash
# Quick test deploy
modal run modal_deploy_simple.py::comprehensive_modal_test

# Test basic imports
modal run modal_deploy_simple.py::test_basic_imports

# Check credentials
modal run modal_deploy_simple.py::ping_credentials
```

**Deploy time**: ~5 seconds

## Architecture

### Layered Caching Strategy

The optimized deployment uses 4 layers:

1. **Base Layer**: FastAPI, Pydantic, basic utilities
2. **Cloud Layer**: Google Cloud, Firebase dependencies  
3. **ML Layer**: PyTorch, transformers, scientific libraries
4. **Code Layer**: Your application source code

Only the code layer rebuilds when you make changes, making deploys extremely fast.

### Scaling Configuration

Production deployment includes:
- **4 CPU cores** for ML workloads
- **4GB memory** for model inference
- **Auto-scaling** up to 10 containers
- **5-minute warm containers** for better performance

## Live Endpoints

Your deployed application is available at:
- **Production**: `https://crave-trinity--clarity-backend-optimized-fastapi-app.modal.run`
- **Health Check**: `https://crave-trinity--clarity-backend-optimized-fastapi-app.modal.run/health`

## Environment Setup

Required Modal secrets:
- `googlecloud-secret` containing:
  - `GOOGLE_APPLICATION_CREDENTIALS_JSON`
  - `GEMINI_API_KEY`
  - `FIREBASE_PROJECT_ID`
  - `GCP_PROJECT_ID`

## Best Practices

1. **Use optimized deployment** for all regular work
2. **Use simple deployment** only for quick tests
3. **Always test health check** after deploying
4. **Monitor build times** - should be ~20s after first deploy
5. **Check logs** if health check fails

## Troubleshooting

- **Long build times**: Check if you're using the optimized script
- **Import errors**: Verify all dependencies in the relevant layer
- **Health check fails**: Check Modal secrets configuration
- **Memory issues**: Increase memory in function decorator

## Commands Reference

```bash
# Production workflows
modal deploy modal_deploy_optimized.py
modal run modal_deploy_optimized.py::health_check
curl https://your-app.modal.run/health

# Development workflows  
modal run modal_deploy_simple.py::test_basic_imports
modal run modal_deploy_simple.py::ping_credentials

# Monitoring
modal app logs clarity-backend-optimized
modal app list
```