# Repository Audit Report - Root Directory Cleanup

## Executive Summary

The Docker image is 6GB+ primarily due to:
1. **1.6GB `clarity-amd64.tar.gz`** - An exported Docker image sitting in the root directory
2. **341MB `ops/terraform/` directory** - Contains Terraform provider binaries
3. **13MB `node_modules/`** - Not needed for Python application
4. **7.2MB `htmlcov/`** - Test coverage HTML reports
5. Various test artifacts and logs

## Critical Issues Found

### 1. Large Files Not Properly Ignored

**Problem**: The `.dockerignore` file includes `*.tar.gz`, but the 1.6GB `clarity-amd64.tar.gz` is still present.

**Files to Remove**:
- `clarity-amd64.tar.gz` (1.6GB) - Exported Docker image, should not be in source control
- `coverage.json` (610KB) - Test coverage data
- `coverage.xml` (366KB) - Test coverage data
- `htmlcov/` (7.2MB) - HTML coverage reports

### 2. Mixed Deployment Configurations

**Multiple Cloud Platforms**:
- **AWS**: `gunicorn.aws.conf.py`, `AWS_DEPLOYMENT_PLAN.md`, `AWS_SETUP_GUIDE.md`, `ops/ecs-*` files
- **Google Cloud**: `cloudbuild.yaml`, Terraform configs in `ops/terraform/`
- **Modal**: `deploy_modal_prod.sh`, `.modalignore`, `src/clarity/auth/modal_auth_fix.py`

**Redundant Configuration Files**:
- `gunicorn.conf.py` - Generic config
- `gunicorn.aws.conf.py` - AWS-specific config
- Multiple Dockerfiles: `Dockerfile`, `Dockerfile.minimal`

### 3. Development Artifacts in Root

**Files that shouldn't be in production**:
- `test_auth_definitive.py` - Test script in root
- `test_deployment.py` - Test script in root
- `start_clarity_services.py` - Development helper script
- `quick_demo.sh` - Demo script

### 4. Infrastructure Files Bloating the Image

**Large Directories**:
- `ops/terraform/.terraform/` (341MB) - Terraform provider binaries
- `node_modules/` (13MB) - JavaScript dependencies (not needed for Python app)
- `logs/` (3.9MB) - Application logs
- `dist/` (224KB) - Build artifacts

## Recommendations

### Immediate Actions (Reduce Docker Image Size)

1. **Delete the 1.6GB tar.gz file**:
   ```bash
   rm clarity-amd64.tar.gz
   ```

2. **Clean up test artifacts**:
   ```bash
   rm -rf htmlcov/ coverage.json coverage.xml
   rm test_auth_definitive.py test_deployment.py
   ```

3. **Remove unnecessary directories**:
   ```bash
   rm -rf node_modules/
   rm -rf ops/terraform/.terraform/
   rm -rf logs/*
   rm -rf dist/
   ```

4. **Update `.dockerignore`** to ensure it includes:
   ```
   # Already has *.tar.gz but add explicitly
   clarity-amd64.tar.gz
   
   # Test files in root
   test_*.py
   
   # Development scripts
   start_clarity_services.py
   quick_demo.sh
   deploy_modal_prod.sh
   
   # Coverage artifacts
   coverage.json
   coverage.xml
   .coverage
   
   # Infrastructure
   ops/terraform/.terraform/
   ```

### Platform-Specific Cleanup

**For AWS Deployment**:
- Keep: `Dockerfile`, `gunicorn.aws.conf.py`, `entrypoint.sh`, AWS guides
- Remove: `cloudbuild.yaml`, Modal files, `gunicorn.conf.py`

**For Google Cloud Deployment**:
- Keep: `Dockerfile`, `gunicorn.conf.py`, `cloudbuild.yaml`
- Remove: AWS-specific files, Modal files

**For Modal Deployment**:
- Keep: Modal-specific files
- Remove: Docker files, gunicorn configs, cloud build files

### Docker Optimization

1. **Use `Dockerfile.minimal`** for AWS - it's already more optimized
2. **Consider multi-stage builds** - Already implemented in main Dockerfile
3. **Add `.dockerignore` entries** for all development/test files

### Git Repository Cleanup

Add to `.gitignore`:
```
# Docker exports
*.tar.gz
clarity-amd64.tar.gz

# Coverage artifacts
coverage.json
coverage.xml
htmlcov/

# Terraform state
ops/terraform/.terraform/
*.tfstate
*.tfstate.backup

# Logs
logs/

# Build artifacts
dist/
build/

# Node modules (if not needed)
node_modules/
```

## Impact

After cleanup, the Docker build context will reduce from ~2GB to ~10MB, resulting in:
- Faster Docker builds
- Smaller final images
- No ECR push timeouts
- Cleaner repository structure

## Next Steps

1. Remove identified files
2. Update `.dockerignore` and `.gitignore`
3. Choose a single deployment platform and remove others
4. Rebuild Docker image and verify size reduction