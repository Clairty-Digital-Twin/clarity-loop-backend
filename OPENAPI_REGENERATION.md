# OpenAPI Specification Regeneration Guide

## Overview

This document explains how to regenerate the OpenAPI specification after making changes to API models (like ActigraphyAnalysis).

## Quick Regeneration

To regenerate the OpenAPI spec after model changes:

```bash
# From the project root directory
python scripts/generate_openapi.py

# Clean up the generated spec (optional)
python scripts/clean_openapi.py

# Validate the spec (optional)
./scripts/validate_openapi.sh
```

## When to Regenerate

You should regenerate the OpenAPI spec when:

1. **Adding new fields to response models** (e.g., adding `mania_risk_score` and `mania_alert_level` to ActigraphyAnalysis)
2. **Modifying existing field types or descriptions**
3. **Adding new API endpoints**
4. **Changing request/response schemas**
5. **Updating validation rules**

## Recent Changes

The ActigraphyAnalysis model now includes mania risk fields:

```python
class ActigraphyAnalysis(BaseModel):
    # ... existing fields ...
    mania_risk_score: float = Field(default=0.0, description="Mania risk score (0-1)")
    mania_alert_level: str = Field(default="none", description="Mania risk level: none/low/moderate/high")
```

These fields have default values to ensure backward compatibility.

## Files Affected

- `/docs/api/openapi.json` - Main OpenAPI spec (generated)
- `/docs/api/openapi-cleaned.json` - Cleaned version (generated)
- `/docs/api/openapi-cleaned.yaml` - YAML version (generated)

## CI/CD Integration

The OpenAPI spec generation is integrated into the CI pipeline:

1. **GitHub Actions**: The CI workflow validates that the OpenAPI spec is up-to-date
2. **Pre-commit**: Consider adding a pre-commit hook to regenerate specs automatically

## Troubleshooting

If regeneration fails:

1. Ensure all required environment variables are set:
   ```bash
   export SKIP_AWS_INIT=true
   export ENABLE_AUTH=false
   ```

2. Check that all model imports are working correctly

3. Verify that the FastAPI app can be imported without errors

## Best Practices

1. **Always regenerate** after model changes
2. **Commit the updated specs** with your code changes
3. **Review the diff** to ensure expected changes
4. **Test API compatibility** with existing clients after regeneration