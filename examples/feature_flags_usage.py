"""Example usage of enhanced feature flag system in CLARITY application."""

import asyncio
import logging

from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel

from clarity.core.config_aws import Settings
from clarity.core.feature_flags_integration import (
    get_feature_flag_health,
    is_enhanced_security_enabled,
    is_mania_risk_enabled,
    is_pat_model_v2_enabled,
    setup_feature_flags_for_app,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Feature Flags Example")

# Create settings (would normally come from environment)
settings = Settings(
    environment="staging",
    redis_url="redis://localhost:6379",
)


class HealthAnalysisRequest(BaseModel):
    """Request model for health analysis."""

    user_id: str
    data: dict[str, float]


class HealthAnalysisResponse(BaseModel):
    """Response model for health analysis."""

    user_id: str
    mania_risk_enabled: bool
    pat_v2_enabled: bool
    enhanced_security: bool
    analysis_results: dict[str, any]


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize feature flag system on startup."""
    # Set up enhanced feature flags
    manager = setup_feature_flags_for_app(app, settings)
    logger.info("Feature flag system initialized")

    # Log initial configuration
    health = get_feature_flag_health()
    logger.info("Feature flag system health: %s", health)


@app.get("/health")
async def health_check():
    """Health check endpoint including feature flag status."""
    ff_health = get_feature_flag_health()

    # Determine overall health
    is_healthy = ff_health["healthy"] and ff_health["circuit_breaker_state"] == "closed"

    return {
        "status": "healthy" if is_healthy else "degraded",
        "feature_flags": ff_health,
    }


@app.get("/feature-flags/status")
async def feature_flag_status():
    """Get detailed feature flag system status."""
    return get_feature_flag_health()


@app.post("/feature-flags/refresh")
async def refresh_feature_flags():
    """Manually trigger feature flag refresh."""
    if not hasattr(app.state, "feature_flag_manager"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature flag manager not initialized",
        )

    manager = app.state.feature_flag_manager
    success = await manager.refresh_async()

    if success:
        return {
            "status": "success",
            "message": "Feature flags refreshed successfully",
            "health": get_feature_flag_health(),
        }
    return Response(
        content={
            "status": "failed",
            "message": "Failed to refresh feature flags",
            "health": get_feature_flag_health(),
        },
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.post("/analyze", response_model=HealthAnalysisResponse)
async def analyze_health_data(request: HealthAnalysisRequest):
    """Analyze health data with feature flag controlled behavior."""
    user_id = request.user_id

    # Check feature flags for this user
    mania_enabled = is_mania_risk_enabled(user_id)
    pat_v2_enabled = is_pat_model_v2_enabled(user_id)
    enhanced_security = is_enhanced_security_enabled()

    results = {
        "base_analysis": "completed",
    }

    # Conditionally add mania risk analysis
    if mania_enabled:
        logger.info("Running mania risk analysis for user %s", user_id)
        results["mania_risk"] = {
            "status": "analyzed",
            "risk_level": "low",  # Placeholder
        }

    # Use appropriate PAT model version
    if pat_v2_enabled:
        logger.info("Using PAT model v2 for user %s", user_id)
        results["pat_analysis"] = {
            "version": "v2",
            "result": "normal",  # Placeholder
        }
    else:
        logger.info("Using PAT model v1 for user %s", user_id)
        results["pat_analysis"] = {
            "version": "v1",
            "result": "normal",  # Placeholder
        }

    # Apply enhanced security if enabled
    if enhanced_security:
        logger.info("Applying enhanced security checks")
        # Would perform additional security validations here

    return HealthAnalysisResponse(
        user_id=user_id,
        mania_risk_enabled=mania_enabled,
        pat_v2_enabled=pat_v2_enabled,
        enhanced_security=enhanced_security,
        analysis_results=results,
    )


@app.websocket("/feature-flags/stream")
async def feature_flag_stream(websocket) -> None:
    """WebSocket endpoint for real-time feature flag updates."""
    await websocket.accept()

    try:
        # Send initial status
        await websocket.send_json(get_feature_flag_health())

        # Stream updates periodically
        while True:
            await asyncio.sleep(5)  # Update every 5 seconds
            health = get_feature_flag_health()
            await websocket.send_json(
                {
                    "type": "update",
                    "data": health,
                }
            )

    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    # Run the example app
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
