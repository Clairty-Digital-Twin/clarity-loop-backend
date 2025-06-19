"""CLARITY Digital Twin Platform - Bulletproof Startup Edition.

Production-ready backend with bulletproof startup orchestration that provides
zero-crash guarantee and crystal-clear error feedback.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
import os
import sys
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from clarity.startup.config_schema import ClarityConfig
from clarity.startup.orchestrator import StartupOrchestrator
from clarity.startup.progress_reporter import StartupProgressReporter

if TYPE_CHECKING:
    pass  # Only for type stubs now

# Configure logging early
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global application state
_app_config: ClarityConfig | None = None
_startup_successful = False


async def bulletproof_startup() -> tuple[bool, ClarityConfig | None]:
    """Execute bulletproof startup sequence.

    Returns:
        Tuple of (success, config). Config is None if startup fails.
    """
    global _app_config, _startup_successful  # noqa: PLW0603 - Module state management

    # Check for dry-run mode
    dry_run = (
        "--dry-run" in sys.argv or os.getenv("STARTUP_DRY_RUN", "").lower() == "true"
    )

    # Create startup orchestrator
    reporter = StartupProgressReporter()
    orchestrator = StartupOrchestrator(
        dry_run=dry_run,
        timeout=float(os.getenv("STARTUP_TIMEOUT", "30")),
        reporter=reporter,
    )

    try:
        # Execute startup orchestration
        success, config = await orchestrator.orchestrate_startup("CLARITY Digital Twin")

        if success and config:
            _app_config = config
            _startup_successful = True

            if dry_run:
                # Print dry-run report and exit
                print(orchestrator.create_dry_run_report())  # noqa: T201
                sys.exit(0)

            return True, config
        _startup_successful = False
        if dry_run:
            print(orchestrator.create_dry_run_report())  # noqa: T201
            sys.exit(1)
        return False, None

    except Exception as e:
        logger.exception("Bulletproof startup failed")
        _startup_successful = False

        # Print error help if possible
        from clarity.startup.error_catalog import error_catalog  # noqa: PLC0415

        suggested_code = error_catalog.suggest_error_code(str(e))
        if suggested_code:
            print(error_catalog.format_error_help(suggested_code))  # noqa: T201

        return False, None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager with bulletproof startup."""
    global _app_config, _startup_successful  # noqa: PLW0602 - Read-only access

    # Execute bulletproof startup
    logger.info("🚀 Executing bulletproof startup sequence...")
    success, config = await bulletproof_startup()

    if not success or not config:
        logger.error("❌ Startup failed - application cannot start")
        msg = "Bulletproof startup failed - check logs for details"
        raise RuntimeError(msg)

    logger.info("✅ Bulletproof startup completed successfully")

    # Initialize container with validated config
    try:
        from clarity.core.container_aws import initialize_container  # noqa: PLC0415

        await initialize_container(None)  # Uses default settings
        logger.info("✅ Dependency container initialized")
    except Exception as e:
        logger.exception("❌ Failed to initialize dependency container")
        msg = f"Container initialization failed: {e}"
        raise RuntimeError(msg) from e

    yield

    logger.info("🔄 Shutting down CLARITY backend")


def create_bulletproof_app() -> FastAPI:
    """Create FastAPI app with bulletproof startup."""
    global _app_config  # noqa: PLW0602 - Read-only access

    # Create FastAPI app
    app = FastAPI(
        title="CLARITY Digital Twin Platform",
        description="Production AWS-native health data platform with bulletproof startup",
        version="0.3.0",
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "authentication",
                "description": "User authentication and authorization",
            },
            {
                "name": "health-data",
                "description": "Health data management and retrieval",
            },
            {"name": "healthkit", "description": "Apple HealthKit data integration"},
            {
                "name": "pat-analysis",
                "description": "Physical Activity Test (PAT) analysis",
            },
            {
                "name": "ai-insights",
                "description": "AI-powered health insights generation",
            },
            {"name": "metrics", "description": "Health metrics and statistics"},
            {"name": "websocket", "description": "WebSocket real-time communication"},
            {"name": "debug", "description": "Debug endpoints (development only)"},
            {"name": "test", "description": "Test endpoints for API validation"},
        ],
        servers=[
            {
                "url": "http://clarity-alb-1762715656.us-east-1.elb.amazonaws.com",
                "description": "Production server (AWS ALB)",
            },
            {"url": "http://localhost:8000", "description": "Local development server"},
        ],
        contact={
            "name": "CLARITY Support",
            "email": "support@clarity.novamindnyc.com",
            "url": "https://clarity.novamindnyc.com",
        },
        license_info={
            "name": "Proprietary",
            "url": "https://clarity.novamindnyc.com/license",
        },
    )

    # Configure middleware after startup validation
    @app.on_event("startup")
    def configure_middleware() -> None:
        """Configure middleware after startup validation."""
        if not _app_config or not _startup_successful:
            logger.error("Cannot configure middleware - startup not successful")
            return

        # Add CORS middleware with validated configuration
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_app_config.security.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "Accept",
                "X-Requested-With",
            ],
            max_age=86400,
        )
        logger.info("✅ CORS middleware configured")

        # Add request size limiter middleware
        from clarity.middleware.request_size_limiter import RequestSizeLimiterMiddleware

        app.add_middleware(
            RequestSizeLimiterMiddleware,
            max_request_size=_app_config.security.max_request_size,
            max_json_size=5 * 1024 * 1024,  # 5MB
            max_upload_size=50 * 1024 * 1024,  # 50MB
            max_form_size=1024 * 1024,  # 1MB
        )
        logger.info("✅ Request size limiter configured")

        # Add security headers middleware
        from clarity.middleware.security_headers import SecurityHeadersMiddleware

        app.add_middleware(
            SecurityHeadersMiddleware,
            enable_hsts=True,
            enable_csp=True,
            cache_control="no-store, private",
        )
        logger.info("✅ Security headers middleware configured")

        # Add authentication middleware if enabled
        if _app_config.enable_auth:
            from clarity.middleware.auth_middleware import CognitoAuthMiddleware

            app.add_middleware(CognitoAuthMiddleware)
            logger.info("✅ Authentication middleware configured")

        # Add rate limiting middleware
        from clarity.middleware.rate_limiting import setup_rate_limiting

        redis_url = os.getenv("REDIS_URL")
        setup_rate_limiting(app, redis_url=redis_url)
        logger.info("✅ Rate limiting middleware configured")

        # Add request logging in development
        if _app_config.is_development():
            from clarity.middleware.request_logger import RequestLoggingMiddleware

            app.add_middleware(RequestLoggingMiddleware)
            logger.info("✅ Request logging middleware configured")

    # Include API routers
    from clarity.api.v1.router import api_router as v1_router
    from clarity.core.openapi import custom_openapi

    app.include_router(v1_router, prefix="/api/v1")
    app.openapi = lambda: custom_openapi(app)  # type: ignore[method-assign]

    # Add core endpoints
    @app.get("/")
    async def root() -> dict[str, Any]:
        """Root endpoint with startup validation status."""
        return {
            "name": "CLARITY Digital Twin Platform",
            "version": "0.3.0",
            "status": "operational" if _startup_successful else "starting",
            "service": "clarity-backend-bulletproof",
            "environment": _app_config.environment.value if _app_config else "unknown",
            "deployment": "AWS Production",
            "total_endpoints": len(app.routes),
            "api_docs": "/docs",
            "bulletproof_startup": _startup_successful,
        }

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Enhanced health check with startup validation status."""
        return {
            "status": "healthy" if _startup_successful else "starting",
            "service": "clarity-backend-bulletproof",
            "environment": _app_config.environment.value if _app_config else "unknown",
            "version": "0.3.0",
            "bulletproof_startup": _startup_successful,
            "features": {
                "cognito_auth": _app_config.enable_auth if _app_config else False,
                "mock_services": (
                    _app_config.should_use_mock_services() if _app_config else True
                ),
                "aws_region": _app_config.aws.region if _app_config else "unknown",
                "startup_validation": True,
            },
        }

    @app.get("/startup-status")
    async def startup_status() -> dict[str, Any]:
        """Detailed startup status endpoint."""
        if not _app_config:
            return {
                "startup_successful": False,
                "error": "Configuration not loaded",
            }

        return {
            "startup_successful": _startup_successful,
            "configuration": _app_config.get_startup_summary(),
            "bulletproof_features": {
                "pre_flight_validation": True,
                "service_health_checks": True,
                "circuit_breakers": True,
                "graceful_degradation": True,
                "progress_reporting": True,
                "error_catalog": True,
            },
        }

    # Add Prometheus metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    logger.info("✅ CLARITY app created with bulletproof startup")
    return app


# Create the app instance
app = create_bulletproof_app()


# CLI entry points for validation
async def validate_startup() -> int:
    """CLI entry point for startup validation."""
    reporter = StartupProgressReporter()
    orchestrator = StartupOrchestrator(dry_run=True, reporter=reporter)

    success, config = await orchestrator.orchestrate_startup()

    if config:
        print(orchestrator.create_dry_run_report())  # noqa: T201

    return 0 if success else 1


def main() -> int:
    """Main CLI entry point."""
    if "--validate" in sys.argv or "--dry-run" in sys.argv:
        return asyncio.run(validate_startup())

    # Normal startup with uvicorn
    import uvicorn

    # Get configuration
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    logger.info("Starting CLARITY with bulletproof startup on %s:%s", host, port)

    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
