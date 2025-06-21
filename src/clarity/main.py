"""CLARITY Digital Twin Platform - Unified Production Main.

This is the professional, unified entry point that combines the best of both
startup approaches with a clean, configurable architecture.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
import os
import sys
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from clarity.core.config_adapter import clarity_config_to_settings
from clarity.core.container_aws import get_container, initialize_container
from clarity.core.openapi import custom_openapi
from clarity.services.gcp_credentials import initialize_gcp_credentials
from clarity.startup.config_schema import ClarityConfig
from clarity.startup.orchestrator import StartupOrchestrator
from clarity.startup.progress_reporter import StartupProgressReporter
from clarity.version import get_version

if TYPE_CHECKING:
    from clarity.core.container_aws import DependencyContainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
_config: ClarityConfig | None = None
_container: DependencyContainer | None = None


def get_startup_mode() -> str:
    """Determine startup mode from environment or command line."""
    if "--bulletproof" in sys.argv:
        return "bulletproof"
    if os.getenv("STARTUP_MODE", "").lower() == "bulletproof":
        return "bulletproof"
    if os.getenv("ENVIRONMENT", "").lower() == "production":
        return "bulletproof"  # Default to bulletproof in production
    return "standard"


def standard_startup() -> tuple[bool, ClarityConfig | None]:
    """Execute standard startup sequence."""
    try:
        from clarity.startup.config_schema import load_config  # noqa: PLC0415

        config = load_config()
        return True, config
    except Exception:
        logger.exception("Standard startup failed")
        return False, None


async def bulletproof_startup() -> tuple[bool, ClarityConfig | None]:
    """Execute bulletproof startup sequence with comprehensive validation."""
    dry_run = (
        "--dry-run" in sys.argv or os.getenv("STARTUP_DRY_RUN", "").lower() == "true"
    )

    reporter = StartupProgressReporter()
    orchestrator = StartupOrchestrator(
        dry_run=dry_run,
        timeout=float(os.getenv("STARTUP_TIMEOUT", "30")),
        reporter=reporter,
    )

    try:
        success, config = await orchestrator.orchestrate_startup("CLARITY Digital Twin")

        if dry_run and success:
            print(orchestrator.create_dry_run_report())  # noqa: T201
            sys.exit(0)

        return success, config

    except Exception as e:
        logger.exception("Bulletproof startup failed")

        # Provide error help
        from clarity.startup.error_catalog import error_catalog  # noqa: PLC0415

        suggested_code = error_catalog.suggest_error_code(str(e))
        if suggested_code:
            print(error_catalog.format_error_help(suggested_code))  # noqa: T201

        return False, None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Unified application lifespan manager."""
    global _config, _container  # noqa: PLW0603

    # Determine startup mode
    mode = get_startup_mode()
    logger.info("Starting CLARITY in %s mode", mode)

    # Execute appropriate startup
    if mode == "bulletproof":
        success, config = await bulletproof_startup()
    else:
        success, config = standard_startup()

    if not success or not config:
        msg = f"{mode.title()} startup failed - check logs for details"
        raise RuntimeError(msg)

    _config = config
    logger.info("✅ Configuration loaded successfully")

    # Initialize GCP credentials
    try:
        initialize_gcp_credentials()
        logger.info("✅ GCP credentials initialized")
    except Exception as e:  # noqa: BLE001
        logger.warning("GCP credentials initialization warning: %s", e)
        # Non-fatal - some features may not work

    # Initialize dependency container
    try:
        # Convert ClarityConfig to Settings for container initialization
        settings = clarity_config_to_settings(config)
        _container = await initialize_container(settings)
        logger.info("✅ Dependency container initialized")
    except Exception as e:
        logger.exception("Failed to initialize container")
        msg = f"Container initialization failed: {e}"
        raise RuntimeError(msg) from e

    logger.info("✅ CLARITY backend started successfully")

    yield

    # Cleanup
    logger.info("Shutting down CLARITY backend...")
    if _container:
        # Add any cleanup logic here
        pass


def configure_middleware_from_env(app: FastAPI) -> None:
    """Configure middleware based on environment variables."""
    # Get environment settings
    environment = os.getenv("ENVIRONMENT", "development").lower()
    enable_auth = os.getenv("ENABLE_AUTH", "true").lower() == "true"
    cors_origins_str = os.getenv(
        "CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000"
    )
    cors_origins = [
        origin.strip() for origin in cors_origins_str.split(",") if origin.strip()
    ]
    max_request_size = int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB default

    # NOTE: Middleware is executed in REVERSE order of registration
    # So SecurityHeadersMiddleware should be added LAST to ensure it runs FIRST

    # Development middleware (runs last)
    if environment == "development":
        from clarity.middleware.request_logger import (  # noqa: PLC0415
            RequestLoggingMiddleware,
        )

        app.add_middleware(RequestLoggingMiddleware)

    # Auth middleware if enabled
    if enable_auth:
        from clarity.middleware.auth_middleware import (  # noqa: PLC0415
            CognitoAuthMiddleware,
        )

        app.add_middleware(CognitoAuthMiddleware)

    # Rate limiting
    from clarity.middleware.rate_limiting import setup_rate_limiting  # noqa: PLC0415

    redis_url = os.getenv("REDIS_URL")
    setup_rate_limiting(app, redis_url=redis_url)

    # Request size limiter
    from clarity.middleware.request_size_limiter import (  # noqa: PLC0415
        RequestSizeLimiterMiddleware,
    )

    app.add_middleware(
        RequestSizeLimiterMiddleware,
        max_request_size=max_request_size,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=86400,
    )

    # Security headers - MUST BE LAST (runs first on responses)
    from clarity.middleware.security_headers import (  # noqa: PLC0415
        SecurityHeadersMiddleware,
    )

    app.add_middleware(
        SecurityHeadersMiddleware,
        enable_hsts=environment == "production",
        enable_csp=True,
        cache_control="no-store, private",
    )


def include_routers(app: FastAPI) -> None:
    """Include all API routers."""
    from clarity.api.v1.router import api_router as v1_router  # noqa: PLC0415

    app.include_router(v1_router, prefix="/api/v1")

    # Mount Prometheus metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="CLARITY Digital Twin Platform",
        description="AI-powered health insights platform",
        version=get_version(),
        lifespan=lifespan,
        openapi_url="/api/v1/openapi.json",
        docs_url=None,  # Disable default docs to use custom self-hosted
        redoc_url="/api/v1/redoc",
    )

    # Configure middleware based on environment variables
    # This must happen before the app starts
    configure_middleware_from_env(app)

    # Set custom OpenAPI schema
    app.openapi = lambda: custom_openapi(app)  # type: ignore[method-assign]

    # Include routers here so they're available immediately for tests
    include_routers(app)

    # Mount static files for self-hosted Swagger UI
    # Get the directory where this file is located
    from pathlib import Path  # noqa: PLC0415

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Custom Swagger UI with self-hosted assets
    @app.get("/api/v1/docs", include_in_schema=False)
    async def custom_swagger_ui() -> HTMLResponse:
        """Serve Swagger UI with self-hosted assets to avoid CSP issues."""
        return get_swagger_ui_html(
            openapi_url="/api/v1/openapi.json",
            title=f"{app.title} - Swagger UI",
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
            swagger_favicon_url="/static/favicon.png",
        )

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        """Root endpoint."""
        return {
            "message": "CLARITY Digital Twin Platform API",
            "version": get_version(),
            "status": "operational",
        }

    # Health check
    @app.get("/health", tags=["Health"])
    async def health_check() -> dict[str, str]:
        """Basic health check endpoint."""
        container = get_container()
        status = "healthy" if container else "initializing"
        return {"status": status, "version": get_version()}

    return app


# Create application instance
app = create_app()


def get_app() -> FastAPI:
    """Get the FastAPI app instance (used by tests)."""
    return app


if __name__ == "__main__":
    import uvicorn

    # Unified CLI handling
    if "--validate" in sys.argv:
        mode = get_startup_mode()
        if mode == "bulletproof":
            asyncio.run(bulletproof_startup())
        else:
            standard_startup()
        sys.exit(0)

    # Run server
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    logger.info("Starting CLARITY on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)
