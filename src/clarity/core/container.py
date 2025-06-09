"""Dependency Injection Container - Clean Architecture Implementation.

Following Robert C. Martin's Clean Architecture and Gang of Four Factory Pattern.
This container manages all dependencies and wiring according to SOLID principles.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
import logging
import time
from typing import TYPE_CHECKING, Any, TypeVar, cast

# Only needed at runtime, not for type checking
if not TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from fastapi import FastAPI

from clarity.auth.mock_auth import MockAuthProvider
from clarity.core.config import get_settings
from clarity.ports.auth_ports import IAuthProvider
from clarity.ports.config_ports import IConfigProvider
from clarity.ports.data_ports import IHealthDataRepository
from clarity.storage.firestore_client import FirestoreHealthDataRepository
from clarity.storage.mock_repository import MockHealthDataRepository

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

T = TypeVar("T")

# Factory type for creating instances
Factory = Callable[[], Any]

# Configure logger
logger = logging.getLogger(__name__)


class DependencyContainer:
    """IoC Container following Gang of Four Factory Pattern.

    Manages all application dependencies according to Clean Architecture
    principles. Dependencies flow inward only.
    """

    def __init__(self) -> None:
        """Initialize dependency container."""
        self._instances: dict[type, Any] = {}
        self._factories: dict[type, Factory] = {}
        self._settings = get_settings()
        self._register_factories()

    def _register_factories(self) -> None:
        """Register factory functions for each interface (Factory Pattern)."""
        self._factories[IConfigProvider] = self._create_config_provider
        self._factories[IAuthProvider] = self._create_auth_provider
        self._factories[IHealthDataRepository] = self._create_health_data_repository

    def _create_config_provider(self) -> IConfigProvider:
        """Factory method for creating configuration provider."""
        from clarity.core.config_provider import ConfigProvider  # noqa: PLC0415

        return ConfigProvider(self._settings)

    def _create_auth_provider(self) -> IAuthProvider:
        """Factory method for creating authentication provider."""
        config_provider = self.get_config_provider()
        middleware_config_obj = config_provider.get_middleware_config()

        if middleware_config_obj.enabled and config_provider.get_setting(
            "enable_auth", default=False
        ):
            from clarity.auth.firebase_middleware import (  # noqa: PLC0415 - Local import to avoid circular dependency
                FirebaseAuthProvider,
            )

            firebase_config = config_provider.get_firebase_config()
            # Convert MiddlewareConfig object to dict before passing
            middleware_config_dict = asdict(middleware_config_obj)

            # Get Firestore client for enhanced auth functionality
            firestore_client = None
            try:
                # Try to get Firestore client from repository
                repository = self.get_health_data_repository()
                if hasattr(repository, "client"):
                    firestore_client = repository.client  # type: ignore[attr-defined]
                    logger.info("✅ Firestore client available for enhanced auth")
                else:
                    logger.warning("⚠️ Repository doesn't have Firestore client")
            except Exception as e:
                logger.warning("⚠️ Could not get Firestore client for auth: %s", e)

            return FirebaseAuthProvider(
                credentials_path=firebase_config.get("credentials_path"),
                project_id=firebase_config.get("project_id"),
                middleware_config=middleware_config_dict,  # Pass the dict
                firestore_client=firestore_client,  # Pass Firestore client
            )

        return MockAuthProvider()

    def _create_health_data_repository(self) -> IHealthDataRepository:
        """Factory method for creating health data repository."""
        config_provider = self.get_config_provider()

        # Use mock repository in development or when Firestore credentials aren't available
        if (
            config_provider.is_development()
            or config_provider.should_skip_external_services()
        ):
            return MockHealthDataRepository()

        return FirestoreHealthDataRepository(
            project_id=config_provider.get_gcp_project_id(),
            credentials_path=config_provider.get_firebase_config().get(
                "credentials_path"
            ),
        )

    def get_instance(self, interface: type[T]) -> T:
        """Get instance using Singleton pattern with lazy initialization."""
        if interface not in self._instances:
            if interface not in self._factories:
                msg = f"No factory registered for {interface.__name__}"
                raise ValueError(msg)

            self._instances[interface] = self._factories[interface]()

        return cast("T", self._instances[interface])

    def get_config_provider(self) -> IConfigProvider:
        """Get configuration provider (Singleton pattern)."""
        if IConfigProvider not in self._instances:
            self._instances[IConfigProvider] = self._factories[IConfigProvider]()
        return cast("IConfigProvider", self._instances[IConfigProvider])

    def get_auth_provider(self) -> IAuthProvider:
        """Get authentication provider (Singleton pattern)."""
        if IAuthProvider not in self._instances:
            self._instances[IAuthProvider] = self._factories[IAuthProvider]()
        return cast("IAuthProvider", self._instances[IAuthProvider])

    def get_health_data_repository(self) -> IHealthDataRepository:
        """Get health data repository (Singleton pattern)."""
        if IHealthDataRepository not in self._instances:
            self._instances[IHealthDataRepository] = self._factories[
                IHealthDataRepository
            ]()
        return cast("IHealthDataRepository", self._instances[IHealthDataRepository])

    async def _initialize_auth_provider(self) -> None:
        """Initialize authentication provider with timeout and fallback."""
        logger.info("🔐 Initializing authentication provider...")

        # Get middleware config for timeout settings
        config_provider = self.get_config_provider()
        middleware_config = config_provider.get_middleware_config()
        timeout = middleware_config.initialization_timeout_seconds

        try:
            auth_provider = self.get_auth_provider()
            logger.info("   • Provider type: %s", type(auth_provider).__name__)
            logger.info("   • Initialization timeout: %ss", timeout)

            if hasattr(auth_provider, "initialize"):
                await asyncio.wait_for(auth_provider.initialize(), timeout=timeout)
            logger.info("✅ Authentication provider ready")

        except TimeoutError:
            logger.exception("💥 Auth provider initialization TIMEOUT (%ss)", timeout)
            if middleware_config.fallback_to_mock:
                logger.warning("🔄 Falling back to mock auth provider...")
                self._instances[IAuthProvider] = MockAuthProvider()
                logger.info("✅ Mock auth provider activated")
            else:
                logger.exception(
                    "❌ Auth fallback disabled - authentication unavailable"
                )

        except Exception:
            logger.exception("💥 Auth provider initialization failed")
            if middleware_config.graceful_degradation:
                logger.warning("🔄 Falling back to mock auth provider...")
                self._instances[IAuthProvider] = MockAuthProvider()
                logger.info("✅ Mock auth provider activated")
            else:
                logger.exception(
                    "❌ Graceful degradation disabled - authentication unavailable"
                )

    async def _initialize_repository(self) -> None:
        """Initialize health data repository with timeout and fallback."""
        logger.info("🗄️ Initializing health data repository...")
        try:
            repository = self.get_health_data_repository()
            logger.info("   • Repository type: %s", type(repository).__name__)

            if hasattr(repository, "initialize"):
                await asyncio.wait_for(repository.initialize(), timeout=8.0)
            logger.info("✅ Health data repository ready")

        except TimeoutError:
            logger.exception("💥 Repository initialization TIMEOUT (8s)")
            logger.warning("🔄 Falling back to mock repository...")
            self._instances[IHealthDataRepository] = MockHealthDataRepository()
            logger.info("✅ Mock repository activated")

        except Exception:
            logger.exception("💥 Repository initialization failed")
            logger.warning("🔄 Falling back to mock repository...")
            self._instances[IHealthDataRepository] = MockHealthDataRepository()
            logger.info("✅ Mock repository activated")

    async def _cleanup_services(self) -> None:
        """Clean up all services with timeout protection."""
        logger.info("🛑 Shutting down application...")
        cleanup_start = time.perf_counter()

        for service_type, instance in self._instances.items():
            if hasattr(instance, "cleanup"):
                try:
                    await asyncio.wait_for(instance.cleanup(), timeout=3.0)
                    logger.debug("✅ Cleaned up %s", service_type.__name__)
                except TimeoutError:
                    logger.warning("⚠️ Cleanup timeout for %s", service_type.__name__)
                except (OSError, AttributeError, RuntimeError) as cleanup_error:
                    logger.warning(
                        "⚠️ Cleanup error for %s: %s",
                        service_type.__name__,
                        cleanup_error,
                    )

        cleanup_elapsed = time.perf_counter() - cleanup_start
        logger.info("🏁 Shutdown complete in %.2fs", cleanup_elapsed)

    @asynccontextmanager
    async def app_lifespan(self, _app: FastAPI) -> AsyncGenerator[None]:
        """Application lifespan with timeout protection and graceful degradation."""
        startup_timeout = 15.0
        startup_start = time.perf_counter()

        try:
            logger.info("🚀 Starting CLARITY Digital Twin Platform lifespan...")
            logger.info("🆕 Code revision: d76b185-force-rebuild-v3")

            # Step 1: Logging setup
            logger.info("📝 Setting up logging configuration...")
            from clarity.core.logging_config import setup_logging  # noqa: PLC0415

            setup_logging()
            logger.info("✅ Logging configuration complete")

            # Step 2: Configuration validation
            logger.info("⚙️ Validating configuration...")
            config_provider = self.get_config_provider()
            logger.info("✅ Configuration validated")
            logger.info(
                "   • Environment: %s",
                config_provider.get_setting("environment", default="unknown"),
            )
            logger.info("   • Development mode: %s", config_provider.is_development())
            logger.info("   • Auth enabled: %s", config_provider.is_auth_enabled())

            # Skip external services in development or when explicitly configured
            if config_provider.should_skip_external_services():
                logger.info(
                    "⚠️ Skipping external service initialization (development mode)"
                )
                yield
                return

            # Step 3: Initialize services
            await self._initialize_auth_provider()
            await self._initialize_repository()

            # Step 4: Startup completion
            elapsed = time.perf_counter() - startup_start
            logger.info("🎉 Startup complete in %.2fs", elapsed)

            if elapsed > startup_timeout * 0.8:
                logger.warning("⚠️ Slow startup detected (%.2fs)", elapsed)

            yield

        except Exception:
            elapsed = time.perf_counter() - startup_start
            logger.exception("💥 STARTUP FAILED after %.2fs", elapsed)
            logger.warning("🔄 Starting with minimal functionality...")

            try:
                self._instances[IAuthProvider] = MockAuthProvider()
                self._instances[IHealthDataRepository] = MockHealthDataRepository()
                logger.info("✅ Minimal providers activated")
            except Exception as fallback_error:
                logger.critical(
                    "💥 CRITICAL: Fallback initialization failed: %s", fallback_error
                )
                msg = "Complete startup failure"
                raise RuntimeError(msg) from fallback_error

            yield

        finally:
            await self._cleanup_services()

    def create_fastapi_app(self) -> FastAPI:
        """Factory method creates FastAPI app with all dependencies wired.

        This is the main Factory Pattern implementation that creates
        the complete application with proper dependency injection.
        """
        # Create FastAPI application WITH FIXED lifespan
        app = FastAPI(
            title="CLARITY Digital Twin Platform",
            description="Healthcare AI platform built with Clean Architecture",
            version="1.0.0",
            lifespan=self.app_lifespan,  # ✅ RE-ENABLED with proper timeout handling
        )

        # Configure RFC 7807 Problem Details exception handling
        self._configure_exception_handlers(app)

        # Configure request size limits for DoS protection
        self._configure_request_limits(app)

        # Wire middleware (Decorator Pattern)
        self._configure_middleware(app)

        # Wire routers with dependencies injected
        self._configure_routes(app)

        return app

    @staticmethod
    def _configure_exception_handlers(app: FastAPI) -> None:
        """Configure RFC 7807 Problem Details exception handling."""
        from clarity.core.exceptions import (  # noqa: PLC0415
            ClarityAPIException,
            generic_exception_handler,
            problem_detail_exception_handler,
        )

        # Register custom exception handler for ClarityAPIException
        # Cast to the expected handler type to resolve typing issues
        app.add_exception_handler(ClarityAPIException, problem_detail_exception_handler)  # type: ignore[arg-type]

        # Register generic exception handler for unhandled exceptions
        app.add_exception_handler(Exception, generic_exception_handler)  # type: ignore[arg-type]

        logger.info("✅ RFC 7807 Problem Details exception handling configured")

    @staticmethod
    def _configure_request_limits(app: FastAPI) -> None:
        """Configure request size limits to prevent DoS attacks."""
        from collections.abc import Awaitable  # noqa: PLC0415
        from typing import TYPE_CHECKING  # noqa: PLC0415

        from fastapi import Request, Response  # noqa: PLC0415
        from clarity.core.exceptions import ClarityAPIException  # noqa: PLC0415

        # Maximum request size: 10MB for health data uploads
        max_request_size = 10 * 1024 * 1024  # 10MB

        @app.middleware("http")
        async def limit_upload_size(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            """Middleware to limit request size and prevent DoS attacks."""
            if request.method in {"POST", "PUT", "PATCH"}:
                content_length = request.headers.get("content-length")
                if content_length and int(content_length) > max_request_size:
                    raise ClarityAPIException(
                        status_code=413,
                        problem_type="request_too_large",
                        title="Request Too Large",
                        detail=f"Request size {content_length} exceeds maximum {max_request_size} bytes",
                    )

            return await call_next(request)

        logger.info(
            "✅ Request size limits configured (max: %d MB)",
            max_request_size // (1024 * 1024),
        )

    def _configure_middleware(self, app: FastAPI) -> None:
        """Configure middleware with dependency injection."""
        config_provider = self.get_config_provider()
        middleware_config = config_provider.get_middleware_config()
        
        logger.warning("🔍 MIDDLEWARE CONFIG: enabled=%s", middleware_config.enabled)
        logger.warning("🔍 APP ID: %s", id(app))  # Track app instance

        # Add authentication middleware if enabled
        if middleware_config.enabled:
            auth_provider = self.get_auth_provider()
            exempt_paths = middleware_config.exempt_paths or [
                "/",
                "/health",
                "/docs",
                "/openapi.json",
                "/redoc",
                "/api/docs",
                "/api/health",
            ]

            # Create middleware function first
            async def firebase_auth_middleware(
                request: Request, call_next: Callable[[Request], Awaitable[Response]]
            ) -> Response:
                """Firebase authentication middleware using function-based approach."""
                from datetime import UTC, datetime  # noqa: PLC0415
                from clarity.models.auth import AuthError  # noqa: PLC0415
                from clarity.auth.firebase_middleware import FirebaseAuthMiddleware  # noqa: PLC0415
                from starlette.responses import JSONResponse  # noqa: PLC0415
                
                path = request.url.path
                logger.warning("🔥🔥 MIDDLEWARE ACTUALLY RUNNING: %s %s", request.method, path)
                logger.warning("🔥🔥 APP INSTANCE IN MIDDLEWARE: %s", id(app))
                
                # Check if path is exempt
                is_exempt = any(path.startswith(p) for p in exempt_paths)
                if is_exempt:
                    return await call_next(request)
                
                # Extract token
                auth_header = request.headers.get("authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    return JSONResponse(
                        status_code=401,
                        content={
                            "error": "missing_token",
                            "message": "Missing or invalid Authorization header",
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    )
                
                token = auth_header[7:]  # Remove "Bearer " prefix
                
                try:
                    # Verify token
                    user_info = await auth_provider.verify_token(token)
                    if not user_info:
                        return JSONResponse(
                            status_code=401,
                            content={
                                "error": "invalid_token",
                                "message": "Invalid or expired token",
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                    
                    # Create user context
                    if hasattr(auth_provider, 'get_or_create_user_context'):
                        user_context = await auth_provider.get_or_create_user_context(user_info)
                    else:
                        user_context = FirebaseAuthMiddleware._create_user_context(user_info)
                    
                    # Store user in request state
                    request.state.user = user_context
                    
                except AuthError as e:
                    return JSONResponse(
                        status_code=e.status_code,
                        content={
                            "error": e.error_code,
                            "message": e.message,
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    )
                except Exception as e:
                    logger.exception("Authentication failed: %s", e)
                    if not middleware_config.graceful_degradation:
                        return JSONResponse(
                            status_code=401,
                            content={
                                "error": "authentication_failed",
                                "message": "Authentication required",
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                    # For graceful degradation, set user as None
                    request.state.user = None
                
                return await call_next(request)
            
            # Register the middleware with the app
            app.middleware("http")(firebase_auth_middleware)

            logger.info("Firebase authentication middleware enabled")
            logger.info("   • Exempt paths: %s", exempt_paths)
            logger.info("   • Cache enabled: %s", middleware_config.cache_enabled)
            logger.info(
                "   • Graceful degradation: %s", middleware_config.graceful_degradation
            )
        else:
            logger.info("⚠️ Authentication middleware disabled in configuration")

    def _configure_routes(self, app: FastAPI) -> None:
        """Configure API routes with dependency injection."""

        # Add root-level health endpoint first (no auth required)
        # NOTE: Function appears unused but is registered by FastAPI @app.get decorator
        @app.get("/health")
        async def health_endpoint_handler() -> dict[str, Any]:
            """Root health check endpoint for application monitoring."""
            from datetime import UTC, datetime  # noqa: PLC0415

            return {
                "status": "healthy",
                "service": "clarity-digital-twin",
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0",
            }

        try:
            # Import the unified v1 router and individual modules for dependency injection
            from clarity.api.v1 import (  # noqa: PLC0415
                auth,
                gemini_insights,
                health_data,
            )
            from clarity.api.v1 import router as v1_router  # noqa: PLC0415
            from clarity.api.v1.debug import router as debug_router  # noqa: PLC0415

            # Get shared dependencies
            auth_provider = self.get_auth_provider()
            repository = self.get_health_data_repository()
            config_provider = self.get_config_provider()

            # Inject dependencies into route modules
            health_data.set_dependencies(
                auth_provider=auth_provider,
                repository=repository,
                config_provider=config_provider,
            )

            # Set up authentication endpoints with Firestore client if available
            firestore_client = None
            if hasattr(repository, "client"):  # type: ignore[misc]
                # Extract FirestoreClient from FirestoreHealthDataRepository
                firestore_client = repository.client  # type: ignore[attr-defined]

            auth.set_dependencies(
                auth_provider=auth_provider,
                repository=repository,
                firestore_client=firestore_client,  # type: ignore[arg-type]
            )

            # Inject dependencies into Gemini insights module
            gemini_insights.set_dependencies(
                auth_provider=auth_provider,
                config_provider=config_provider,
            )

            # 🔥 ADDED: Include metrics router for Prometheus monitoring
            from clarity.api.v1.metrics import router as metrics_router  # noqa: PLC0415

            app.include_router(metrics_router)

            # Include the unified v1 router (includes all endpoints: auth, health_data, pat_analysis, gemini_insights)
            app.include_router(v1_router)

            # Include debug router (REMOVE IN PRODUCTION!)
            app.include_router(debug_router, prefix="/api/v1")
            logger.warning("⚠️ DEBUG ENDPOINTS ENABLED - REMOVE IN PRODUCTION!")

            logger.info("✅ API routes configured")
            logger.info("   • Prometheus metrics: /metrics")
            logger.info("   • Debug endpoints: /api/v1/debug/*")
            logger.info("   • V1 API endpoints: /api/v1")
            logger.info("   • Authentication: /api/v1/auth")
            logger.info("   • Health data: /api/v1/health-data")
            logger.info("   • PAT analysis: /api/v1/pat")
            logger.info("   • Gemini insights: /api/v1/insights")

        except Exception:
            logger.exception("💥 Failed to configure routes")
            logger.info("🔄 API routes failed but root health endpoint still available")


# Global container instance (Singleton)
_container: DependencyContainer | None = None


def get_container() -> DependencyContainer:
    """Get global dependency container (Singleton pattern)."""
    global _container  # noqa: PLW0603
    if _container is None:
        _container = DependencyContainer()
    return _container


def create_application() -> FastAPI:
    """Factory function creates application using dependency injection.

    This is the main entry point that follows Clean Architecture principles.
    All dependencies are properly injected and no circular imports exist.
    """
    container = get_container()
    return container.create_fastapi_app()
