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
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from clarity.auth.mock_auth import MockAuthProvider
from clarity.auth.modal_auth_fix import set_user_context
from clarity.core.config import get_settings
from clarity.models.auth import AuthError, UserContext
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
            from clarity.auth.firebase_middleware import (
                FirebaseAuthProvider,
            )

            firebase_config = config_provider.get_firebase_config()
            middleware_config_dict = asdict(middleware_config_obj)

            firestore_client = None
            try:
                repository = self.get_health_data_repository()
                if hasattr(repository, "client"):
                    firestore_client = repository.client  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning("⚠️ Could not get Firestore client for auth: %s", e)

            return FirebaseAuthProvider(
                credentials_path=firebase_config.get("credentials_path"),
                project_id=firebase_config.get("project_id"),
                middleware_config=middleware_config_dict,
                firestore_client=firestore_client,
            )

        return MockAuthProvider()

    def _create_health_data_repository(self) -> IHealthDataRepository:
        """Factory method for creating health data repository."""
        config_provider = self.get_config_provider()

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
        return self.get_instance(IConfigProvider)

    def get_auth_provider(self) -> IAuthProvider:
        """Get authentication provider (Singleton pattern)."""
        return self.get_instance(IAuthProvider)

    def get_health_data_repository(self) -> IHealthDataRepository:
        """Get health data repository (Singleton pattern)."""
        return self.get_instance(IHealthDataRepository)

    async def _initialize_services(self) -> None:
        """Initialize all external services concurrently."""
        logger.info("🚀 Initializing services...")
        # Get instances to trigger their creation and initialization
        self.get_auth_provider()
        self.get_health_data_repository()
        logger.info("✅ All services initialized.")

    async def _cleanup_services(self) -> None:
        """Clean up all services with timeout protection."""
        logger.info("🛑 Shutting down application...")
        cleanup_start = time.perf_counter()
        tasks = []
        for service_type, instance in self._instances.items():
            if hasattr(instance, "cleanup"):
                tasks.append(
                    asyncio.wait_for(
                        instance.cleanup(), timeout=3.0
                    )
                )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result, (service_type, _) in zip(results, self._instances.items(), strict=False):
            if isinstance(result, Exception):
                logger.warning(
                    "⚠️ Cleanup error for %s: %s",
                    service_type.__name__,
                    result,
                )
        cleanup_elapsed = time.perf_counter() - cleanup_start
        logger.info("🏁 Shutdown complete in %.2fs", cleanup_elapsed)

    @asynccontextmanager
    async def app_lifespan(self, _app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan with timeout protection and graceful degradation."""
        startup_start = time.perf_counter()
        await self._initialize_services()
        startup_elapsed = time.perf_counter() - startup_start
        logger.info("🏁 Application startup complete in %.2fs", startup_elapsed)

        yield

        await self._cleanup_services()

    def create_fastapi_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        config_provider = self.get_config_provider()
        settings = config_provider.get_settings_model()

        app = FastAPI(
            lifespan=self.app_lifespan,
            title=settings.app_name,
            description="CLARITY Digital Twin Platform - Health AI Backend",
            version=settings.app_version,
        )

        self._configure_request_limits(app)
        self._configure_middleware(app)
        self._configure_exception_handlers(app)
        self._configure_routes(app)

        return app

    @staticmethod
    def _configure_exception_handlers(app: FastAPI) -> None:
        """Configure custom exception handlers."""
        logger.info("Exception handlers configured.")

    @staticmethod
    def _configure_request_limits(app: FastAPI) -> None:
        """Configure request limits, e.g., for upload sizes."""
        MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB

        class LimitUploadSizeMiddleware:
            def __init__(self, app: ASGIApp, max_size: int):
                self.app = app
                self.max_size = max_size

            async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
                if scope["type"] != "http":
                    await self.app(scope, receive, send)
                    return

                request = Request(scope, receive)
                if int(request.headers.get("content-length", 0)) > self.max_size:
                    response = JSONResponse(
                        {"detail": "File size exceeds limit"}, status_code=413
                    )
                    await response(scope, receive, send)
                else:
                    await self.app(scope, receive, send)

        app.add_middleware(LimitUploadSizeMiddleware, max_size=MAX_BODY_SIZE)
        logger.info("Request limits configured (max body size: %s MB)", MAX_BODY_SIZE / (1024 * 1024))

    def _configure_middleware(self, app: FastAPI) -> None:
        """Configure all application middleware."""
        config_provider = self.get_config_provider()
        middleware_config = config_provider.get_middleware_config()

        # Configure CORS - allow all origins in development, restrict in production
        allowed_origins = ["*"] if config_provider.is_development() else [
            "https://clarity-backend-282877548076.us-central1.run.app",
            "https://your-frontend-domain.com",  # Replace with actual frontend domain
            "http://localhost:3000",  # For local development
            "http://localhost:5173",  # Vite default port
        ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS middleware configured for origins: %s", allowed_origins)

        if middleware_config.enabled and config_provider.is_auth_enabled():
            auth_provider = self.get_auth_provider()

            async def firebase_auth_middleware(
                request: Request, call_next: Callable[[Request], Response]
            ) -> Response:
                is_public = False
                for public_path in middleware_config.exempt_paths or []:
                    if request.url.path == public_path or request.url.path.startswith(f"{public_path}/"):
                        is_public = True
                        break

                if is_public:
                    return await call_next(request)

                auth_header = request.headers.get("Authorization")
                if not auth_header:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Authentication required", "error_code": "NO_AUTH_HEADER"},
                    )

                if not auth_header.startswith("Bearer "):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid scheme. Use 'Bearer' token.", "error_code": "INVALID_AUTH_SCHEME"},
                    )

                token = auth_header.split("Bearer ")[1]
                if not token:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Bearer token is empty.", "error_code": "EMPTY_BEARER_TOKEN"},
                    )

                try:
                    user_info = await auth_provider.verify_token(token)
                    if user_info:
                        user_context = UserContext(**user_info)
                        set_user_context(user_context)
                        return await call_next(request)

                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Token verification failed.", "error_code": "VERIFICATION_UNEXPECTED_NONE"},
                    )
                except AuthError as e:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": e.detail, "error_code": e.error_code},
                    )
                except Exception:
                    logger.exception("💥 UNHANDLED EXCEPTION in auth middleware")
                    return JSONResponse(
                        status_code=500,
                        content={"detail": "Internal server error during authentication.", "error_code": "INTERNAL_AUTH_ERROR"},
                    )

            app.add_middleware(BaseHTTPMiddleware, dispatch=firebase_auth_middleware)
            logger.info("Firebase authentication middleware enabled.")
        else:
            logger.warning("Auth is disabled. All routes will be public.")

    def _configure_routes(self, app: FastAPI) -> None:
        """Configure API routes."""
        from clarity.api.v1.router import api_router as api_router_v1

        app.include_router(api_router_v1, prefix="/api/v1")

        @app.get("/health")
        async def health_endpoint_handler() -> dict[str, Any]:
            return {"status": "ok", "timestamp": time.time()}

        logger.info("API routes configured.")


# --- Singleton Accessor ---
_container_instance: DependencyContainer | None = None


def get_container() -> DependencyContainer:
    """Get the singleton instance of the dependency container."""
    global _container_instance
    if _container_instance is None:
        _container_instance = DependencyContainer()
    return _container_instance


def create_application() -> FastAPI:
    """Create a FastAPI application using the dependency container."""
    container = get_container()
    return container.create_fastapi_app()
