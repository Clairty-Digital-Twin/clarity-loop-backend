"""Dependency Injection Container - Clean Architecture Implementation.

Following Robert C. Martin's Clean Architecture and Gang of Four Factory Pattern.
This container manages all dependencies and wiring according to SOLID principles.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeVar, cast

# Only needed at runtime, not for type checking
if not TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from fastapi import FastAPI

from clarity.core.config import get_settings
from clarity.core.interfaces import (
    IAuthProvider,
    IConfigProvider,
    IHealthDataRepository,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

T = TypeVar("T")

# Factory type for creating instances
Factory = Callable[[], Any]


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

        if config_provider.get_setting("enable_auth", default=False):
            from clarity.auth.firebase_auth import FirebaseAuthProvider  # noqa: PLC0415

            firebase_config = config_provider.get_firebase_config()
            return FirebaseAuthProvider(
                credentials_path=firebase_config.get("credentials_path"),
                project_id=firebase_config.get("project_id"),
            )

        from clarity.auth.mock_auth import MockAuthProvider  # noqa: PLC0415

        return MockAuthProvider()

    def _create_health_data_repository(self) -> IHealthDataRepository:
        """Factory method for creating health data repository."""
        config_provider = self.get_config_provider()

        # Use mock repository in development or when Firestore credentials aren't available
        if config_provider.is_development():
            from clarity.storage.mock_repository import (
                MockHealthDataRepository,
            )

            return MockHealthDataRepository()

        from clarity.storage.firestore_client import (
            FirestoreHealthDataRepository,
        )

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
        return cast("IConfigProvider", self.get_instance(IConfigProvider))

    def get_auth_provider(self) -> IAuthProvider:
        """Get authentication provider (Singleton pattern)."""
        return cast("IAuthProvider", self.get_instance(IAuthProvider))

    def get_health_data_repository(self) -> IHealthDataRepository:
        """Get health data repository (Singleton pattern)."""
        return cast("IHealthDataRepository", self.get_instance(IHealthDataRepository))

    @asynccontextmanager
    async def app_lifespan(self, _app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan context manager for FastAPI."""
        # Startup
        from clarity.core.logging_config import setup_logging  # noqa: PLC0415

        setup_logging()

        # Initialize any async resources
        auth_provider = self.get_auth_provider()
        if hasattr(auth_provider, "initialize"):
            await auth_provider.initialize()

        repository = self.get_health_data_repository()
        if hasattr(repository, "initialize"):
            await repository.initialize()

        yield

        # Shutdown - Clean up resources
        for instance in self._instances.values():
            if hasattr(instance, "cleanup"):
                await instance.cleanup()

    def create_fastapi_app(self) -> FastAPI:
        """Factory method creates FastAPI app with all dependencies wired.

        This is the main Factory Pattern implementation that creates
        the complete application with proper dependency injection.
        """
        # Create FastAPI application with lifespan
        app = FastAPI(
            title="CLARITY Digital Twin Platform",
            description="Healthcare AI platform built with Clean Architecture",
            version="1.0.0",
            lifespan=self.app_lifespan,
        )

        # Wire middleware (Decorator Pattern)
        self._configure_middleware(app)

        # Wire routers with dependencies injected
        self._configure_routes(app)

        return app

    def _configure_middleware(self, app: FastAPI) -> None:
        """Configure middleware with dependency injection."""
        config_provider = self.get_config_provider()

        # Add authentication middleware if enabled
        if config_provider.is_auth_enabled():
            from clarity.auth.firebase_auth import (
                FirebaseAuthMiddleware,
            )

            auth_provider = self.get_auth_provider()
            app.add_middleware(FirebaseAuthMiddleware, auth_provider=auth_provider)

    def _configure_routes(self, app: FastAPI) -> None:
        """Configure API routes with dependency injection."""
        from clarity.api.v1 import health_data  # noqa: PLC0415

        # Inject dependencies into route modules
        health_data.set_dependencies(
            auth_provider=self.get_auth_provider(),
            repository=self.get_health_data_repository(),
            config_provider=self.get_config_provider(),
        )

        # Include routers
        app.include_router(health_data.router, prefix="/api/v1", tags=["health-data"])


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
