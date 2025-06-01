"""Dependency Injection Container - Clean Architecture Implementation.

Following Robert C. Martin's Clean Architecture and Gang of Four Factory Pattern.
This container manages all dependencies and wiring according to SOLID principles.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI

from clarity.core.config import get_settings
from clarity.core.interfaces import (
    IAuthProvider,
    IConfigProvider,
    IHealthDataRepository,
)


class DependencyContainer:
    """IoC Container following Gang of Four Factory Pattern.

    Manages all application dependencies according to Clean Architecture
    principles. Dependencies flow inward only.
    """

    def __init__(self) -> None:
        """Initialize dependency container."""
        self._instances: dict[type, Any] = {}
        self._settings = get_settings()

    def get_config_provider(self) -> IConfigProvider:
        """Get configuration provider (Singleton pattern)."""
        if IConfigProvider not in self._instances:
            from clarity.core.config_provider import ConfigProvider

            self._instances[IConfigProvider] = ConfigProvider(self._settings)
        return cast("IConfigProvider", self._instances[IConfigProvider])

    def get_auth_provider(self) -> IAuthProvider:
        """Get authentication provider (Singleton pattern)."""
        if IAuthProvider not in self._instances:
            # Only create if authentication is enabled
            config_provider = self.get_config_provider()
            if config_provider.get_setting("enable_auth", False):
                from clarity.auth.firebase_auth import FirebaseAuthProvider

                firebase_config = config_provider.get_firebase_config()
                self._instances[IAuthProvider] = FirebaseAuthProvider(
                    credentials_path=firebase_config.get("credentials_path"),
                    project_id=firebase_config.get("project_id"),
                )
            else:
                from clarity.auth.mock_auth import MockAuthProvider

                self._instances[IAuthProvider] = MockAuthProvider()
        return cast("IAuthProvider", self._instances[IAuthProvider])

    def get_health_data_repository(self) -> IHealthDataRepository:
        """Get health data repository (Singleton pattern)."""
        if IHealthDataRepository not in self._instances:
            config_provider = self.get_config_provider()

            # Use mock repository in development or when Firestore credentials aren't available
            if config_provider.is_development():
                from clarity.storage.mock_repository import MockHealthDataRepository

                self._instances[IHealthDataRepository] = MockHealthDataRepository()
            else:
                from clarity.storage.firestore_client import (
                    FirestoreHealthDataRepository,
                )

                self._instances[IHealthDataRepository] = FirestoreHealthDataRepository(
                    project_id=config_provider.get_gcp_project_id(),
                    credentials_path=config_provider.get_firebase_config().get(
                        "credentials_path"
                    ),
                )
        return cast("IHealthDataRepository", self._instances[IHealthDataRepository])

    @asynccontextmanager
    async def app_lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan context manager for FastAPI."""
        # Startup
        from clarity.core.logging_config import setup_logging

        setup_logging()

        # Initialize any async resources
        auth_provider = self.get_auth_provider()
        if hasattr(auth_provider, "initialize"):
            await auth_provider.initialize()

        repository = self.get_health_data_repository()
        if hasattr(repository, "initialize"):
            await repository.initialize()

        yield

        # Shutdown
        # Clean up resources
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
            from clarity.auth.firebase_auth import FirebaseAuthMiddleware

            auth_provider = self.get_auth_provider()
            app.add_middleware(FirebaseAuthMiddleware, auth_provider=auth_provider)

    def _configure_routes(self, app: FastAPI) -> None:
        """Configure API routes with dependency injection."""
        from clarity.api.v1 import health_data

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
    global _container
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
