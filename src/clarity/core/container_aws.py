"""Dependency Injection Container for AWS deployment.

This module configures all AWS service dependencies and their initialization.
"""

# removed - breaks FastAPI

import logging
from typing import Any

from fastapi import FastAPI
from prometheus_client import Counter, Histogram

from clarity.auth.aws_auth_provider import CognitoAuthProvider
from clarity.auth.mock_auth import MockAuthProvider
from clarity.core.config_aws import Settings, get_settings

# AWS container - using settings directly
from clarity.core.exceptions import ConfigurationError
from clarity.core.logging_config import setup_logging

# Port types are imported from their respective modules
from clarity.ml.gemini_service import GeminiService
from clarity.ports.auth_ports import IAuthProvider
from clarity.ports.data_ports import IHealthDataRepository
from clarity.storage.dynamodb_client import DynamoDBHealthDataRepository
from clarity.storage.mock_repository import MockHealthDataRepository

logger = logging.getLogger(__name__)

# Metrics
service_initialization_counter = Counter(
    "service_initialization_total",
    "Total number of service initialization attempts",
    ["service", "status"],
)

service_initialization_duration = Histogram(
    "service_initialization_duration_seconds",
    "Time spent initializing services",
    ["service"],
)


class DependencyContainer:
    """AWS Dependency Injection Container.

    Manages initialization and lifecycle of all AWS service dependencies.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the dependency container with AWS settings."""
        self.settings = settings or get_settings()
        setup_logging()

        # Initialize service containers
        # Using settings directly
        self._auth_provider: IAuthProvider | None = None
        self._health_data_repository: IHealthDataRepository | None = None
        self._gemini_service: GeminiService | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all AWS services with proper error handling."""
        if self._initialized:
            return

        logger.info("Initializing AWS dependency container...")

        try:
            # Initialize configuration provider
            # Using settings directly

            # Initialize auth provider (Cognito or Mock)
            await self._initialize_auth_provider()

            # Initialize data repository (DynamoDB or Mock)
            await self._initialize_repository()

            # Initialize Gemini service (keeping this for AI functionality)
            await self._initialize_gemini_service()

            self._initialized = True
            logger.info("AWS dependency container initialized successfully")

        except Exception as e:
            logger.exception("Failed to initialize container")
            msg = f"Container initialization failed: {e!s}"
            raise ConfigurationError(msg) from e

    async def _initialize_auth_provider(self) -> None:
        """Initialize AWS Cognito auth provider with fallback to mock."""
        service_name = "auth_provider"

        with service_initialization_duration.labels(service=service_name).time():
            # Check configuration before try block
            if self.settings.should_use_mock_services():
                logger.info("Using mock auth provider (skip_external_services=True)")
                self._auth_provider = MockAuthProvider()
                service_initialization_counter.labels(
                    service=service_name, status="mock"
                ).inc()
                return

            if (
                not self.settings.cognito_user_pool_id
                or not self.settings.cognito_client_id
            ):
                if self.settings.is_development():
                    logger.warning("Cognito not configured, using mock auth provider")
                    self._auth_provider = MockAuthProvider()
                    service_initialization_counter.labels(
                        service=service_name, status="mock"
                    ).inc()
                    return
                msg = "Cognito configuration missing in production"
                raise ConfigurationError(msg)

            try:
                # Initialize Cognito auth provider
                self._auth_provider = CognitoAuthProvider(
                    region=self.settings.cognito_region or self.settings.aws_region,
                    user_pool_id=self.settings.cognito_user_pool_id,
                    client_id=self.settings.cognito_client_id,
                )

                logger.info("AWS Cognito auth provider initialized")
                service_initialization_counter.labels(
                    service=service_name, status="success"
                ).inc()

            except Exception:
                logger.exception("Failed to initialize Cognito")
                service_initialization_counter.labels(
                    service=service_name, status="error"
                ).inc()

                if self.settings.is_development():
                    logger.warning("Falling back to mock auth provider")
                    self._auth_provider = MockAuthProvider()
                    service_initialization_counter.labels(
                        service=service_name, status="fallback"
                    ).inc()
                else:
                    raise

    async def _initialize_repository(self) -> None:
        """Initialize DynamoDB repository with fallback to mock."""
        service_name = "health_data_repository"

        with service_initialization_duration.labels(service=service_name).time():
            try:
                if self.settings.should_use_mock_services():
                    logger.info("Using mock repository (skip_external_services=True)")
                    self._health_data_repository = MockHealthDataRepository()
                    service_initialization_counter.labels(
                        service=service_name, status="mock"
                    ).inc()
                    return

                # Initialize DynamoDB repository
                self._health_data_repository = DynamoDBHealthDataRepository(
                    table_name=self.settings.dynamodb_table_name,
                    region=self.settings.aws_region,
                    endpoint_url=self.settings.dynamodb_endpoint_url,  # For local testing
                )

                logger.info("DynamoDB repository initialized")
                service_initialization_counter.labels(
                    service=service_name, status="success"
                ).inc()

            except Exception:
                logger.exception("Failed to initialize DynamoDB")
                service_initialization_counter.labels(
                    service=service_name, status="error"
                ).inc()

                if self.settings.is_development():
                    logger.warning("Falling back to mock repository")
                    self._health_data_repository = MockHealthDataRepository()
                    service_initialization_counter.labels(
                        service=service_name, status="fallback"
                    ).inc()
                else:
                    raise

    async def _initialize_gemini_service(self) -> None:
        """Initialize Gemini AI service."""
        service_name = "gemini_service"

        with service_initialization_duration.labels(service=service_name).time():
            try:
                # Create enterprise Gemini service (Vertex AI)
                self._gemini_service = GeminiService(
                    project_id=getattr(self.settings, "gcp_project_id", None),
                    location=getattr(
                        self.settings, "vertex_ai_location", "us-central1"
                    ),
                    testing=self.settings.is_development(),
                )

                # Initialize the service (this handles Vertex AI setup)
                await self._gemini_service.initialize()

                logger.info("Enterprise Gemini service (Vertex AI) initialized")
                service_initialization_counter.labels(
                    service=service_name, status="success"
                ).inc()

            except Exception as e:
                logger.exception("Failed to initialize Gemini service")
                service_initialization_counter.labels(
                    service=service_name, status="error"
                ).inc()

                # Always continue without Gemini service - it's not critical for core functionality
                logger.warning(
                    "Continuing without Gemini service. AI features will be unavailable. Error: %s",
                    str(e),
                )
                # Set to None so AI endpoints can handle gracefully
                self._gemini_service = None

    async def shutdown(self) -> None:
        """Gracefully shutdown all services."""
        logger.info("Shutting down AWS dependency container...")

        # Add any cleanup logic here
        # For example, closing database connections, flushing queues, etc.

        self._initialized = False
        logger.info("AWS dependency container shutdown complete")

    def configure_routes(self, app: FastAPI) -> None:
        """Configure FastAPI routes with AWS dependencies.

        Note: This method is currently not used as route configuration
        is handled directly in main.py to avoid circular imports.
        """
        if not self._initialized:
            msg = "Container must be initialized before configuring routes"
            raise RuntimeError(msg)

        # Routes are configured in main.py to avoid circular imports
        logger.warning("configure_routes called but routes are configured in main.py")

        # Add health check endpoint
        @app.get("/health")
        async def health_check() -> dict[str, Any]:
            """Health check endpoint."""
            return {
                "status": "ok",
                "timestamp": self.settings.startup_timeout,
                "environment": self.settings.environment,
                "services": {
                    "auth": (
                        "cognito"
                        if isinstance(self._auth_provider, CognitoAuthProvider)
                        else "mock"
                    ),
                    "database": (
                        "dynamodb"
                        if isinstance(
                            self._health_data_repository, DynamoDBHealthDataRepository
                        )
                        else "mock"
                    ),
                    "ai": "gemini" if self._gemini_service else "disabled",
                },
            }

    # Property accessors for services
    # AWS-native implementation

    @property
    def auth_provider(self) -> IAuthProvider:
        """Get auth provider."""
        if not self._auth_provider:
            msg = "Auth provider not initialized"
            raise RuntimeError(msg)
        return self._auth_provider

    @property
    def health_data_repository(self) -> IHealthDataRepository:
        """Get health data repository."""
        if not self._health_data_repository:
            msg = "Health data repository not initialized"
            raise RuntimeError(msg)
        return self._health_data_repository

    @property
    def gemini_service(self) -> GeminiService | None:
        """Get Gemini service (may be None if not configured)."""
        return self._gemini_service


# Global container instance
_container: DependencyContainer | None = None


def get_container() -> DependencyContainer:
    """Get the global dependency container instance."""
    global _container  # noqa: PLW0603 - Singleton pattern for dependency injection container
    if _container is None:
        _container = DependencyContainer()
    return _container


async def initialize_container(settings: Settings | None = None) -> DependencyContainer:
    """Initialize and return the global container."""
    global _container  # noqa: PLW0603 - Singleton pattern for dependency injection container
    _container = DependencyContainer(settings)
    await _container.initialize()
    return _container
