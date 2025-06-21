"""Comprehensive tests for AWS dependency injection container.

Tests critical production infrastructure including service initialization,
fallback strategies, error handling, and dependency management.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
import pytest

from clarity.auth.aws_auth_provider import CognitoAuthProvider
from clarity.auth.mock_auth import MockAuthProvider
from clarity.core.config_aws import Settings
import clarity.core.container_aws
from clarity.core.container_aws import (
    DependencyContainer,
    get_container,
    initialize_container,
)
from clarity.core.exceptions import ConfigurationError
from clarity.ml.gemini_service import GeminiService
from clarity.storage.dynamodb_client import DynamoDBHealthDataRepository
from clarity.storage.mock_repository import MockHealthDataRepository


class TestDependencyContainerInitialization:
    """Test dependency container initialization."""

    def test_dependency_container_basic_initialization(self):
        """Test basic container initialization without services."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = True

        container = DependencyContainer(settings)

        assert container.settings == settings
        assert not container._initialized
        assert container._auth_provider is None
        assert container._health_data_repository is None
        assert container._gemini_service is None

    def test_dependency_container_uses_default_settings(self):
        """Test container uses default settings when none provided."""
        with (
            patch("clarity.core.container_aws.get_settings") as mock_get_settings,
            patch("clarity.core.container_aws.setup_logging"),
        ):
            mock_settings = Mock(spec=Settings)
            mock_get_settings.return_value = mock_settings

            container = DependencyContainer()

            assert container.settings == mock_settings
            mock_get_settings.assert_called_once()

    @pytest.mark.asyncio
    async def test_container_initialize_success(self):
        """Test successful container initialization."""
        settings = Mock(spec=Settings)
        container = DependencyContainer(settings)

        with (
            patch.object(container, "_initialize_auth_provider") as mock_auth,
            patch.object(container, "_initialize_repository") as mock_repo,
            patch.object(container, "_initialize_gemini_service") as mock_gemini,
        ):
            await container.initialize()

            assert container._initialized is True
            mock_auth.assert_called_once()
            mock_repo.assert_called_once()
            mock_gemini.assert_called_once()

    @pytest.mark.asyncio
    async def test_container_initialize_idempotent(self):
        """Test that initialize is idempotent."""
        settings = Mock(spec=Settings)
        container = DependencyContainer(settings)
        container._initialized = True

        with patch.object(container, "_initialize_auth_provider") as mock_auth:
            await container.initialize()

            # Should not call initialization methods again
            mock_auth.assert_not_called()

    @pytest.mark.asyncio
    async def test_container_initialize_failure(self):
        """Test container initialization failure handling."""
        settings = Mock(spec=Settings)
        container = DependencyContainer(settings)

        with patch.object(
            container,
            "_initialize_auth_provider",
            side_effect=Exception("Auth init failed"),
        ):
            with pytest.raises(
                ConfigurationError, match="Container initialization failed"
            ):
                await container.initialize()

            assert not container._initialized


class TestAuthProviderInitialization:
    """Test auth provider initialization strategies."""

    @pytest.mark.asyncio
    async def test_initialize_auth_provider_mock_services(self):
        """Test auth provider initialization with mock services enabled."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = True

        container = DependencyContainer(settings)

        await container._initialize_auth_provider()

        assert isinstance(container._auth_provider, MockAuthProvider)

    @pytest.mark.asyncio
    async def test_initialize_auth_provider_cognito_success(self):
        """Test successful Cognito auth provider initialization."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = False
        settings.cognito_user_pool_id = "us-east-1_ABC123"
        settings.cognito_client_id = "client123"
        settings.cognito_region = "us-east-1"
        settings.aws_region = "us-east-1"

        container = DependencyContainer(settings)

        with patch("clarity.core.container_aws.CognitoAuthProvider") as mock_cognito:
            mock_auth_instance = Mock(spec=CognitoAuthProvider)
            mock_cognito.return_value = mock_auth_instance

            await container._initialize_auth_provider()

            mock_cognito.assert_called_once_with(
                region="us-east-1",
                user_pool_id="us-east-1_ABC123",
                client_id="client123",
            )
            assert container._auth_provider == mock_auth_instance

    @pytest.mark.asyncio
    async def test_initialize_auth_provider_missing_config_development(self):
        """Test auth provider initialization with missing config in development."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = False
        settings.cognito_user_pool_id = None
        settings.cognito_client_id = "client123"
        settings.is_development.return_value = True

        container = DependencyContainer(settings)

        await container._initialize_auth_provider()

        assert isinstance(container._auth_provider, MockAuthProvider)

    @pytest.mark.asyncio
    async def test_initialize_auth_provider_missing_config_production(self):
        """Test auth provider initialization with missing config in production."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = False
        settings.cognito_user_pool_id = None
        settings.cognito_client_id = "client123"
        settings.is_development.return_value = False

        container = DependencyContainer(settings)

        with pytest.raises(
            ConfigurationError, match="Cognito configuration missing in production"
        ):
            await container._initialize_auth_provider()

    @pytest.mark.asyncio
    async def test_initialize_auth_provider_cognito_failure_fallback(self):
        """Test Cognito failure with fallback to mock in development."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = False
        settings.cognito_user_pool_id = "us-east-1_ABC123"
        settings.cognito_client_id = "client123"
        settings.cognito_region = "us-east-1"
        settings.aws_region = "us-east-1"
        settings.is_development.return_value = True

        container = DependencyContainer(settings)

        with patch(
            "clarity.core.container_aws.CognitoAuthProvider",
            side_effect=Exception("Cognito init failed"),
        ):
            await container._initialize_auth_provider()

            assert isinstance(container._auth_provider, MockAuthProvider)

    @pytest.mark.asyncio
    async def test_initialize_auth_provider_cognito_failure_production(self):
        """Test Cognito failure in production raises exception."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = False
        settings.cognito_user_pool_id = "us-east-1_ABC123"
        settings.cognito_client_id = "client123"
        settings.cognito_region = "us-east-1"
        settings.aws_region = "us-east-1"
        settings.is_development.return_value = False

        container = DependencyContainer(settings)

        with (
            patch(
                "clarity.core.container_aws.CognitoAuthProvider",
                side_effect=Exception("Cognito init failed"),
            ),
            pytest.raises(Exception, match="Cognito init failed"),
        ):
            await container._initialize_auth_provider()


class TestRepositoryInitialization:
    """Test health data repository initialization strategies."""

    @pytest.mark.asyncio
    async def test_initialize_repository_mock_services(self):
        """Test repository initialization with mock services enabled."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = True

        container = DependencyContainer(settings)

        await container._initialize_repository()

        assert isinstance(container._health_data_repository, MockHealthDataRepository)

    @pytest.mark.asyncio
    async def test_initialize_repository_dynamodb_success(self):
        """Test successful DynamoDB repository initialization."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = False
        settings.dynamodb_table_name = "health-data-table"
        settings.aws_region = "us-east-1"
        settings.dynamodb_endpoint_url = None

        container = DependencyContainer(settings)

        with patch(
            "clarity.core.container_aws.DynamoDBHealthDataRepository"
        ) as mock_dynamo:
            mock_repo_instance = Mock(spec=DynamoDBHealthDataRepository)
            mock_dynamo.return_value = mock_repo_instance

            await container._initialize_repository()

            mock_dynamo.assert_called_once_with(
                table_name="health-data-table", region="us-east-1", endpoint_url=None
            )
            assert container._health_data_repository == mock_repo_instance

    @pytest.mark.asyncio
    async def test_initialize_repository_dynamodb_failure_fallback(self):
        """Test DynamoDB failure with fallback to mock in development."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = False
        settings.dynamodb_table_name = "health-data-table"
        settings.aws_region = "us-east-1"
        settings.dynamodb_endpoint_url = None
        settings.is_development.return_value = True

        container = DependencyContainer(settings)

        with patch(
            "clarity.core.container_aws.DynamoDBHealthDataRepository",
            side_effect=Exception("DynamoDB init failed"),
        ):
            await container._initialize_repository()

            assert isinstance(
                container._health_data_repository, MockHealthDataRepository
            )

    @pytest.mark.asyncio
    async def test_initialize_repository_dynamodb_failure_production(self):
        """Test DynamoDB failure in production raises exception."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = False
        settings.dynamodb_table_name = "health-data-table"
        settings.aws_region = "us-east-1"
        settings.dynamodb_endpoint_url = None
        settings.is_development.return_value = False

        container = DependencyContainer(settings)

        with (
            patch(
                "clarity.core.container_aws.DynamoDBHealthDataRepository",
                side_effect=Exception("DynamoDB init failed"),
            ),
            pytest.raises(Exception, match="DynamoDB init failed"),
        ):
            await container._initialize_repository()


class TestGeminiServiceInitialization:
    """Test Gemini AI service initialization."""

    @pytest.mark.asyncio
    async def test_initialize_gemini_service_success(self):
        """Test successful Gemini service initialization."""
        settings = Mock(spec=Settings)
        settings.is_development.return_value = True

        container = DependencyContainer(settings)

        with patch("clarity.core.container_aws.GeminiService") as mock_gemini:
            mock_gemini_instance = Mock(spec=GeminiService)
            mock_gemini_instance.initialize = AsyncMock()
            mock_gemini.return_value = mock_gemini_instance

            await container._initialize_gemini_service()

            mock_gemini.assert_called_once()
            mock_gemini_instance.initialize.assert_called_once()
            assert container._gemini_service == mock_gemini_instance

    @pytest.mark.asyncio
    async def test_initialize_gemini_service_failure_non_production(self):
        """Test Gemini service failure in non-production continues gracefully."""
        settings = Mock(spec=Settings)
        settings.is_production.return_value = False

        container = DependencyContainer(settings)

        with patch(
            "clarity.core.container_aws.GeminiService",
            side_effect=Exception("Gemini init failed"),
        ):
            await container._initialize_gemini_service()

            assert container._gemini_service is None

    @pytest.mark.asyncio
    async def test_initialize_gemini_service_failure_production(self):
        """Test Gemini service failure in production continues without service."""
        settings = Mock(spec=Settings)
        settings.is_production.return_value = True
        settings.is_development.return_value = False

        container = DependencyContainer(settings)

        with patch(
            "clarity.core.container_aws.GeminiService",
            side_effect=Exception("Gemini init failed"),
        ):
            # Should not raise exception, but set service to None
            await container._initialize_gemini_service()

            # Verify the service is set to None after failure
            assert container._gemini_service is None


class TestContainerPropertyAccessors:
    """Test container property accessors."""

    def test_auth_provider_property_success(self):
        """Test auth provider property when initialized."""
        container = DependencyContainer(Mock(spec=Settings))
        mock_auth = Mock(spec=MockAuthProvider)
        container._auth_provider = mock_auth

        assert container.auth_provider == mock_auth

    def test_auth_provider_property_not_initialized(self):
        """Test auth provider property when not initialized."""
        container = DependencyContainer(Mock(spec=Settings))

        with pytest.raises(RuntimeError, match="Auth provider not initialized"):
            _ = container.auth_provider

    def test_health_data_repository_property_success(self):
        """Test health data repository property when initialized."""
        container = DependencyContainer(Mock(spec=Settings))
        mock_repo = Mock(spec=MockHealthDataRepository)
        container._health_data_repository = mock_repo

        assert container.health_data_repository == mock_repo

    def test_health_data_repository_property_not_initialized(self):
        """Test health data repository property when not initialized."""
        container = DependencyContainer(Mock(spec=Settings))

        with pytest.raises(
            RuntimeError, match="Health data repository not initialized"
        ):
            _ = container.health_data_repository

    def test_gemini_service_property_success(self):
        """Test Gemini service property when initialized."""
        container = DependencyContainer(Mock(spec=Settings))
        mock_gemini = Mock(spec=GeminiService)
        container._gemini_service = mock_gemini

        assert container.gemini_service == mock_gemini

    def test_gemini_service_property_none(self):
        """Test Gemini service property when not initialized."""
        container = DependencyContainer(Mock(spec=Settings))

        assert container.gemini_service is None


class TestContainerShutdown:
    """Test container shutdown functionality."""

    @pytest.mark.asyncio
    async def test_container_shutdown(self):
        """Test container shutdown process."""
        container = DependencyContainer(Mock(spec=Settings))
        container._initialized = True

        await container.shutdown()

        assert not container._initialized


class TestContainerRouteConfiguration:
    """Test container route configuration."""

    def test_configure_routes_not_initialized(self):
        """Test route configuration when container not initialized."""
        container = DependencyContainer(Mock(spec=Settings))
        app = Mock(spec=FastAPI)

        with pytest.raises(RuntimeError, match="Container must be initialized"):
            container.configure_routes(app)

    def test_configure_routes_initialized(self):
        """Test route configuration when container is initialized."""
        container = DependencyContainer(Mock(spec=Settings))
        container._initialized = True
        app = Mock(spec=FastAPI)

        # Should not raise an exception
        container.configure_routes(app)


class TestGlobalContainerFunctions:
    """Test global container access functions."""

    def test_get_container_creates_new_instance(self):
        """Test get_container creates new instance when none exists."""
        # Clear global container
        clarity.core.container_aws._container = None

        with patch(
            "clarity.core.container_aws.DependencyContainer"
        ) as mock_container_class:
            mock_instance = Mock(spec=DependencyContainer)
            mock_container_class.return_value = mock_instance

            container = get_container()

            mock_container_class.assert_called_once()
            assert container == mock_instance

    def test_get_container_returns_existing_instance(self):
        """Test get_container returns existing instance."""
        # Set up existing container
        existing_container = Mock(spec=DependencyContainer)
        clarity.core.container_aws._container = existing_container

        container = get_container()

        assert container == existing_container

    @pytest.mark.asyncio
    async def test_initialize_container_function(self):
        """Test initialize_container function."""
        settings = Mock(spec=Settings)

        with patch(
            "clarity.core.container_aws.DependencyContainer"
        ) as mock_container_class:
            mock_instance = Mock(spec=DependencyContainer)
            mock_instance.initialize = AsyncMock()
            mock_container_class.return_value = mock_instance

            container = await initialize_container(settings)

            mock_container_class.assert_called_once_with(settings)
            mock_instance.initialize.assert_called_once()
            assert container == mock_instance


class TestProductionScenarios:
    """Test realistic production scenarios."""

    @pytest.mark.asyncio
    async def test_complete_production_initialization(self):
        """Test complete production initialization flow."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = False
        settings.cognito_user_pool_id = "us-east-1_ABC123"
        settings.cognito_client_id = "client123"
        settings.cognito_region = "us-east-1"
        settings.aws_region = "us-east-1"
        settings.dynamodb_table_name = "health-data-prod"
        settings.dynamodb_endpoint_url = None
        settings.is_development.return_value = False
        settings.is_production.return_value = True

        container = DependencyContainer(settings)

        with (
            patch("clarity.core.container_aws.CognitoAuthProvider") as mock_cognito,
            patch(
                "clarity.core.container_aws.DynamoDBHealthDataRepository"
            ) as mock_dynamo,
            patch("clarity.core.container_aws.GeminiService") as mock_gemini,
        ):
            mock_auth = Mock(spec=CognitoAuthProvider)
            mock_repo = Mock(spec=DynamoDBHealthDataRepository)
            mock_gemini_instance = Mock(spec=GeminiService)
            mock_gemini_instance.initialize = AsyncMock()

            mock_cognito.return_value = mock_auth
            mock_dynamo.return_value = mock_repo
            mock_gemini.return_value = mock_gemini_instance

            await container.initialize()

            assert container._initialized is True
            assert isinstance(container._auth_provider, type(mock_auth))
            assert isinstance(container._health_data_repository, type(mock_repo))
            assert isinstance(container._gemini_service, type(mock_gemini_instance))

    @pytest.mark.asyncio
    async def test_development_fallback_scenario(self):
        """Test development environment with fallback strategies."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = False
        settings.cognito_user_pool_id = None  # Missing config
        settings.cognito_client_id = "client123"
        settings.is_development.return_value = True
        settings.is_production.return_value = False
        settings.dynamodb_table_name = "health-data-dev"
        settings.aws_region = "us-east-1"
        settings.dynamodb_endpoint_url = "http://localhost:8000"

        container = DependencyContainer(settings)

        with (
            patch(
                "clarity.core.container_aws.DynamoDBHealthDataRepository",
                side_effect=Exception("Connection failed"),
            ),
            patch(
                "clarity.core.container_aws.GeminiService",
                side_effect=Exception("AI service unavailable"),
            ),
        ):
            await container.initialize()

            # Should fallback to mock services in development
            assert container._initialized is True
            assert isinstance(container._auth_provider, MockAuthProvider)
            assert isinstance(
                container._health_data_repository, MockHealthDataRepository
            )
            assert container._gemini_service is None

    @pytest.mark.asyncio
    async def test_production_failure_scenarios(self):
        """Test production environment failure handling."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = False
        settings.cognito_user_pool_id = None  # Missing config
        settings.cognito_client_id = "client123"
        settings.is_development.return_value = False
        settings.is_production.return_value = True

        container = DependencyContainer(settings)

        with pytest.raises(ConfigurationError, match="Container initialization failed"):
            await container.initialize()

        assert not container._initialized

    @pytest.mark.asyncio
    async def test_mixed_service_initialization(self):
        """Test mixed success/failure initialization scenarios."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = False
        settings.cognito_user_pool_id = "us-east-1_ABC123"
        settings.cognito_client_id = "client123"
        settings.cognito_region = "us-east-1"
        settings.aws_region = "us-east-1"
        settings.dynamodb_table_name = "health-data"
        settings.dynamodb_endpoint_url = None
        settings.is_development.return_value = True
        settings.is_production.return_value = False

        container = DependencyContainer(settings)

        with (
            patch("clarity.core.container_aws.CognitoAuthProvider") as mock_cognito,
            patch(
                "clarity.core.container_aws.DynamoDBHealthDataRepository",
                side_effect=Exception("DynamoDB unavailable"),
            ),
            patch(
                "clarity.core.container_aws.GeminiService",
                side_effect=Exception("Gemini unavailable"),
            ),
        ):
            mock_auth = Mock(spec=CognitoAuthProvider)
            mock_cognito.return_value = mock_auth

            await container.initialize()

            # Cognito should succeed, others should fallback/fail gracefully
            assert container._initialized is True
            assert isinstance(container._auth_provider, type(mock_auth))
            assert isinstance(
                container._health_data_repository, MockHealthDataRepository
            )
            assert container._gemini_service is None

    @pytest.mark.asyncio
    async def test_metrics_tracking_during_initialization(self):
        """Test that metrics are properly tracked during service initialization."""
        settings = Mock(spec=Settings)
        settings.should_use_mock_services.return_value = True

        container = DependencyContainer(settings)

        with (
            patch(
                "clarity.core.container_aws.service_initialization_counter"
            ) as mock_counter,
            patch(
                "clarity.core.container_aws.service_initialization_duration"
            ) as mock_duration,
        ):
            mock_duration.labels.return_value.time.return_value.__enter__ = Mock()
            mock_duration.labels.return_value.time.return_value.__exit__ = Mock()

            await container.initialize()

            # Should track metrics for all services
            assert mock_counter.labels.call_count >= 3  # auth, repo, gemini
            assert mock_duration.labels.call_count >= 3
