"""Tests for CLARITY startup orchestrator."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import Mock, patch

import pytest

from clarity.startup.config_schema import ClarityConfig, Environment
from clarity.startup.health_checks import HealthCheckResult, ServiceStatus
from clarity.startup.orchestrator import StartupError, StartupOrchestrator
from clarity.startup.progress_reporter import StartupProgressReporter


class TestStartupOrchestrator:
    """Test startup orchestrator functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.reporter = StartupProgressReporter(enable_colors=False)
        self.orchestrator = StartupOrchestrator(
            timeout=5.0,
            reporter=self.reporter,
        )

    @pytest.mark.asyncio
    async def test_successful_startup_development(self) -> None:
        """Test successful startup in development mode."""
        env_vars = {
            "ENVIRONMENT": "development",
            "SKIP_EXTERNAL_SERVICES": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            success, config = await self.orchestrator.orchestrate_startup()

            assert success is True
            assert config is not None
            assert config.environment == Environment.DEVELOPMENT
            assert config.should_use_mock_services() is True

    @pytest.mark.asyncio
    async def test_configuration_validation_failure(self) -> None:
        """Test startup failure due to configuration validation."""
        env_vars = {
            "ENVIRONMENT": "production",
            "ENABLE_AUTH": "true",
            # Missing required production settings
        }

        with patch.dict(os.environ, env_vars, clear=True):
            success, config = await self.orchestrator.orchestrate_startup()

            assert success is False
            assert config is None
            assert len(self.orchestrator.startup_errors) > 0

    @pytest.mark.asyncio
    async def test_service_health_check_failure(self) -> None:
        """Test startup failure due to service health check failures."""
        env_vars = {
            "ENVIRONMENT": "production",
            "SECRET_KEY": "production-secret-key-with-sufficient-length",
            "COGNITO_USER_POOL_ID": "us-east-1_validpool",
            "COGNITO_CLIENT_ID": "valid-client-id-12345678901234567890",
            "CORS_ALLOWED_ORIGINS": "https://app.example.com",
            "SKIP_EXTERNAL_SERVICES": "false",
        }

        # Mock health checker to return unhealthy services
        mock_health_results = {
            "cognito": HealthCheckResult(
                service_name="cognito",
                status=ServiceStatus.UNHEALTHY,
                message="Connection failed",
            ),
            "dynamodb": HealthCheckResult(
                service_name="dynamodb",
                status=ServiceStatus.UNHEALTHY,
                message="Table not found",
            ),
        }

        with (
            patch.dict(os.environ, env_vars, clear=True),
            patch.object(
                self.orchestrator.health_checker, "check_all_services"
            ) as mock_check,
            patch.object(
                self.orchestrator.health_checker, "get_overall_health"
            ) as mock_overall,
        ):

            mock_check.return_value = mock_health_results
            mock_overall.return_value = ServiceStatus.UNHEALTHY

            success, _config = await self.orchestrator.orchestrate_startup()

            assert success is False
            # When startup fails, config might be None or not based on implementation
            # The important thing is that success is False

    @pytest.mark.asyncio
    async def test_dry_run_mode(self) -> None:
        """Test dry-run mode functionality."""
        env_vars = {
            "ENVIRONMENT": "development",
        }

        dry_run_orchestrator = StartupOrchestrator(
            dry_run=True,
            reporter=self.reporter,
        )

        with patch.dict(os.environ, env_vars, clear=True):
            success, config = await dry_run_orchestrator.orchestrate_startup()

            assert success is True
            assert config is not None

            # Should be able to create dry-run report
            report = dry_run_orchestrator.create_dry_run_report()
            assert "Dry-Run Report" in report
            assert config.environment.value in report

    @pytest.mark.asyncio
    async def test_service_health_with_degraded_services(self) -> None:
        """Test startup with degraded services."""
        env_vars = {
            "ENVIRONMENT": "development",
            "SKIP_EXTERNAL_SERVICES": "false",
        }

        # Mock health checker to return degraded services
        mock_health_results = {
            "cognito": HealthCheckResult(
                service_name="cognito",
                status=ServiceStatus.DEGRADED,
                message="Slow response times",
            ),
            "dynamodb": HealthCheckResult(
                service_name="dynamodb",
                status=ServiceStatus.HEALTHY,
                message="Active",
            ),
        }

        with (
            patch.dict(os.environ, env_vars, clear=True),
            patch.object(
                self.orchestrator.health_checker, "check_all_services"
            ) as mock_check,
            patch.object(
                self.orchestrator.health_checker, "get_overall_health"
            ) as mock_overall,
        ):

            mock_check.return_value = mock_health_results
            mock_overall.return_value = ServiceStatus.DEGRADED

            success, config = await self.orchestrator.orchestrate_startup()

            # Should succeed with degraded services in development
            assert success is True
            assert config is not None

    @pytest.mark.asyncio
    async def test_startup_timeout(self) -> None:
        """Test startup timeout handling."""
        timeout_orchestrator = StartupOrchestrator(
            timeout=0.1,  # Very short timeout
            reporter=self.reporter,
        )

        env_vars = {
            "ENVIRONMENT": "development",
            "SKIP_EXTERNAL_SERVICES": "false",
        }

        # Mock a slow health check
        async def slow_health_check(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(1.0)  # Longer than timeout
            return {}

        with (
            patch.dict(os.environ, env_vars, clear=True),
            patch.object(
                timeout_orchestrator.health_checker,
                "check_all_services",
                side_effect=slow_health_check,
            ),
        ):

            success, _config = await timeout_orchestrator.orchestrate_startup()

            assert success is False

    @pytest.mark.asyncio
    async def test_skip_services(self) -> None:
        """Test skipping specific services during health checks."""
        env_vars = {
            "ENVIRONMENT": "development",
            "SKIP_EXTERNAL_SERVICES": "false",
        }

        skip_orchestrator = StartupOrchestrator(
            skip_services={"cognito", "s3"},
            reporter=self.reporter,
        )

        mock_health_results = {
            "dynamodb": HealthCheckResult(
                service_name="dynamodb",
                status=ServiceStatus.HEALTHY,
                message="Active",
            ),
        }

        with (
            patch.dict(os.environ, env_vars, clear=True),
            patch.object(
                skip_orchestrator.health_checker, "check_all_services"
            ) as mock_check,
            patch.object(
                skip_orchestrator.health_checker, "get_overall_health"
            ) as mock_overall,
        ):

            mock_check.return_value = mock_health_results
            mock_overall.return_value = ServiceStatus.HEALTHY

            success, config = await skip_orchestrator.orchestrate_startup()

            assert success is True
            assert config is not None

            # Verify skip_services was passed to health checker
            mock_check.assert_called_once()
            call_args = mock_check.call_args
            assert call_args[1]["skip_services"] == {"cognito", "s3"}

    def test_should_startup_succeed_prediction(self) -> None:
        """Test startup success prediction."""
        # No config loaded
        assert self.orchestrator.should_startup_succeed() is False

        # Config loaded but has errors
        self.orchestrator.startup_errors = ["Some error"]
        assert self.orchestrator.should_startup_succeed() is False

        # Create valid config
        env_vars = {
            "ENVIRONMENT": "development",
            "SKIP_EXTERNAL_SERVICES": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config, _errors = ClarityConfig.validate_from_env()
            self.orchestrator.config = config
            self.orchestrator.startup_errors = []

            # Should succeed with valid config and no errors
            assert self.orchestrator.should_startup_succeed() is True

            # Add unhealthy critical service
            self.orchestrator.health_results = {
                "cognito": HealthCheckResult(
                    service_name="cognito",
                    status=ServiceStatus.UNHEALTHY,
                    message="Failed",
                ),
            }

            # Should still succeed because we're using mock services
            assert self.orchestrator.should_startup_succeed() is True

    def test_production_validation_requirements(self) -> None:
        """Test production validation requirements."""
        env_vars = {
            "ENVIRONMENT": "production",
            "SECRET_KEY": "dev-secret-key",  # Invalid for production
            "CORS_ALLOWED_ORIGINS": "*",  # Invalid for production
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config, _errors = ClarityConfig.validate_from_env()

            if config:
                # Test production requirements validation
                prod_errors = self.orchestrator._validate_production_requirements()
                assert len(prod_errors) > 0
                assert any("SECRET_KEY" in error for error in prod_errors)

    @pytest.mark.asyncio
    async def test_service_initialization(self) -> None:
        """Test service initialization phase."""
        env_vars = {
            "ENVIRONMENT": "development",
            "SKIP_EXTERNAL_SERVICES": "true",
        }

        with (
            patch.dict(os.environ, env_vars, clear=True),
            patch("clarity.core.container_aws.initialize_container") as mock_init,
            patch(
                "clarity.core.config_adapter.clarity_config_to_settings"
            ) as mock_convert,
        ):

            mock_container = Mock()
            mock_init.return_value = mock_container
            mock_settings = Mock()
            mock_convert.return_value = mock_settings

            # Set up orchestrator with valid config
            config, _ = ClarityConfig.validate_from_env()
            self.orchestrator.config = config

            success = await self.orchestrator._initialize_services()

            assert success is True
            mock_convert.assert_called_once_with(config)
            mock_init.assert_called_once_with(mock_settings)

    @pytest.mark.asyncio
    async def test_service_initialization_failure(self) -> None:
        """Test service initialization failure handling."""
        env_vars = {
            "ENVIRONMENT": "development",
            "SKIP_EXTERNAL_SERVICES": "true",
        }

        with (
            patch.dict(os.environ, env_vars, clear=True),
            patch("clarity.core.container_aws.initialize_container") as mock_init,
        ):

            mock_init.side_effect = Exception("Container initialization failed")

            # Set up orchestrator with valid config
            config, _ = ClarityConfig.validate_from_env()
            self.orchestrator.config = config

            success = await self.orchestrator._initialize_services()

            assert success is False

    def test_create_dry_run_report(self) -> None:
        """Test dry-run report creation."""
        env_vars = {
            "ENVIRONMENT": "development",
            "AWS_REGION": "us-east-1",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config, _ = ClarityConfig.validate_from_env()
            self.orchestrator.config = config
            self.orchestrator.health_results = {
                "cognito": HealthCheckResult(
                    service_name="cognito",
                    status=ServiceStatus.SKIPPED,
                    message="Auth disabled",
                ),
            }

            report = self.orchestrator.create_dry_run_report()

            assert "Dry-Run Report" in report
            assert "development" in report
            assert "us-east-1" in report
            assert "cognito" in report
            assert "STARTUP SHOULD SUCCEED" in report or "All checks passed" in report


class TestStartupError:
    """Test startup error handling."""

    def test_startup_error_creation(self) -> None:
        """Test startup error creation with context."""
        error = StartupError(
            message="Configuration validation failed",
            phase="validation",
            details={
                "missing_vars": ["SECRET_KEY", "COGNITO_USER_POOL_ID"],
                "environment": "production",
            },
        )

        assert str(error) == "Configuration validation failed"
        assert error.phase == "validation"
        assert "SECRET_KEY" in error.details["missing_vars"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
