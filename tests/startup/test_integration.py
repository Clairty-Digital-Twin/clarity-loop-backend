"""Integration tests for CLARITY startup system."""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
import subprocess  # noqa: S404
import sys
from typing import Any
from unittest.mock import patch

import pytest

from clarity.startup.config_schema import ClarityConfig
from clarity.startup.error_catalog import error_catalog
from clarity.startup.orchestrator import StartupOrchestrator
from clarity.startup.progress_reporter import StartupProgressReporter


class TestStartupIntegration:
    """Integration tests for the complete startup system."""

    @pytest.mark.asyncio
    async def test_full_startup_flow_development(self) -> None:
        """Test complete startup flow in development mode."""
        env_vars = {
            "ENVIRONMENT": "development",
            "SKIP_EXTERNAL_SERVICES": "true",
            "DEBUG": "true",
            "LOG_LEVEL": "INFO",
            "AWS_REGION": "us-east-1",
            "SECRET_KEY": "test-secret-key-for-development",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Use string output to capture progress
            output = io.StringIO()
            reporter = StartupProgressReporter(output=output, enable_colors=False)

            orchestrator = StartupOrchestrator(
                dry_run=False,
                timeout=10.0,
                reporter=reporter,
            )

            success, config = await orchestrator.orchestrate_startup()

            assert success is True
            assert config is not None
            assert config.is_development()
            assert config.should_use_mock_services()

            # Check progress output
            output_text = output.getvalue()
            assert "🚀 Starting" in output_text or "Starting" in output_text
            assert "Startup Complete" in output_text

    @pytest.mark.asyncio
    async def test_dry_run_flow_with_report(self) -> None:
        """Test dry-run flow with report generation."""
        env_vars = {
            "ENVIRONMENT": "production",
            "SECRET_KEY": "production-secret-key-with-sufficient-length",
            "COGNITO_USER_POOL_ID": "us-east-1_validpool123",
            "COGNITO_CLIENT_ID": "valid-client-id-12345678901234",
            "CORS_ALLOWED_ORIGINS": "https://app.example.com,https://api.example.com",
            "SKIP_EXTERNAL_SERVICES": "true",  # Skip for dry-run
        }

        with patch.dict(os.environ, env_vars, clear=True):
            output = io.StringIO()
            reporter = StartupProgressReporter(output=output, enable_colors=False)

            orchestrator = StartupOrchestrator(
                dry_run=True,
                timeout=10.0,
                reporter=reporter,
            )

            success, config = await orchestrator.orchestrate_startup()

            assert success is True
            assert config is not None
            assert config.is_production()

            # Generate and verify dry-run report
            report = orchestrator.create_dry_run_report()
            assert "Dry-Run Report" in report
            assert "production" in report
            assert "Configuration Summary" in report
            assert "Service Health Checks" in report
            assert "STARTUP SHOULD SUCCEED" in report

    @pytest.mark.asyncio
    async def test_configuration_error_handling(self) -> None:
        """Test error handling for configuration issues."""
        env_vars = {
            "ENVIRONMENT": "production",
            "ENABLE_AUTH": "true",
            "PORT": "invalid-port",  # Invalid value
            # Missing required production settings
        }

        with patch.dict(os.environ, env_vars, clear=True):
            output = io.StringIO()
            reporter = StartupProgressReporter(output=output, enable_colors=False)

            orchestrator = StartupOrchestrator(
                dry_run=True,
                timeout=10.0,
                reporter=reporter,
            )

            success, config = await orchestrator.orchestrate_startup()

            assert success is False
            assert config is None
            assert len(orchestrator.startup_errors) > 0

            # Check that errors are captured
            error_text = " ".join(orchestrator.startup_errors)
            assert (
                "production" in error_text.lower()
                or "configuration" in error_text.lower()
            )

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        """Test timeout handling during startup."""
        env_vars = {
            "ENVIRONMENT": "development",
            "SKIP_EXTERNAL_SERVICES": "false",
        }

        # Mock a very slow health check
        async def slow_check(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(2.0)  # Longer than timeout
            return {}

        with patch.dict(os.environ, env_vars, clear=True):
            output = io.StringIO()
            reporter = StartupProgressReporter(output=output, enable_colors=False)

            orchestrator = StartupOrchestrator(
                dry_run=False,
                timeout=0.5,  # Very short timeout
                reporter=reporter,
            )

            # Mock the health checker to be slow
            with patch.object(
                orchestrator.health_checker,
                "check_all_services",
                side_effect=slow_check,
            ):
                success, _config = await orchestrator.orchestrate_startup()

                assert success is False

                # Check output contains timeout message
                output_text = output.getvalue()
                assert (
                    "timed out" in output_text.lower()
                    or "timeout" in output_text.lower()
                )

    def test_cli_script_execution(self) -> None:
        """Test CLI script can be executed."""
        # Test the startup validator script
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "startup_validator.py"
        )

        # Verify script exists and is executable
        assert script_path.exists()

        # Test import works

        # Run script with --help to test basic functionality
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        # Should not crash and should show help
        assert (
            result.returncode == 0
            or "usage:" in result.stdout.lower()
            or "help" in result.stdout.lower()
        )

    @pytest.mark.asyncio
    async def test_error_catalog_integration(self) -> None:
        """Test error catalog integration with startup system."""
        # Test error code suggestion
        error_message = "Cognito credentials not found"
        suggested_code = error_catalog.suggest_error_code(error_message)
        assert suggested_code is not None

        # Test error help formatting
        if suggested_code:
            help_text = error_catalog.format_error_help(suggested_code)
            assert "🚨" in help_text or "Description:" in help_text
            assert "Solutions:" in help_text

    @pytest.mark.asyncio
    async def test_bulletproof_guarantees(self) -> None:
        """Test bulletproof startup guarantees."""
        # Test various failure scenarios to ensure they don't crash

        failure_scenarios = [
            {
                "name": "invalid_environment_variable",
                "env": {"ENVIRONMENT": "invalid-env"},
            },
            {
                "name": "missing_required_production_vars",
                "env": {
                    "ENVIRONMENT": "production",
                    "ENABLE_AUTH": "true",
                },
            },
            {
                "name": "invalid_aws_region",
                "env": {
                    "AWS_REGION": "invalid-region-format",
                },
            },
            {
                "name": "invalid_port",
                "env": {
                    "PORT": "99999",
                },
            },
        ]

        for scenario in failure_scenarios:
            with patch.dict(os.environ, scenario["env"], clear=True):
                output = io.StringIO()
                reporter = StartupProgressReporter(output=output, enable_colors=False)

                orchestrator = StartupOrchestrator(
                    dry_run=True,
                    timeout=5.0,
                    reporter=reporter,
                )

                # Should never raise unhandled exceptions
                try:
                    success, config = await orchestrator.orchestrate_startup()

                    # May succeed or fail, but should never crash
                    assert isinstance(success, bool)

                    if not success:
                        # Should have captured the error
                        assert len(orchestrator.startup_errors) > 0 or config is None

                except Exception as e:  # noqa: BLE001
                    pytest.fail(
                        f"Scenario '{scenario['name']}' crashed with unhandled exception: {e}"
                    )

    @pytest.mark.asyncio
    async def test_progress_reporting_completeness(self) -> None:
        """Test that progress reporting covers all startup phases."""
        env_vars = {
            "ENVIRONMENT": "development",
            "SKIP_EXTERNAL_SERVICES": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            output = io.StringIO()
            reporter = StartupProgressReporter(output=output, enable_colors=False)

            orchestrator = StartupOrchestrator(
                dry_run=True,
                timeout=10.0,
                reporter=reporter,
            )

            success, _config = await orchestrator.orchestrate_startup()

            assert success is True

            output_text = output.getvalue()

            # Check that key phases are reported
            expected_phases = [
                "Starting",
                "Validating",
                "Configuration",
                "Complete" if success else "Failed",
            ]

            for phase in expected_phases:
                assert (
                    phase in output_text
                ), f"Expected phase '{phase}' not found in output"

    def test_startup_summary_accuracy(self) -> None:
        """Test startup summary provides accurate information."""
        env_vars = {
            "ENVIRONMENT": "development",
            "AWS_REGION": "us-east-1",
            "DEBUG": "true",
            "ENABLE_AUTH": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            output = io.StringIO()
            reporter = StartupProgressReporter(output=output, enable_colors=False)

            orchestrator = StartupOrchestrator(
                dry_run=True,
                timeout=10.0,
                reporter=reporter,
            )

            # Manually load config to test summary
            config, _ = ClarityConfig.validate_from_env()
            orchestrator.config = config

            summary = config.get_startup_summary() if config else {}

            assert summary["environment"] == "development"
            assert summary["aws_region"] == "us-east-1"
            assert summary["auth_enabled"] is False
            assert summary["mock_services"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
