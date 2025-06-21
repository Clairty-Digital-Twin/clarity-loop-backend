"""CLARITY Startup Orchestrator.

Bulletproof startup system that orchestrates configuration validation,
service health checks, and application initialization with zero-crash guarantee.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from clarity.startup.config_schema import ClarityConfig
from clarity.startup.health_checks import ServiceHealthChecker, ServiceStatus
from clarity.startup.progress_reporter import ProgressPhase, StartupProgressReporter

logger = logging.getLogger(__name__)


class StartupError(Exception):
    """Startup-specific error with detailed context."""

    def __init__(
        self, message: str, phase: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.details = details or {}


class StartupOrchestrator:
    """Orchestrates bulletproof application startup.

    Provides zero-crash guarantee by validating all configuration and external
    services before attempting to start the application.
    """

    def __init__(
        self,
        *,
        dry_run: bool = False,
        skip_services: set[str] | None = None,
        timeout: float = 30.0,
        reporter: StartupProgressReporter | None = None,
    ) -> None:
        """Initialize startup orchestrator.

        Args:
            dry_run: If True, only validate configuration without starting services
            skip_services: Set of service names to skip during health checks
            timeout: Maximum startup time in seconds
            reporter: Progress reporter (creates default if not provided)
        """
        self.dry_run = dry_run
        self.skip_services = skip_services or set()
        self.timeout = timeout
        self.reporter = reporter or StartupProgressReporter()
        self.health_checker = ServiceHealthChecker()

        self.config: ClarityConfig | None = None
        self.health_results: dict[str, Any] = {}
        self.startup_errors: list[str] = []

    async def orchestrate_startup(
        self, app_name: str = "CLARITY Digital Twin"
    ) -> tuple[bool, ClarityConfig | None]:
        """Orchestrate complete startup process.

        Args:
            app_name: Application name for display

        Returns:
            Tuple of (success, config). Config is None if startup fails.
        """
        self.reporter.start_startup(app_name)

        try:
            # Phase 1: Pre-flight configuration validation
            success = await self._validate_configuration()
            if not success:
                self.reporter.report_startup_complete(
                    success=False, message="Configuration validation failed"
                )
                return False, None

            # Phase 2: Service health checks
            if not self.dry_run:
                success = await self._check_service_health()
                if not success:
                    self.reporter.report_startup_complete(
                        success=False, message="Service health checks failed"
                    )
                    return False, None

            # Phase 3: Initialize services (if not dry run)
            if not self.dry_run:
                success = await self._initialize_services()
                if not success:
                    self.reporter.report_startup_complete(
                        success=False, message="Service initialization failed"
                    )
                    return False, None

            # Success!
            if self.dry_run:
                self.reporter.report_startup_complete(
                    success=True, message="Dry-run completed successfully"
                )
            else:
                self.reporter.report_startup_complete(
                    success=True, message="All systems operational"
                )

            return True, self.config

        except TimeoutError:
            self.reporter.report_startup_complete(
                success=False, message=f"Startup timed out after {self.timeout}s"
            )
            return False, None
        except Exception as e:
            logger.exception("Unexpected startup error")
            self.reporter.report_startup_complete(
                success=False, message=f"Unexpected error: {e!s}"
            )
            return False, None

    async def _validate_configuration(self) -> bool:
        """Validate configuration with comprehensive error reporting."""
        self.reporter.start_phase(ProgressPhase.VALIDATING_CONFIG)

        # Load and validate configuration
        step = self.reporter.start_step("Loading configuration schema")

        try:
            self.config, validation_errors = ClarityConfig.validate_from_env()

            if validation_errors:
                self.reporter.fail_step(
                    step, f"Found {len(validation_errors)} configuration error(s)"
                )
                self.startup_errors.extend(validation_errors)

                # Report detailed validation errors
                self.reporter.report_config_validation(None, validation_errors)
                return False

            if self.config is None:
                self.reporter.fail_step(step, "Failed to create configuration object")
                return False

            self.reporter.complete_step(step, "Configuration loaded successfully")

            # Validate environment-specific requirements
            env_step = self.reporter.start_step("Validating environment requirements")

            try:
                # Additional validation based on environment
                if self.config.is_production():
                    prod_errors = self._validate_production_requirements()
                    if prod_errors:
                        self.reporter.fail_step(
                            env_step,
                            f"Production validation failed: {len(prod_errors)} error(s)",
                        )
                        self.startup_errors.extend(prod_errors)
                        return False

                self.reporter.complete_step(
                    env_step, f"Environment: {self.config.environment.value}"
                )

                # Report final configuration summary
                summary_step = self.reporter.start_step("Configuration summary")
                summary = self.config.get_startup_summary()
                self.reporter.complete_step(
                    summary_step, "Configuration valid", summary
                )

                return True

            except Exception as e:  # noqa: BLE001 - Bulletproof startup
                self.reporter.fail_step(env_step, "Environment validation failed", e)
                return False

        except Exception as e:  # noqa: BLE001 - Bulletproof startup
            self.reporter.fail_step(step, "Configuration loading failed", e)
            return False

    def _validate_production_requirements(self) -> list[str]:
        """Validate production-specific requirements."""
        errors = []

        if not self.config:
            return ["Configuration not loaded"]

        # Security validations
        if self.config.security.secret_key == "dev-secret-key":  # noqa: S105
            errors.append("Production requires custom SECRET_KEY")

        # CORS validations
        if "*" in self.config.security.cors_origins:
            errors.append("Production should not use wildcard CORS origins")

        # Service requirement validations
        if self.config.enable_auth and not self.config.should_use_mock_services():
            if not self.config.cognito.user_pool_id:
                errors.append("Production with auth requires COGNITO_USER_POOL_ID")
            if not self.config.cognito.client_id:
                errors.append("Production with auth requires COGNITO_CLIENT_ID")

        return errors

    async def _check_service_health(self) -> bool:
        """Check health of all required external services."""
        if not self.config:
            return False

        self.reporter.start_phase(ProgressPhase.CHECKING_SERVICES)

        try:
            # Run health checks with timeout
            health_check_timeout = min(
                self.config.health_check_timeout, self.timeout / 2
            )

            self.health_results = await asyncio.wait_for(
                self.health_checker.check_all_services(
                    self.config,
                    skip_services=self.skip_services,
                ),
                timeout=health_check_timeout,
            )

            # Report health check results
            self.reporter.report_health_checks(self.health_results)

            # Determine if we can proceed
            overall_health = self.health_checker.get_overall_health(self.health_results)

            if overall_health == ServiceStatus.UNHEALTHY:
                # Check if we can proceed with degraded functionality
                if self.config.should_use_mock_services():
                    degraded_step = self.reporter.start_step("Enabling degraded mode")
                    self.reporter.complete_step(
                        degraded_step, "Proceeding with mock services"
                    )
                    return True
                return False

            return True

        except TimeoutError:
            step = self.reporter.start_step("Service health timeout")
            self.reporter.fail_step(
                step, f"Health checks timed out after {health_check_timeout}s"
            )
            return False
        except Exception as e:  # noqa: BLE001 - Bulletproof startup
            step = self.reporter.start_step("Service health check error")
            self.reporter.fail_step(step, "Health check failed", e)
            return False

    async def _initialize_services(self) -> bool:
        """Initialize application services based on health check results."""
        if not self.config:
            return False

        self.reporter.start_phase(ProgressPhase.STARTING_SERVICES)

        try:
            # Initialize dependency container
            container_step = self.reporter.start_step(
                "Initializing dependency container"
            )

            # Import here to avoid circular imports
            from clarity.core.config_adapter import (  # noqa: PLC0415
                clarity_config_to_settings,
            )
            from clarity.core.container_aws import initialize_container  # noqa: PLC0415

            # Convert ClarityConfig to Settings for container initialization
            settings = clarity_config_to_settings(self.config)
            await initialize_container(settings)

            self.reporter.complete_step(
                container_step, "Container initialized successfully"
            )

            # Additional service initialization steps can be added here

            return True

        except Exception as e:  # noqa: BLE001 - Bulletproof startup
            step = self.reporter.start_step("Service initialization error")
            self.reporter.fail_step(step, "Service initialization failed", e)
            return False

    def create_dry_run_report(self) -> str:
        """Create comprehensive dry-run report."""
        if not self.config:
            return "❌ Dry-run failed: Configuration could not be loaded"

        return self.reporter.create_dry_run_report(
            self.config,
            self.health_results,
            self.startup_errors,
        )

    def should_startup_succeed(self) -> bool:
        """Predict if startup would succeed based on current state."""
        if self.startup_errors:
            return False

        if not self.config:
            return False

        if not self.health_results:
            return True  # Haven't checked yet

        # Check if any critical services are unhealthy
        critical_services = {"cognito", "dynamodb", "s3"}

        for service_name, result in self.health_results.items():
            if (
                service_name in critical_services
                and result.status == ServiceStatus.UNHEALTHY
                and not self.config.should_use_mock_services()
            ):
                return False

        return True


# CLI entry point functions


async def validate_config_only(*, dry_run: bool = True) -> bool:
    """Validate configuration only (CLI entry point)."""
    orchestrator = StartupOrchestrator(dry_run=dry_run)
    success, config = await orchestrator.orchestrate_startup()

    if dry_run and config:
        print(orchestrator.create_dry_run_report())  # noqa: T201

    return success


async def full_startup_check() -> bool:
    """Perform full startup validation including service checks."""
    orchestrator = StartupOrchestrator(dry_run=True)
    success, config = await orchestrator.orchestrate_startup()

    if config:
        print(orchestrator.create_dry_run_report())  # noqa: T201

    return success


def main() -> int:
    """Main CLI entry point."""
    import argparse  # noqa: PLC0415 - Main function import

    parser = argparse.ArgumentParser(description="CLARITY Startup Validation")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and services without starting",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Validate configuration only (skip service checks)",
    )
    parser.add_argument(
        "--skip-services", nargs="*", help="Services to skip during health checks"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Startup timeout in seconds"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        if args.config_only:
            success = asyncio.run(validate_config_only(dry_run=True))
        else:
            skip_services = set(args.skip_services) if args.skip_services else set()
            orchestrator = StartupOrchestrator(
                dry_run=args.dry_run,
                skip_services=skip_services,
                timeout=args.timeout,
            )
            success, config = asyncio.run(orchestrator.orchestrate_startup())

            if args.dry_run and config:
                print("\n" + orchestrator.create_dry_run_report())  # noqa: T201

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n❌ Startup validation cancelled")  # noqa: T201
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")  # noqa: T201
        logger.exception("Unexpected error in startup validation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
