"""CLARITY Startup Progress Reporter.

Provides clear, real-time feedback during application startup with progress
indicators and detailed status messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import logging
import sys
import time
from typing import Any, TextIO

from clarity.startup.health_checks import HealthCheckResult, ServiceStatus

logger = logging.getLogger(__name__)


class ProgressPhase(StrEnum):
    """Startup progress phases."""

    INITIALIZING = "initializing"
    VALIDATING_CONFIG = "validating_config"
    CHECKING_SERVICES = "checking_services"
    STARTING_SERVICES = "starting_services"
    CONFIGURING_APP = "configuring_app"
    READY = "ready"
    FAILED = "failed"


@dataclass
class ProgressStep:
    """Individual progress step."""

    name: str
    phase: ProgressPhase
    status: str = "pending"  # pending, running, completed, failed
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None
    error: Exception | None = None

    @property
    def duration_ms(self) -> float:
        """Get step duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def start(self) -> None:
        """Mark step as started."""
        self.status = "running"
        self.start_time = time.time()

    def complete(
        self, message: str = "", details: dict[str, Any] | None = None
    ) -> None:
        """Mark step as completed."""
        self.status = "completed"
        self.end_time = time.time()
        if message:
            self.message = message
        if details:
            self.details.update(details)

    def fail(
        self,
        message: str,
        error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Mark step as failed."""
        self.status = "failed"
        self.end_time = time.time()
        self.message = message
        self.error = error
        if details:
            self.details.update(details)


class StartupProgressReporter:
    """Reports startup progress with clear status messages."""

    def __init__(
        self, output: TextIO | None = None, *, enable_colors: bool = True
    ) -> None:
        """Initialize progress reporter.

        Args:
            output: Output stream (defaults to stdout)
            enable_colors: Whether to use colored output
        """
        self.output = output or sys.stdout
        self.enable_colors = (
            enable_colors and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        )
        self.steps: list[ProgressStep] = []
        self.current_phase = ProgressPhase.INITIALIZING
        self.start_time = time.time()
        self.end_time: float | None = None

        # Color codes
        self.colors = (
            {
                "reset": "\033[0m",
                "bold": "\033[1m",
                "green": "\033[32m",
                "yellow": "\033[33m",
                "red": "\033[31m",
                "blue": "\033[34m",
                "cyan": "\033[36m",
                "gray": "\033[90m",
            }
            if self.enable_colors
            else dict.fromkeys(
                ["reset", "bold", "green", "yellow", "red", "blue", "cyan", "gray"], ""
            )
        )

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        return f"{self.colors[color]}{text}{self.colors['reset']}"

    def _print(self, message: str) -> None:
        """Print message to output."""
        print(message, file=self.output, flush=True)

    def _get_phase_emoji(self, phase: ProgressPhase) -> str:
        """Get emoji for progress phase."""
        phase_emojis = {
            ProgressPhase.INITIALIZING: "🚀",
            ProgressPhase.VALIDATING_CONFIG: "⚙️",
            ProgressPhase.CHECKING_SERVICES: "🔍",
            ProgressPhase.STARTING_SERVICES: "🔧",
            ProgressPhase.CONFIGURING_APP: "📋",
            ProgressPhase.READY: "✅",
            ProgressPhase.FAILED: "❌",
        }
        return phase_emojis.get(phase, "📍")

    def _get_status_symbol(self, status: str) -> str:
        """Get symbol for step status."""
        symbols = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "skipped": "⏭️",
        }
        return symbols.get(status, "❓")

    def start_startup(self, app_name: str = "CLARITY Digital Twin") -> None:
        """Start startup progress reporting."""
        self.start_time = time.time()
        header = f"{self._colorize('🚀 Starting', 'bold')} {self._colorize(app_name, 'cyan')}"
        self._print(f"\n{header}")
        self._print(f"{self._colorize('=' * 60, 'gray')}")

    def start_phase(self, phase: ProgressPhase, message: str = "") -> None:
        """Start a new startup phase."""
        self.current_phase = phase
        emoji = self._get_phase_emoji(phase)
        phase_name = phase.value.replace("_", " ").title()

        if message:
            display_message = f"{emoji} {self._colorize(phase_name, 'bold')}: {message}"
        else:
            display_message = f"{emoji} {self._colorize(phase_name, 'bold')}"

        self._print(f"\n{display_message}")
        logger.info("Startup phase: %s", phase_name)

    def add_step(self, name: str, phase: ProgressPhase | None = None) -> ProgressStep:
        """Add a new progress step."""
        step = ProgressStep(
            name=name,
            phase=phase or self.current_phase,
        )
        self.steps.append(step)
        return step

    def start_step(self, name: str, message: str = "") -> ProgressStep:
        """Start a new progress step."""
        step = self.add_step(name)
        step.start()

        symbol = self._get_status_symbol("running")
        display_message = f"  {symbol} {name}"
        if message:
            display_message += f": {self._colorize(message, 'gray')}"

        self._print(display_message)
        return step

    def complete_step(
        self,
        step: ProgressStep,
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Mark step as completed."""
        step.complete(message, details)

        symbol = self._get_status_symbol("completed")
        display_message = f"  {symbol} {self._colorize(step.name, 'green')}"

        if message:
            display_message += f": {message}"

        if step.duration_ms > 0:
            duration_str = f"({step.duration_ms:.0f}ms)"
            display_message += f" {self._colorize(duration_str, 'gray')}"

        self._print(display_message)
        logger.info("Completed: %s in %.0fms", step.name, step.duration_ms)

    def fail_step(
        self,
        step: ProgressStep,
        message: str,
        error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Mark step as failed."""
        step.fail(message, error, details)

        symbol = self._get_status_symbol("failed")
        display_message = f"  {symbol} {self._colorize(step.name, 'red')}: {self._colorize(message, 'red')}"

        self._print(display_message)

        if error:
            self._print(f"    {self._colorize('Error:', 'red')} {error!s}")
            logger.error("Step failed: %s - %s", step.name, message, exc_info=error)
        else:
            logger.error("Step failed: %s - %s", step.name, message)

    def skip_step(self, step: ProgressStep, reason: str) -> None:
        """Mark step as skipped."""
        step.status = "skipped"
        step.message = reason
        step.end_time = time.time()

        symbol = self._get_status_symbol("skipped")
        display_message = f"  {symbol} {self._colorize(step.name, 'yellow')}: {self._colorize(reason, 'gray')}"

        self._print(display_message)
        logger.info("Skipped: %s - %s", step.name, reason)

    def report_health_checks(self, results: dict[str, HealthCheckResult]) -> None:
        """Report health check results."""
        self.start_phase(
            ProgressPhase.CHECKING_SERVICES, "Running service health checks"
        )

        for service_name, result in results.items():
            step = self.start_step(f"Health check: {service_name}")

            if result.status == ServiceStatus.HEALTHY:
                self.complete_step(
                    step,
                    result.message,
                    {
                        "response_time": f"{result.response_time_ms:.0f}ms",
                        **result.details,
                    },
                )
            elif result.status == ServiceStatus.DEGRADED:
                # Treat degraded as completed with warning
                step.complete(result.message, result.details)
                symbol = "⚠️"
                display_message = f"  {symbol} {self._colorize(step.name, 'yellow')}: {self._colorize(result.message, 'yellow')}"
                self._print(display_message)
            elif result.status == ServiceStatus.SKIPPED:
                self.skip_step(step, result.message)
            else:
                self.fail_step(step, result.message, result.error, result.details)

    def report_config_validation(
        self, config: Any, validation_errors: list[str]
    ) -> None:
        """Report configuration validation results."""
        self.start_phase(ProgressPhase.VALIDATING_CONFIG, "Validating configuration")

        if validation_errors:
            step = self.start_step("Configuration validation")
            error_msg = f"Found {len(validation_errors)} configuration error(s)"
            self.fail_step(step, error_msg, details={"errors": validation_errors})

            # Print detailed error information
            self._print(f"\n{self._colorize('Configuration Errors:', 'red')}")
            for i, error in enumerate(validation_errors, 1):
                self._print(f"  {i}. {self._colorize(error, 'red')}")
        else:
            step = self.start_step("Configuration validation")
            summary = config.get_startup_summary()
            self.complete_step(step, "All configuration valid", summary)

    def report_startup_complete(
        self, *, success: bool = True, message: str = ""
    ) -> None:
        """Report startup completion."""
        self.end_time = time.time()
        total_duration = (self.end_time - self.start_time) * 1000

        if success:
            self.current_phase = ProgressPhase.READY
            emoji = self._get_phase_emoji(ProgressPhase.READY)
            status_msg = f"{emoji} {self._colorize('Startup Complete', 'green')} ({total_duration:.0f}ms)"
            if message:
                status_msg += f": {message}"
            self._print(f"\n{status_msg}")
            logger.info("Startup completed successfully in %.0fms", total_duration)
        else:
            self.current_phase = ProgressPhase.FAILED
            emoji = self._get_phase_emoji(ProgressPhase.FAILED)
            status_msg = f"{emoji} {self._colorize('Startup Failed', 'red')} ({total_duration:.0f}ms)"
            if message:
                status_msg += f": {message}"
            self._print(f"\n{status_msg}")
            logger.error("Startup failed after %.0fms: %s", total_duration, message)

        self._print(f"{self._colorize('=' * 60, 'gray')}\n")

    def get_startup_summary(self) -> dict[str, Any]:
        """Get startup summary."""
        completed_steps = [s for s in self.steps if s.status == "completed"]
        failed_steps = [s for s in self.steps if s.status == "failed"]
        skipped_steps = [s for s in self.steps if s.status == "skipped"]

        total_duration = 0.0
        if self.end_time:
            total_duration = (self.end_time - self.start_time) * 1000

        return {
            "total_duration_ms": total_duration,
            "total_steps": len(self.steps),
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "skipped_steps": len(skipped_steps),
            "final_phase": self.current_phase.value,
            "success": len(failed_steps) == 0
            and self.current_phase == ProgressPhase.READY,
        }

    def print_startup_summary(self) -> None:
        """Print detailed startup summary."""
        summary = self.get_startup_summary()

        self._print(f"{self._colorize('Startup Summary:', 'bold')}")
        self._print(f"  Total time: {summary['total_duration_ms']:.0f}ms")
        self._print(f"  Total steps: {summary['total_steps']}")
        self._print(f"  ✅ Completed: {summary['completed_steps']}")
        self._print(f"  ❌ Failed: {summary['failed_steps']}")
        self._print(f"  ⏭️  Skipped: {summary['skipped_steps']}")

        if summary["failed_steps"] > 0:
            self._print(f"\n{self._colorize('Failed Steps:', 'red')}")
            for step in self.steps:
                if step.status == "failed":
                    self._print(f"  • {step.name}: {step.message}")

    def create_dry_run_report(
        self,
        config: Any,
        health_results: dict[str, HealthCheckResult],
        validation_errors: list[str],
    ) -> str:
        """Create dry-run report."""
        lines = [
            "🔍 CLARITY Startup Dry-Run Report",
            "=" * 50,
            "",
            "📋 Configuration Summary:",
        ]
        summary = config.get_startup_summary()
        for key, value in summary.items():
            lines.append(f"  • {key}: {value}")
        lines.append("")

        # Validation results
        if validation_errors:
            lines.append(
                f"❌ Configuration Validation: {len(validation_errors)} error(s)"
            )
            lines.extend(f"  • {error}" for error in validation_errors)
        else:
            lines.append("✅ Configuration Validation: All valid")
        lines.extend(("", "🔍 Service Health Checks:"))
        for service_name, result in health_results.items():
            status_symbol = {
                ServiceStatus.HEALTHY: "✅",
                ServiceStatus.DEGRADED: "⚠️",
                ServiceStatus.UNHEALTHY: "❌",
                ServiceStatus.SKIPPED: "⏭️",
                ServiceStatus.UNKNOWN: "❓",
            }.get(result.status, "❓")

            lines.append(f"  {status_symbol} {service_name}: {result.message}")
            if result.response_time_ms > 0:
                lines.append(f"    Response time: {result.response_time_ms:.0f}ms")

        lines.append("")

        # Overall assessment
        overall_healthy = all(
            result.status
            in {ServiceStatus.HEALTHY, ServiceStatus.SKIPPED, ServiceStatus.DEGRADED}
            for result in health_results.values()
        )

        if validation_errors:
            lines.append("🚨 STARTUP WOULD FAIL: Configuration errors must be fixed")
        elif not overall_healthy:
            lines.append("⚠️  STARTUP MAY FAIL: Some services are unhealthy")
        else:
            lines.append("✅ STARTUP SHOULD SUCCEED: All checks passed")

        return "\n".join(lines)
