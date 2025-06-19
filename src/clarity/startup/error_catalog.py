"""CLARITY Startup Error Catalog.

Comprehensive catalog of startup errors with clear messages and solutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ErrorCategory(StrEnum):
    """Error categories for organization."""

    CONFIGURATION = "configuration"
    CREDENTIALS = "credentials"
    NETWORKING = "networking"
    PERMISSIONS = "permissions"
    RESOURCES = "resources"
    DEPENDENCIES = "dependencies"
    ENVIRONMENT = "environment"


class ErrorSeverity(StrEnum):
    """Error severity levels."""

    CRITICAL = "critical"  # Prevents startup
    HIGH = "high"  # Major functionality affected
    MEDIUM = "medium"  # Some functionality affected
    LOW = "low"  # Minor issues or warnings


@dataclass
class ErrorSolution:
    """Suggested solution for an error."""

    description: str
    steps: list[str]
    documentation_links: list[str] = field(default_factory=list)


@dataclass
class StartupErrorInfo:
    """Comprehensive error information."""

    code: str
    title: str
    description: str
    category: ErrorCategory
    severity: ErrorSeverity
    solutions: list[ErrorSolution]
    common_causes: list[str]
    related_errors: list[str] = field(default_factory=list)


class StartupErrorCatalog:
    """Catalog of startup errors with solutions."""

    def __init__(self) -> None:
        self.errors: dict[str, StartupErrorInfo] = self._build_error_catalog()

    def _build_error_catalog(self) -> dict[str, StartupErrorInfo]:
        """Build comprehensive error catalog."""
        errors = {}

        # Configuration Errors
        errors["CONFIG_001"] = StartupErrorInfo(
            code="CONFIG_001",
            title="Missing Required Environment Variable",
            description="A required environment variable is not set or is empty.",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            common_causes=[
                "Environment variable not set in deployment",
                "Typo in environment variable name",
                "Variable set but value is empty",
            ],
            solutions=[
                ErrorSolution(
                    description="Set the missing environment variable",
                    steps=[
                        "Check the specific variable name in the error message",
                        "Set the variable in your environment or .env file",
                        "Verify the variable value is not empty",
                        "Restart the application",
                    ],
                ),
                ErrorSolution(
                    description="Use SKIP_EXTERNAL_SERVICES for development",
                    steps=[
                        "Set SKIP_EXTERNAL_SERVICES=true in development",
                        "This will use mock services instead of real ones",
                        "Only use this for development/testing",
                    ],
                ),
            ],
        )

        errors["CONFIG_002"] = StartupErrorInfo(
            code="CONFIG_002",
            title="Invalid Configuration Value",
            description="An environment variable has an invalid value or format.",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            common_causes=[
                "Invalid URL format",
                "Invalid numeric value",
                "Invalid enum value",
                "Value outside allowed range",
            ],
            solutions=[
                ErrorSolution(
                    description="Fix the invalid configuration value",
                    steps=[
                        "Check the error message for the specific invalid value",
                        "Refer to the configuration schema for valid formats",
                        "Update the environment variable with a valid value",
                        "Restart the application",
                    ],
                ),
            ],
        )

        errors["CONFIG_003"] = StartupErrorInfo(
            code="CONFIG_003",
            title="Production Security Requirements",
            description="Production environment requires additional security configurations.",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            common_causes=[
                "Using default SECRET_KEY in production",
                "Wildcard CORS origins in production",
                "Development credentials in production",
            ],
            solutions=[
                ErrorSolution(
                    description="Configure production security settings",
                    steps=[
                        "Set a strong SECRET_KEY (min 32 characters)",
                        "Configure specific CORS origins (no wildcards)",
                        "Use production-grade credentials",
                        "Enable all security features",
                    ],
                ),
            ],
        )

        # Credential Errors
        errors["CRED_001"] = StartupErrorInfo(
            code="CRED_001",
            title="AWS Credentials Not Found",
            description="AWS credentials are not available or invalid.",
            category=ErrorCategory.CREDENTIALS,
            severity=ErrorSeverity.HIGH,
            common_causes=[
                "AWS credentials not configured",
                "IAM role not attached to ECS task",
                "Invalid AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY",
                "Expired temporary credentials",
            ],
            solutions=[
                ErrorSolution(
                    description="Configure AWS credentials",
                    steps=[
                        "For ECS: Attach IAM role to task definition",
                        "For local: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
                        "For local: Run 'aws configure' to set up credentials",
                        "Verify credentials with 'aws sts get-caller-identity'",
                    ],
                ),
                ErrorSolution(
                    description="Use development mode with mock services",
                    steps=[
                        "Set SKIP_EXTERNAL_SERVICES=true",
                        "Set ENVIRONMENT=development",
                        "This will use mock services instead of AWS",
                    ],
                ),
            ],
        )

        errors["CRED_002"] = StartupErrorInfo(
            code="CRED_002",
            title="Insufficient AWS Permissions",
            description="AWS credentials lack required permissions for services.",
            category=ErrorCategory.PERMISSIONS,
            severity=ErrorSeverity.HIGH,
            common_causes=[
                "IAM policy missing required permissions",
                "Restrictive resource ARNs",
                "Cross-region permission issues",
                "Service-specific permissions not granted",
            ],
            solutions=[
                ErrorSolution(
                    description="Update IAM permissions",
                    steps=[
                        "Check AWS CloudTrail for denied API calls",
                        "Add required permissions to IAM policy",
                        "Verify resource ARNs are correct",
                        "Test permissions with AWS CLI",
                    ],
                    documentation_links=[
                        "https://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html"
                    ],
                ),
            ],
        )

        # Service Connectivity Errors
        errors["NET_001"] = StartupErrorInfo(
            code="NET_001",
            title="Service Timeout",
            description="Connection to external service timed out.",
            category=ErrorCategory.NETWORKING,
            severity=ErrorSeverity.MEDIUM,
            common_causes=[
                "Network connectivity issues",
                "Service temporarily unavailable",
                "Firewall blocking connections",
                "DNS resolution issues",
            ],
            solutions=[
                ErrorSolution(
                    description="Check network connectivity",
                    steps=[
                        "Verify internet connectivity",
                        "Check if service endpoints are reachable",
                        "Verify DNS resolution",
                        "Check firewall rules",
                        "Increase timeout if needed",
                    ],
                ),
                ErrorSolution(
                    description="Enable graceful degradation",
                    steps=[
                        "Set SKIP_EXTERNAL_SERVICES=true temporarily",
                        "Use mock services until connectivity is restored",
                        "Monitor service status pages",
                    ],
                ),
            ],
        )

        # Resource Errors
        errors["RES_001"] = StartupErrorInfo(
            code="RES_001",
            title="AWS Resource Not Found",
            description="Required AWS resource (table, bucket, etc.) does not exist.",
            category=ErrorCategory.RESOURCES,
            severity=ErrorSeverity.CRITICAL,
            common_causes=[
                "Resource not created in the specified region",
                "Incorrect resource name in configuration",
                "Resource deleted or renamed",
                "Cross-account access issues",
            ],
            solutions=[
                ErrorSolution(
                    description="Create or verify AWS resources",
                    steps=[
                        "Check if resource exists in AWS console",
                        "Verify resource name matches configuration",
                        "Ensure resource is in the correct region",
                        "Create resource if it doesn't exist",
                        "Check cross-account permissions if applicable",
                    ],
                ),
                ErrorSolution(
                    description="Use deployment scripts",
                    steps=[
                        "Run infrastructure deployment scripts",
                        "Use terraform or CloudFormation templates",
                        "Verify all resources are created",
                    ],
                ),
            ],
        )

        errors["RES_002"] = StartupErrorInfo(
            code="RES_002",
            title="DynamoDB Table Not Ready",
            description="DynamoDB table exists but is not in ACTIVE state.",
            category=ErrorCategory.RESOURCES,
            severity=ErrorSeverity.MEDIUM,
            common_causes=[
                "Table is still being created",
                "Table is being updated",
                "Table is being backed up",
                "Global secondary indexes being created",
            ],
            solutions=[
                ErrorSolution(
                    description="Wait for table to become active",
                    steps=[
                        "Check table status in AWS console",
                        "Wait for table to reach ACTIVE state",
                        "Monitor CloudWatch metrics",
                        "Increase startup timeout if needed",
                    ],
                ),
            ],
        )

        # Dependency Errors
        errors["DEP_001"] = StartupErrorInfo(
            code="DEP_001",
            title="Required Dependency Unavailable",
            description="A required service dependency is not available.",
            category=ErrorCategory.DEPENDENCIES,
            severity=ErrorSeverity.HIGH,
            common_causes=[
                "Service dependency not deployed",
                "Dependency in unhealthy state",
                "Network path to dependency blocked",
                "Dependency configuration mismatch",
            ],
            solutions=[
                ErrorSolution(
                    description="Check dependency status",
                    steps=[
                        "Verify dependency service is running",
                        "Check dependency health endpoints",
                        "Verify network connectivity to dependency",
                        "Check for configuration mismatches",
                    ],
                ),
                ErrorSolution(
                    description="Enable graceful degradation",
                    steps=[
                        "Configure circuit breakers",
                        "Use fallback mechanisms",
                        "Enable mock services temporarily",
                    ],
                ),
            ],
        )

        # Environment Errors
        errors["ENV_001"] = StartupErrorInfo(
            code="ENV_001",
            title="Development vs Production Mismatch",
            description="Configuration doesn't match the specified environment.",
            category=ErrorCategory.ENVIRONMENT,
            severity=ErrorSeverity.MEDIUM,
            common_causes=[
                "ENVIRONMENT variable set incorrectly",
                "Development config used in production",
                "Production config used in development",
                "Missing environment-specific settings",
            ],
            solutions=[
                ErrorSolution(
                    description="Fix environment configuration",
                    steps=[
                        "Verify ENVIRONMENT variable is set correctly",
                        "Check environment-specific configurations",
                        "Use appropriate configuration for each environment",
                        "Validate configuration against environment requirements",
                    ],
                ),
            ],
        )

        return errors

    def get_error_info(self, error_code: str) -> StartupErrorInfo | None:
        """Get error information by code."""
        return self.errors.get(error_code)

    def find_errors_by_category(
        self, category: ErrorCategory
    ) -> list[StartupErrorInfo]:
        """Find all errors in a specific category."""
        return [error for error in self.errors.values() if error.category == category]

    def find_errors_by_severity(
        self, severity: ErrorSeverity
    ) -> list[StartupErrorInfo]:
        """Find all errors with specific severity."""
        return [error for error in self.errors.values() if error.severity == severity]

    def suggest_error_code(self, error_message: str) -> str | None:
        """Suggest error code based on error message content."""
        error_message_lower = error_message.lower()

        # Simple keyword matching for error categorization
        if (
            "credentials" in error_message_lower
            or "unauthorized" in error_message_lower
        ):
            return "CRED_001"
        if "not found" in error_message_lower and (
            "table" in error_message_lower or "bucket" in error_message_lower
        ):
            return "RES_001"
        if "timeout" in error_message_lower:
            return "NET_001"
        if (
            "configuration" in error_message_lower
            or "environment" in error_message_lower
        ):
            return "CONFIG_001"
        if (
            "permission" in error_message_lower
            or "access denied" in error_message_lower
        ):
            return "CRED_002"
        if "table status" in error_message_lower:
            return "RES_002"
        return None

    def format_error_help(
        self, error_code: str, context: dict[str, str] | None = None
    ) -> str:
        """Format comprehensive error help message."""
        error_info = self.get_error_info(error_code)
        if not error_info:
            return f"Unknown error code: {error_code}"

        lines: list[str] = []
        lines.extend(
            (
                f"🚨 {error_info.title} ({error_info.code})",
                "=" * 60,
                "",
                f"📝 Description: {error_info.description}",
                f"📊 Severity: {error_info.severity.value.upper()}",
                f"🏷️  Category: {error_info.category.value.title()}",
                "",
            )
        )

        if error_info.common_causes:
            lines.append("🔍 Common Causes:")
            lines.extend(f"  • {cause}" for cause in error_info.common_causes)
            lines.append("")

        if error_info.solutions:
            lines.append("💡 Solutions:")
            for i, solution in enumerate(error_info.solutions, 1):
                lines.append(f"\n  {i}. {solution.description}")
                lines.extend(f"     • {step}" for step in solution.steps)

                if solution.documentation_links:
                    lines.append("     📖 Documentation:")
                    lines.extend(
                        f"        {link}" for link in solution.documentation_links
                    )

        if context:
            lines.extend(("", "🔧 Context:"))
            for key, value in context.items():
                lines.append(f"  • {key}: {value}")

        if error_info.related_errors:
            lines.extend(("", "🔗 Related Errors:"))
            for related_code in error_info.related_errors:
                related_error = self.get_error_info(related_code)
                if related_error:
                    lines.append(f"  • {related_code}: {related_error.title}")

        return "\n".join(lines)


# Global error catalog instance
error_catalog = StartupErrorCatalog()
