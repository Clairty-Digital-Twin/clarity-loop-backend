"""Test error catalog module."""

import pytest

from clarity.startup.error_catalog import (
    ErrorCategory,
    ErrorSeverity,
    ErrorSolution,
    StartupErrorCatalog,
    StartupErrorInfo,
)


class TestErrorCategory:
    """Test ErrorCategory enum."""

    def test_error_category_values(self):
        """Test error category enum values."""
        assert ErrorCategory.CONFIGURATION == "configuration"
        assert ErrorCategory.CREDENTIALS == "credentials"
        assert ErrorCategory.NETWORKING == "networking"
        assert ErrorCategory.PERMISSIONS == "permissions"
        assert ErrorCategory.RESOURCES == "resources"
        assert ErrorCategory.DEPENDENCIES == "dependencies"
        assert ErrorCategory.ENVIRONMENT == "environment"


class TestErrorSeverity:
    """Test ErrorSeverity enum."""

    def test_error_severity_values(self):
        """Test error severity enum values."""
        assert ErrorSeverity.CRITICAL == "critical"
        assert ErrorSeverity.HIGH == "high"
        assert ErrorSeverity.MEDIUM == "medium"
        assert ErrorSeverity.LOW == "low"


class TestErrorSolution:
    """Test ErrorSolution class."""

    def test_error_solution_creation(self):
        """Test creating error solution."""
        solution = ErrorSolution(
            description="Check AWS credentials",
            steps=[
                "Verify AWS_ACCESS_KEY_ID is set",
                "Verify AWS_SECRET_ACCESS_KEY is set",
                "Check AWS region configuration",
            ],
        )

        assert solution.description == "Check AWS credentials"
        assert len(solution.steps) == 3
        assert solution.steps[0] == "Verify AWS_ACCESS_KEY_ID is set"


class TestStartupErrorInfo:
    """Test StartupErrorInfo class."""

    def test_startup_error_info_creation(self):
        """Test creating startup error info."""
        solution = ErrorSolution(
            description="Fix the issue", steps=["Step 1", "Step 2"]
        )

        error_info = StartupErrorInfo(
            code="ERR001",
            title="Test Error",
            description="Test error description",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            solutions=[solution],
            common_causes=["Test cause"],
            related_errors=["ERR002"],
        )

        assert error_info.code == "ERR001"
        assert error_info.title == "Test Error"
        assert error_info.description == "Test error description"
        assert error_info.category == ErrorCategory.CONFIGURATION
        assert error_info.severity == ErrorSeverity.HIGH
        assert error_info.solutions == [solution]
        assert error_info.common_causes == ["Test cause"]
        assert error_info.related_errors == ["ERR002"]


class TestStartupErrorCatalog:
    """Test StartupErrorCatalog functionality."""

    def test_error_catalog_initialization(self):
        """Test error catalog is initialized with errors."""
        catalog = StartupErrorCatalog()

        # Check that some known errors are present
        assert "CONFIG_001" in catalog.errors
        assert "CRED_001" in catalog.errors
        assert "CONFIG_002" in catalog.errors

    def test_get_error_info(self):
        """Test getting error info from catalog."""
        catalog = StartupErrorCatalog()

        error_info = catalog.get_error_info("CONFIG_001")
        assert error_info is not None
        assert error_info.code == "CONFIG_001"
        assert error_info.category == ErrorCategory.CONFIGURATION

    def test_get_nonexistent_error(self):
        """Test getting nonexistent error returns None."""
        catalog = StartupErrorCatalog()

        error_info = catalog.get_error_info("NONEXISTENT")
        assert error_info is None
