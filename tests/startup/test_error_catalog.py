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
        assert ErrorCategory.AUTH == "auth"
        assert ErrorCategory.DATABASE == "database"
        assert ErrorCategory.MESSAGING == "messaging"
        assert ErrorCategory.CONFIGURATION == "configuration"
        assert ErrorCategory.ML_SERVICE == "ml_service"
        assert ErrorCategory.EXTERNAL_API == "external_api"
        assert ErrorCategory.STORAGE == "storage"
        assert ErrorCategory.UNKNOWN == "unknown"


class TestErrorSeverity:
    """Test ErrorSeverity enum."""

    def test_error_severity_values(self):
        """Test error severity enum values."""
        assert ErrorSeverity.INFO == "info"
        assert ErrorSeverity.WARNING == "warning"
        assert ErrorSeverity.ERROR == "error"
        assert ErrorSeverity.CRITICAL == "critical"


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
            message="Test error message",
            category=ErrorCategory.AUTH,
            severity=ErrorSeverity.ERROR,
            solution=solution,
            retry_after=60,
            documentation_link="https://docs.example.com/errors/ERR001",
        )

        assert error_info.code == "ERR001"
        assert error_info.message == "Test error message"
        assert error_info.category == ErrorCategory.AUTH
        assert error_info.severity == ErrorSeverity.ERROR
        assert error_info.solution == solution
        assert error_info.retry_after == 60
        assert error_info.documentation_link == "https://docs.example.com/errors/ERR001"


class TestStartupErrorCatalog:
    """Test StartupErrorCatalog functionality."""

    def test_error_catalog_initialization(self):
        """Test error catalog is initialized with errors."""
        catalog = StartupErrorCatalog()

        # Check that some known errors are present
        assert "COGNITO_INIT_001" in catalog.ERROR_CATALOG
        assert "DB_INIT_001" in catalog.ERROR_CATALOG
        assert "SQS_INIT_001" in catalog.ERROR_CATALOG

    def test_get_error_info(self):
        """Test getting error info from catalog."""
        catalog = StartupErrorCatalog()

        error_info = catalog.get_error_info("COGNITO_INIT_001")
        assert error_info is not None
        assert error_info.code == "COGNITO_INIT_001"
        assert error_info.category == ErrorCategory.AUTH

    def test_get_nonexistent_error(self):
        """Test getting nonexistent error returns None."""
        catalog = StartupErrorCatalog()

        error_info = catalog.get_error_info("NONEXISTENT")
        assert error_info is None
