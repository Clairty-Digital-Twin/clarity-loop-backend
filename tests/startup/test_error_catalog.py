"""Tests for startup error catalog - real functionality tests."""

from clarity.startup.error_catalog import (
    ErrorCategory,
    ErrorSeverity,
    ErrorSolution,
    StartupErrorCatalog,
    StartupErrorInfo,
)


class TestStartupErrorCatalog:
    """Test startup error catalog real functionality."""

    def test_error_catalog_initialization(self):
        """Test that error catalog initializes with actual errors."""
        catalog = StartupErrorCatalog()

        # Should have real errors defined
        assert len(catalog.errors) > 0

        # Check a specific error exists
        assert "CONFIG_001" in catalog.errors

        # Verify error structure
        error = catalog.errors["CONFIG_001"]
        assert error.code == "CONFIG_001"
        assert error.title == "Missing Required Environment Variable"
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.severity == ErrorSeverity.CRITICAL
        assert len(error.solutions) > 0
        assert len(error.common_causes) > 0

    def test_get_error_info_by_code(self):
        """Test retrieving specific error by code."""
        catalog = StartupErrorCatalog()
        error = catalog.get_error_info("CONFIG_001")

        assert error is not None
        assert isinstance(error, StartupErrorInfo)
        assert error.code == "CONFIG_001"

    def test_get_nonexistent_error_info(self):
        """Test retrieving non-existent error returns None."""
        catalog = StartupErrorCatalog()
        error = catalog.get_error_info("FAKE_999")

        assert error is None

    def test_error_solution_structure(self):
        """Test error solutions have proper structure."""
        catalog = StartupErrorCatalog()
        error = catalog.get_error_info("CONFIG_001")

        assert error is not None
        assert len(error.solutions) >= 1

        solution = error.solutions[0]
        assert isinstance(solution, ErrorSolution)
        assert len(solution.description) > 0
        assert len(solution.steps) > 0
        assert isinstance(solution.documentation_links, list)

    def test_format_error_help(self):
        """Test formatting error for user display."""
        catalog = StartupErrorCatalog()
        formatted = catalog.format_error_help("CONFIG_001")

        assert formatted is not None
        assert "CONFIG_001" in formatted
        assert "Missing Required Environment Variable" in formatted
        assert "Common Causes:" in formatted
        assert "Solutions:" in formatted

    def test_suggest_error_code(self):
        """Test suggesting relevant error based on exception message."""
        catalog = StartupErrorCatalog()

        # Test environment variable error
        error_code = catalog.suggest_error_code(
            "COGNITO_USER_POOL_ID environment variable not set"
        )
        assert error_code == "CONFIG_001"

        # Test AWS credentials error
        error_code = catalog.suggest_error_code("Unable to locate credentials")
        assert error_code in {"AWS_001", "CRED_001"}  # Could match multiple

        # Test unknown error
        error_code = catalog.suggest_error_code("Random unrelated error message")
        assert error_code is None

    def test_find_errors_by_category(self):
        """Test filtering errors by category."""
        catalog = StartupErrorCatalog()
        config_errors = catalog.find_errors_by_category(ErrorCategory.CONFIGURATION)

        assert len(config_errors) > 0
        for error in config_errors:
            assert error.category == ErrorCategory.CONFIGURATION

    def test_find_errors_by_severity(self):
        """Test filtering errors by severity."""
        catalog = StartupErrorCatalog()
        critical_errors = catalog.find_errors_by_severity(ErrorSeverity.CRITICAL)

        assert len(critical_errors) > 0
        for error in critical_errors:
            assert error.severity == ErrorSeverity.CRITICAL
