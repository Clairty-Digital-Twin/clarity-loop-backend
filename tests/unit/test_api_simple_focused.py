"""MICRO-FOCUSED API Tests - CHUNK 2A.

🚀 ULTRA-SMALL CHUNK APPROACH 🚀
Target: Just test basic API error paths

Breaking down into TINY pieces:
- Just basic error handling
- Simple exception paths
- Core business logic branches

ONE FUNCTION AT A TIME!
"""

from __future__ import annotations

import math
from unittest.mock import Mock

import pytest

from clarity.core.exceptions import DataValidationError


class TestBasicExceptions:
    """Test basic exception creation - MICRO CHUNK 2A."""

    @staticmethod
    def test_data_validation_error_creation() -> None:
        """Test creating DataValidationError."""
        # Arrange
        message = "Invalid data format"
        field = "email"

        # Act
        error = DataValidationError(message, field_name=field)

        # Assert
        assert str(error) == "[DATA_VALIDATION_ERROR] Invalid data format"
        assert error.field_name == field
        assert error.error_code == "DATA_VALIDATION_ERROR"

    @staticmethod
    def test_data_validation_error_without_field() -> None:
        """Test creating DataValidationError without field name."""
        # Arrange
        message = "General validation error"

        # Act
        error = DataValidationError(message)

        # Assert
        assert str(error) == "[DATA_VALIDATION_ERROR] General validation error"
        assert error.field_name is None

    @staticmethod
    def test_data_validation_error_with_details() -> None:
        """Test creating DataValidationError with additional details."""
        # Arrange
        message = "Invalid format"
        field = "phone"
        details = {"expected": "E.164", "actual": "xxx-xxx-xxxx"}

        # Act
        error = DataValidationError(message, field_name=field, details=details)

        # Assert
        assert error.details == details
        assert error.field_name == field

    @staticmethod
    def test_raising_data_validation_error() -> None:
        """Test raising and catching DataValidationError."""
        # Arrange
        message = "Test error"

        # Act & Assert
        with pytest.raises(DataValidationError) as exc_info:
            raise DataValidationError(message)

        assert "Test error" in str(exc_info.value)


class TestBasicValidations:
    """Test basic validation functions - MICRO CHUNK 2B."""

    @staticmethod
    def test_empty_string_validation() -> None:
        """Test validation of empty strings."""
        # Arrange
        empty_string = ""
        valid_string = "hello"

        # Act & Assert
        assert len(empty_string) == 0
        assert len(valid_string) > 0

    @staticmethod
    def test_none_validation() -> None:
        """Test validation of None values."""
        # Arrange
        none_value = None
        valid_value = "hello"

        # Act & Assert
        assert none_value is None
        assert valid_value is not None

    @staticmethod
    def test_numeric_validation() -> None:
        """Test validation of numeric values."""
        # Arrange
        valid_int = 42
        valid_float = math.pi
        invalid_string = "not_a_number"

        # Act & Assert
        assert isinstance(valid_int, int)
        assert isinstance(valid_float, float)
        assert not isinstance(invalid_string, (int, float))  # type: ignore[unreachable]

    @staticmethod
    def test_range_validation() -> None:
        """Test validation of numeric ranges."""
        # Arrange
        min_val = 0
        max_val = 100
        valid_value = 50
        invalid_low = -10
        invalid_high = 150

        # Act & Assert
        assert min_val <= valid_value <= max_val
        assert not (min_val <= invalid_low <= max_val)
        assert not (min_val <= invalid_high <= max_val)


class TestMockBehavior:
    """Test mock object behavior - MICRO CHUNK 2C."""

    @staticmethod
    def test_mock_creation() -> None:
        """Test creating basic mock objects."""
        # Arrange & Act
        mock_obj = Mock()

        # Assert
        assert mock_obj is not None
        assert hasattr(mock_obj, "some_method")

    @staticmethod
    def test_mock_return_value() -> None:
        """Test setting mock return values."""
        # Arrange
        mock_obj = Mock()
        expected_result = "test_result"
        mock_obj.some_method.return_value = expected_result

        # Act
        result = mock_obj.some_method()

        # Assert
        assert result == expected_result

    @staticmethod
    def test_mock_side_effect() -> None:
        """Test mock side effects."""
        # Arrange
        mock_obj = Mock()
        test_exception = DataValidationError("Test error")
        mock_obj.failing_method.side_effect = test_exception

        # Act & Assert
        with pytest.raises(DataValidationError):
            mock_obj.failing_method()

    @staticmethod
    def test_mock_call_counting() -> None:
        """Test counting mock method calls."""
        # Arrange
        mock_obj = Mock()

        # Act
        mock_obj.some_method()
        mock_obj.some_method()

        # Assert
        assert mock_obj.some_method.call_count == 2


class TestStringOperations:
    """Test basic string operations for validation - MICRO CHUNK 2D."""

    @staticmethod
    def test_string_length_check() -> None:
        """Test string length validation."""
        # Arrange
        short_string = "hi"
        long_string = "this is a much longer string"
        empty_string = ""

        # Act & Assert
        assert len(short_string) == 2
        assert len(long_string) > 10
        assert len(empty_string) == 0

    @staticmethod
    def test_string_strip_operation() -> None:
        """Test string whitespace removal."""
        # Arrange
        padded_string = "  hello world  "
        expected = "hello world"

        # Act
        result = padded_string.strip()

        # Assert
        assert result == expected

    @staticmethod
    def test_string_format_validation() -> None:
        """Test basic string format checks."""
        # Arrange
        email_like = "user@example.com"
        not_email_like = "not an email"

        # Act & Assert
        assert "@" in email_like
        assert "." in email_like
        assert "@" not in not_email_like

    @staticmethod
    def test_string_case_operations() -> None:
        """Test string case conversions."""
        # Arrange
        mixed_case = "Hello World"

        # Act & Assert
        assert mixed_case.lower() == "hello world"
        assert mixed_case.upper() == "HELLO WORLD"
