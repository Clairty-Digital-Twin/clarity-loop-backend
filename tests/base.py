"""Base test classes following Clean Architecture principles.

This module provides base test classes that implement dependency injection
and other testing best practices, making individual test files cleaner
and more focused on testing business logic rather than setup.
"""

from __future__ import annotations

from typing import Any
import unittest

from tests.fakes.storage import FakeCloudStorage, FakeStorage


class BaseTestCase(unittest.TestCase):
    """Base test case with clean dependency injection setup.

    This base class provides:
    1. Proper dependency injection of test doubles
    2. Clean setup/teardown patterns
    3. Common test utilities
    4. Consistent test structure across the codebase

    Follows the "Arrange-Act-Assert" pattern and provides
    clear separation between test setup and test logic.
    """

    def setUp(self) -> None:
        """Set up test dependencies using dependency injection.

        This method creates test doubles (fakes) rather than mocks,
        following the "Use Fakes Instead of Mocks" best practice.
        """
        # Create clean test dependencies
        self.storage = FakeStorage()
        self.cloud_storage = FakeCloudStorage()

        # Add any other common test setup here
        self.test_data: dict[str, Any] = {}

    def tearDown(self) -> None:
        """Clean up after each test."""
        # Clear any test data
        self.test_data.clear()

    def create_test_document(self, collection: str, data: dict[str, Any]) -> str:
        """Helper method to create test documents.

        Args:
            collection: Collection name
            data: Document data

        Returns:
            Document ID
        """
        return self.storage.create_document(collection, data)

    def assert_document_exists(self, collection: str, doc_id: str) -> None:
        """Assert that a document exists in storage.

        Args:
            collection: Collection name
            doc_id: Document ID
        """
        doc = self.storage.get_document(collection, doc_id)
        assert doc is not None, f"Document {doc_id} should exist in {collection}"

    def assert_document_not_exists(self, collection: str, doc_id: str) -> None:
        """Assert that a document does not exist in storage.

        Args:
            collection: Collection name
            doc_id: Document ID
        """
        doc = self.storage.get_document(collection, doc_id)
        assert doc is None, f"Document {doc_id} should not exist in {collection}"

    def assert_document_contains(
        self, collection: str, doc_id: str, expected_data: dict[str, Any]
    ) -> None:
        """Assert that a document contains expected data.

        Args:
            collection: Collection name
            doc_id: Document ID
            expected_data: Expected document data
        """
        doc = self.storage.get_document(collection, doc_id)
        assert doc is not None, f"Document {doc_id} should exist"

        for key, value in expected_data.items():
            assert key in doc, f"Document should contain key '{key}'"
            assert doc[key] == value, f"Document['{key}'] should equal {value}"


class BaseServiceTestCase(BaseTestCase):
    """Base test case for service layer tests.

    Extends BaseTestCase with service-specific testing utilities
    and dependency injection patterns for service objects.
    """

    def setUp(self) -> None:
        """Set up service test dependencies."""
        super().setUp()

        # Services will be injected in subclasses
        self.service: Any | None = None

    def create_service_with_dependencies(
        self,
        service_class: type,
        **kwargs: Any,
    ) -> Any:
        """Factory method to create services with injected dependencies.

        This method demonstrates the proper way to inject dependencies
        into service objects during testing, following the dependency
        inversion principle.

        Args:
            service_class: The service class to instantiate
            **kwargs: Additional arguments for service constructor

        Returns:
            Service instance with injected dependencies
        """
        # Default dependency injection
        dependencies = {
            "storage": self.storage,
            "cloud_storage": self.cloud_storage,
            **kwargs,  # Allow override of dependencies
        }

        return service_class(**dependencies)

    @staticmethod
    def assert_service_call_successful(result: object) -> None:
        """Assert that a service call was successful.

        Args:
            result: Service method result
        """
        assert result is not None, "Service call should return a result"

    def assert_service_call_failed(self, result: Any) -> None:
        """Assert that a service call failed appropriately.

        Args:
            result: Service method result
        """
        # This can be customized based on your error handling patterns


class BaseIntegrationTestCase(BaseTestCase):
    """Base test case for integration tests.

    Integration tests verify that multiple components work together
    correctly. This base class provides utilities for setting up
    more complex test scenarios with multiple services.
    """

    def setUp(self) -> None:
        """Set up integration test dependencies."""
        super().setUp()

        # Integration tests might need multiple services
        self.services: dict[str, Any] = {}

    def register_service(self, name: str, service: Any) -> None:
        """Register a service for use in integration tests.

        Args:
            name: Service name
            service: Service instance
        """
        self.services[name] = service

    def get_service(self, name: str) -> Any:
        """Get a registered service.

        Args:
            name: Service name

        Returns:
            Service instance
        """
        return self.services.get(name)


# Example of how to use these base classes:
"""
class TestHealthDataService(BaseServiceTestCase):
    def setUp(self):
        super().setUp()
        # Inject dependencies cleanly
        self.service = self.create_service_with_dependencies(HealthDataService)

    def test_create_health_record(self):
        # Arrange
        user_id = "test_user"
        health_data = {"heart_rate": 70, "steps": 10000}

        # Act
        result = self.service.create_health_record(user_id, health_data)

        # Assert
        self.assert_service_call_successful(result)
        self.assert_document_exists("health_records", result)
"""
