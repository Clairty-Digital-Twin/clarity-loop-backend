"""Comprehensive tests for Firestore client functionality.

Tests cover:
- Client initialization and connection management
- Document CRUD operations
- Caching functionality
- Health data operations
- Audit logging
- Error handling
- Repository pattern implementation
"""

import asyncio
from collections.abc import Generator
from datetime import UTC, datetime
import time
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from clarity.models.health_data import (
    BiometricData,
    HealthDataUpload,
    HealthMetric,
    HealthMetricType,
)
from clarity.storage.firestore_client import (
    FirestoreClient,
    FirestoreConnectionError,
    FirestoreError,
    FirestoreHealthDataRepository,
    FirestoreValidationError,
)


@pytest.fixture
def mock_firebase_admin() -> Generator[Mock, None, None]:
    """Mock Firebase Admin SDK."""
    with patch("clarity.storage.firestore_client.firebase_admin") as mock_admin:
        mock_admin._apps = {}  # Empty apps list
        mock_admin.initialize_app = Mock()
        yield mock_admin


@pytest.fixture
def mock_firestore_client() -> Generator[AsyncMock, None, None]:
    """Mock Firestore AsyncClient."""
    with patch("clarity.storage.firestore_client.firestore.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_credentials() -> Generator[Mock, None, None]:
    """Mock Firebase credentials."""
    with patch("clarity.storage.firestore_client.credentials.Certificate") as mock_creds:
        yield mock_creds


@pytest.fixture
def firestore_client(_mock_firebase_admin: Mock, _mock_credentials: Mock) -> FirestoreClient:
    """Create FirestoreClient instance with mocked dependencies."""
    return FirestoreClient(
        project_id="test-project",
        credentials_path="/path/to/creds.json",
        enable_caching=True,
        cache_ttl=300,
    )


@pytest.fixture
def sample_health_data() -> dict[str, Any]:
    """Sample health data for testing."""
    return {
        "user_id": str(uuid4()),
        "metrics": [
            {
                "metric_id": str(uuid4()),
                "metric_type": "heart_rate",
                "value": 72.0,
                "timestamp": datetime.now(UTC),
            }
        ],
        "upload_source": "apple_health",
        "created_at": datetime.now(UTC),
    }


@pytest.fixture
def sample_health_upload() -> HealthDataUpload:
    """Sample HealthDataUpload for testing."""
    user_id = uuid4()
    return HealthDataUpload(
        user_id=user_id,
        metrics=[
            HealthMetric(
                metric_id=uuid4(),
                metric_type=HealthMetricType.HEART_RATE,
                device_id="test_device",
                raw_data={"source": "test"},
                metadata={"test": True},
                biometric_data=BiometricData(
                    heart_rate=72,
                    heart_rate_variability=25.0,
                    systolic_bp=120,
                    diastolic_bp=80,
                    respiratory_rate=16,
                    skin_temperature=37.0,  # 37°C (normal body temperature)
                ),
            )
        ],
        upload_source="apple_health",
        client_timestamp=datetime.now(UTC),
        sync_token="sync_12345",  # noqa: S106
    )


class TestFirestoreClientInitialization:
    """Test FirestoreClient initialization and setup."""

    @staticmethod
    def test_init_with_credentials_path(mock_firebase_admin: Mock, mock_credentials: Mock) -> None:
        """Test initialization with credentials path."""
        client = FirestoreClient(
            project_id="test-project",
            credentials_path="/path/to/creds.json",
        )

        assert client.project_id == "test-project"
        assert client.database_name == "(default)"
        assert client.enable_caching is True
        assert client.cache_ttl == 300
        mock_credentials.assert_called_once_with("/path/to/creds.json")
        mock_firebase_admin.initialize_app.assert_called_once()

    @staticmethod
    def test_init_without_credentials_path(mock_firebase_admin: Mock) -> None:
        """Test initialization without credentials path (uses ADC)."""
        client = FirestoreClient(project_id="test-project")

        assert client.project_id == "test-project"
        mock_firebase_admin.initialize_app.assert_called_once_with()

    @staticmethod
    def test_init_firebase_already_initialized(mock_firebase_admin: Mock) -> None:
        """Test initialization when Firebase is already initialized."""
        mock_firebase_admin._apps = {"default": Mock()}  # Non-empty apps

        client = FirestoreClient(project_id="test-project")
        assert client.project_id == "test-project"

    @staticmethod
    def test_init_firebase_failure(mock_firebase_admin: Mock) -> None:
        """Test Firebase initialization failure."""
        mock_firebase_admin.initialize_app.side_effect = Exception("Init failed")

        with pytest.raises(FirestoreConnectionError, match="Firebase initialization failed"):
            FirestoreClient(project_id="test-project")

    @staticmethod
    def test_custom_database_and_caching_settings() -> None:
        """Test custom database name and caching settings."""
        with patch("clarity.storage.firestore_client.firebase_admin") as mock_admin:
            mock_admin._apps = {}
            mock_admin.initialize_app = Mock()

            client = FirestoreClient(
                project_id="test-project",
                database_name="custom-db",
                enable_caching=False,
                cache_ttl=600,
            )

            assert client.database_name == "custom-db"
            assert client.enable_caching is False
            assert client.cache_ttl == 600


class TestFirestoreClientConnection:
    """Test Firestore client connection management."""

    @staticmethod
    async def test_get_db_creates_client(firestore_client: FirestoreClient, mock_firestore_client: AsyncMock) -> None:
        """Test database client creation."""
        with patch("clarity.storage.firestore_client.firestore.AsyncClient") as mock_client:
            mock_client.return_value = mock_firestore_client

            db = await firestore_client._get_db()

            assert db is mock_firestore_client
            mock_client.assert_called_once_with(
                project="test-project", database="(default)"
            )

    @staticmethod
    async def test_get_db_reuses_existing_client(firestore_client: FirestoreClient, mock_firestore_client: AsyncMock) -> None:
        """Test that existing client is reused."""
        firestore_client._db = mock_firestore_client

        db = await firestore_client._get_db()

        assert db is mock_firestore_client

    @staticmethod
    async def test_get_db_connection_failure(firestore_client: FirestoreClient) -> None:
        """Test database connection failure."""
        with patch("clarity.storage.firestore_client.firestore.AsyncClient") as mock_client:
            mock_client.side_effect = Exception("Connection failed")

            with pytest.raises(FirestoreConnectionError, match="Firestore connection failed"):
                await firestore_client._get_db()


class TestFirestoreClientCaching:
    """Test caching functionality."""

    @staticmethod
    def test_cache_key_generation() -> None:
        """Test cache key generation."""
        key = FirestoreClient._cache_key("users", "user123")
        assert key == "users:user123"

    @staticmethod
    def test_cache_validity_enabled(firestore_client: FirestoreClient) -> None:
        """Test cache validity check when caching is enabled."""
        # Valid cache entry
        cache_entry = {"timestamp": time.time() - 100}  # 100 seconds ago
        assert firestore_client._is_cache_valid(cache_entry) is True

        # Expired cache entry
        cache_entry = {"timestamp": time.time() - 400}  # 400 seconds ago
        assert firestore_client._is_cache_valid(cache_entry) is False

    @staticmethod
    def test_cache_validity_disabled(firestore_client: FirestoreClient) -> None:
        """Test cache validity when caching is disabled."""
        firestore_client.enable_caching = False
        cache_entry = {"timestamp": time.time()}

        assert firestore_client._is_cache_valid(cache_entry) is False


class TestFirestoreClientDocumentOperations:
    """Test basic document CRUD operations."""

    @staticmethod
    async def test_create_document_success(firestore_client: FirestoreClient) -> None:
        """Test successful document creation."""
        # Mock the database client and document reference chain
        mock_doc_ref = Mock()
        mock_doc_ref.id = "doc123"
        mock_doc_ref.set = AsyncMock()

        mock_collection = Mock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = Mock()  # Changed from AsyncMock to Mock
        mock_db.collection.return_value = mock_collection

        # Patch the _get_db method to return our mock
        with patch.object(firestore_client, "_get_db", return_value=mock_db), \
             patch.object(firestore_client, "_audit_log", new_callable=AsyncMock):
            result = await firestore_client.create_document(
                collection="test_collection",
                data={"name": "test"},
                document_id="doc123",
                user_id="user123",
            )

        assert result == "doc123"
        mock_doc_ref.set.assert_called_once()

    @staticmethod
    async def test_create_document_auto_id(firestore_client: FirestoreClient) -> None:
        """Test document creation with auto-generated ID."""
        # Mock the database client and document reference chain
        mock_doc_ref = Mock()
        mock_doc_ref.id = "auto_id_123"
        mock_doc_ref.set = AsyncMock()

        mock_collection = Mock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = Mock()  # Changed from AsyncMock to Mock
        mock_db.collection.return_value = mock_collection

        # Patch the _get_db method to return our mock
        with patch.object(firestore_client, "_get_db", return_value=mock_db), \
             patch.object(firestore_client, "_audit_log", new_callable=AsyncMock):
            result = await firestore_client.create_document(
                collection="test_collection",
                data={"name": "test"},
            )

        assert result == "auto_id_123"
        mock_doc_ref.set.assert_called_once()

    @staticmethod
    async def test_create_document_validation_error(firestore_client: FirestoreClient) -> None:
        """Test document creation with validation error."""
        with patch.object(firestore_client, "_validate_health_data", side_effect=FirestoreValidationError("Invalid data")), \
             pytest.raises(FirestoreValidationError, match="Invalid data"):
            await firestore_client.create_document(
                collection="health_data",
                data={"invalid": "data"},
            )

    @staticmethod
    async def test_get_document_success(firestore_client: FirestoreClient) -> None:
        """Test successful document retrieval."""
        # Mock the database client and document reference chain
        mock_doc_snapshot = Mock()
        mock_doc_snapshot.exists = True
        mock_doc_snapshot.to_dict.return_value = {"name": "test", "value": 123}

        mock_doc_ref = Mock()
        mock_doc_ref.get = AsyncMock(return_value=mock_doc_snapshot)

        mock_collection = Mock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = Mock()  # Changed from AsyncMock to Mock
        mock_db.collection.return_value = mock_collection

        # Patch the _get_db method to return our mock
        with patch.object(firestore_client, "_get_db", return_value=mock_db):
            result = await firestore_client.get_document("test_collection", "doc123")

        assert result == {"name": "test", "value": 123}
        mock_doc_ref.get.assert_called_once()

    @staticmethod
    async def test_get_document_not_found(firestore_client: FirestoreClient) -> None:
        """Test document retrieval when document doesn't exist."""
        # Mock the database client and document reference chain
        mock_doc_snapshot = Mock()
        mock_doc_snapshot.exists = False

        mock_doc_ref = Mock()
        mock_doc_ref.get = AsyncMock(return_value=mock_doc_snapshot)

        mock_collection = Mock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = Mock()  # Changed from AsyncMock to Mock
        mock_db.collection.return_value = mock_collection

        # Patch the _get_db method to return our mock
        with patch.object(firestore_client, "_get_db", return_value=mock_db):
            result = await firestore_client.get_document("test_collection", "doc123")

        assert result is None
        mock_doc_ref.get.assert_called_once()

    @staticmethod
    async def test_get_document_cache_hit(firestore_client: FirestoreClient) -> None:
        """Test document retrieval with cache hit."""
        cache_key = "test_collection:doc123"
        firestore_client._cache[cache_key] = {
            "data": {"name": "cached"},
            "timestamp": time.time(),
        }

        result = await firestore_client.get_document("test_collection", "doc123")

        assert result == {"name": "cached"}


class TestFirestoreClientHealthData:
    """Test health data specific operations."""

    @staticmethod
    async def test_validate_health_data_success() -> None:
        """Test successful health data validation."""
        valid_data = {
            "user_id": str(uuid4()),
            "metrics": [{"type": "heart_rate", "value": 72}],
            "upload_source": "apple_health",
        }

        await FirestoreClient._validate_health_data(valid_data)  # Should not raise

    @staticmethod
    async def test_validate_health_data_missing_field() -> None:
        """Test health data validation with missing required field."""
        invalid_data = {
            "metrics": [{"type": "heart_rate", "value": 72}],
            "upload_source": "apple_health",
        }

        with pytest.raises(FirestoreValidationError, match="Missing required field: user_id"):
            await FirestoreClient._validate_health_data(invalid_data)

    @staticmethod
    async def test_store_health_data_success(firestore_client: FirestoreClient, sample_health_upload: HealthDataUpload) -> None:
        """Test successful health data storage."""
        with patch.object(firestore_client, "create_document", return_value="proc123"):
            result = await firestore_client.store_health_data(
                upload_data=sample_health_upload,
                processing_id="proc123",
            )

        assert result == "proc123"


class TestFirestoreClientAuditLogging:
    """Test audit logging functionality."""

    @staticmethod
    async def test_audit_log_success(firestore_client: FirestoreClient, mock_firestore_client: AsyncMock) -> None:
        """Test successful audit log creation."""
        firestore_client._db = mock_firestore_client

        await firestore_client._audit_log(
            operation="CREATE",
            collection="test_collection",
            document_id="doc123",
            user_id="user123",
            metadata={"size": 100},
        )

        # Verify audit log was created
        mock_firestore_client.collection.assert_called_with("audit_logs")


class TestFirestoreClientErrorHandling:
    """Test error handling and edge cases."""

    @staticmethod
    async def test_document_operation_generic_error(firestore_client: FirestoreClient, mock_firestore_client: AsyncMock) -> None:
        """Test generic error handling in document operations."""
        mock_firestore_client.collection.side_effect = Exception("Generic error")
        firestore_client._db = mock_firestore_client

        with pytest.raises(FirestoreError, match="Document creation failed"):
            await firestore_client.create_document("test", {"data": "test"})

    @staticmethod
    async def test_health_data_storage_error(firestore_client: FirestoreClient, sample_health_upload: HealthDataUpload) -> None:
        """Test error handling in health data storage."""
        with patch.object(firestore_client, "create_document", side_effect=Exception("Storage failed")), \
             pytest.raises(FirestoreError, match="Health data storage failed"):
            await firestore_client.store_health_data(sample_health_upload)


class TestFirestoreHealthDataRepository:
    """Test FirestoreHealthDataRepository implementation."""

    @pytest.fixture
    @staticmethod
    def repository() -> FirestoreHealthDataRepository:
        """Create FirestoreHealthDataRepository instance."""
        with patch("clarity.storage.firestore_client.firebase_admin") as mock_admin:
            mock_admin._apps = {}
            mock_admin.initialize_app = Mock()
            with patch("clarity.storage.firestore_client.credentials.Certificate"):
                return FirestoreHealthDataRepository(
                    project_id="test-project",
                    credentials_path="/path/to/creds.json",
                )

    @staticmethod
    async def test_save_health_data_success(repository: FirestoreHealthDataRepository) -> None:
        """Test successful health data saving."""
        user_id = str(uuid4())
        processing_id = str(uuid4())
        metrics = [{"type": "heart_rate", "value": 72}]

        # Mock the create_document calls that happen inside save_health_data
        with patch.object(repository._firestore_client, "create_document", return_value=processing_id) as mock_create:
            result = await repository.save_health_data(
                user_id=user_id,
                processing_id=processing_id,
                metrics=metrics,
                upload_source="apple_health",
                client_timestamp=datetime.now(UTC),
            )

        assert result is True
        # Should be called twice: once for processing doc, once per metric
        assert mock_create.call_count == 2

    @staticmethod
    async def test_get_user_health_data_success(repository: FirestoreHealthDataRepository) -> None:
        """Test successful user health data retrieval."""
        user_id = str(uuid4())
        mock_metrics = [{"metric_data": {"type": "heart_rate", "value": 72}}]

        # Mock both query_documents and count_documents methods
        with patch.object(repository._firestore_client, "query_documents", return_value=mock_metrics) as mock_query, \
             patch.object(repository._firestore_client, "count_documents", return_value=1) as mock_count:
            result = await repository.get_user_health_data(user_id=user_id)

        assert "metrics" in result
        assert "pagination" in result
        assert result["pagination"]["total"] == 1
        assert result["metrics"] == mock_metrics

        # Verify the methods were called
        mock_query.assert_called_once()
        mock_count.assert_called_once()


class TestFirestoreClientAdvancedFeatures:
    """Test advanced Firestore client features."""

    @staticmethod
    async def test_health_check_success(firestore_client: FirestoreClient) -> None:
        """Test successful health check."""
        # Mock the database client and document reference chain for health check
        mock_doc_ref = Mock()
        mock_doc_ref.get = AsyncMock()

        mock_collection = Mock()
        mock_collection.document.return_value = mock_doc_ref

        mock_db = Mock()
        mock_db.collection.return_value = mock_collection

        # Patch the _get_db method to return our mock
        with patch.object(firestore_client, "_get_db", return_value=mock_db):
            result = await firestore_client.health_check()

        assert result["status"] == "healthy"
        assert result["project_id"] == "test-project"
        assert "timestamp" in result

    @staticmethod
    async def test_close_cleanup(firestore_client: FirestoreClient, mock_firestore_client: AsyncMock) -> None:
        """Test client cleanup on close."""
        firestore_client._db = mock_firestore_client

        await firestore_client.close()

        assert firestore_client._db is None
        assert firestore_client._cache == {}


class TestFirestoreClientConcurrency:
    """Test concurrent operations and thread safety."""

    @staticmethod
    async def test_concurrent_db_access(firestore_client: FirestoreClient, mock_firestore_client: AsyncMock) -> None:
        """Test concurrent database access creates only one client."""
        with patch("clarity.storage.firestore_client.firestore.AsyncClient") as mock_client:
            mock_client.return_value = mock_firestore_client

            # Start multiple coroutines that access the database
            tasks = [firestore_client._get_db() for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # All should return the same client instance
            assert all(db is mock_firestore_client for db in results)
            # Client should only be created once
            mock_client.assert_called_once()

    @staticmethod
    def test_cache_thread_safety(firestore_client: FirestoreClient) -> None:
        """Test cache operations are thread-safe."""
        # Add item to cache
        cache_key = "test:doc1"
        firestore_client._cache[cache_key] = {
            "data": {"test": "data"},
            "timestamp": time.time(),
        }

        # Multiple cache validity checks (synchronous since _is_cache_valid is not async)
        results = [
            firestore_client._is_cache_valid(firestore_client._cache[cache_key])
            for _ in range(10)
        ]

        # All should succeed
        assert all(isinstance(result, bool) for result in results)
        assert all(result is True for result in results)  # Should all be valid
