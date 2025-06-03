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
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from clarity.models.health_data import (
    ActivityData,
    BiometricData,
    HealthDataUpload,
    HealthMetric,
    ProcessingStatus,
)
from clarity.storage.firestore_client import (
    DocumentNotFoundError,
    FirestoreClient,
    FirestoreConnectionError,
    FirestoreError,
    FirestoreHealthDataRepository,
    FirestorePermissionError,
    FirestoreValidationError,
)


@pytest.fixture
def mock_firebase_admin():
    """Mock Firebase Admin SDK."""
    with patch("clarity.storage.firestore_client.firebase_admin") as mock_admin:
        mock_admin._apps = {}  # Empty apps list
        mock_admin.initialize_app = Mock()
        yield mock_admin


@pytest.fixture
def mock_firestore_client():
    """Mock Firestore AsyncClient."""
    with patch("clarity.storage.firestore_client.firestore.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_credentials():
    """Mock Firebase credentials."""
    with patch("clarity.storage.firestore_client.credentials.Certificate") as mock_creds:
        yield mock_creds


@pytest.fixture
def firestore_client(mock_firebase_admin, mock_credentials):
    """Create FirestoreClient instance with mocked dependencies."""
    return FirestoreClient(
        project_id="test-project",
        credentials_path="/path/to/creds.json",
        enable_caching=True,
        cache_ttl=300,
    )


@pytest.fixture
def sample_health_data():
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
def sample_health_upload():
    """Sample HealthDataUpload for testing."""
    user_id = uuid4()
    return HealthDataUpload(
        user_id=user_id,
        metrics=[
            HealthMetric(
                metric_id=uuid4(),
                metric_type="heart_rate",
                timestamp=datetime.now(UTC),
                value=72.0,
                unit="bpm",
                biometric_data=BiometricData(heart_rate=72.0),
            )
        ],
        upload_source="apple_health",
        client_timestamp=datetime.now(UTC),
    )


class TestFirestoreClientInitialization:
    """Test FirestoreClient initialization and setup."""

    def test_init_with_credentials_path(self, mock_firebase_admin, mock_credentials):
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

    def test_init_without_credentials_path(self, mock_firebase_admin):
        """Test initialization without credentials path (uses ADC)."""
        client = FirestoreClient(project_id="test-project")

        assert client.project_id == "test-project"
        mock_firebase_admin.initialize_app.assert_called_once_with()

    def test_init_firebase_already_initialized(self, mock_firebase_admin):
        """Test initialization when Firebase is already initialized."""
        mock_firebase_admin._apps = {"default": Mock()}  # Non-empty apps

        client = FirestoreClient(project_id="test-project")
        assert client.project_id == "test-project"

    def test_init_firebase_failure(self, mock_firebase_admin):
        """Test Firebase initialization failure."""
        mock_firebase_admin.initialize_app.side_effect = Exception("Init failed")

        with pytest.raises(FirestoreConnectionError, match="Firebase initialization failed"):
            FirestoreClient(project_id="test-project")

    def test_custom_database_and_caching_settings(self, mock_firebase_admin):
        """Test custom database name and caching settings."""
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

    @pytest.mark.asyncio
    async def test_get_db_creates_client(self, firestore_client, mock_firestore_client):
        """Test database client creation."""
        with patch("clarity.storage.firestore_client.firestore.AsyncClient") as mock_client:
            mock_client.return_value = mock_firestore_client

            db = await firestore_client._get_db()

            assert db is mock_firestore_client
            mock_client.assert_called_once_with(
                project="test-project", database="(default)"
            )

    @pytest.mark.asyncio
    async def test_get_db_reuses_existing_client(self, firestore_client, mock_firestore_client):
        """Test that existing client is reused."""
        firestore_client._db = mock_firestore_client

        db = await firestore_client._get_db()

        assert db is mock_firestore_client

    @pytest.mark.asyncio
    async def test_get_db_connection_failure(self, firestore_client):
        """Test database connection failure."""
        with patch("clarity.storage.firestore_client.firestore.AsyncClient") as mock_client:
            mock_client.side_effect = Exception("Connection failed")

            with pytest.raises(FirestoreConnectionError, match="Firestore connection failed"):
                await firestore_client._get_db()


class TestFirestoreClientCaching:
    """Test caching functionality."""

    def test_cache_key_generation(self):
        """Test cache key generation."""
        key = FirestoreClient._cache_key("users", "user123")
        assert key == "users:user123"

    def test_cache_validity_enabled(self, firestore_client):
        """Test cache validity check when caching is enabled."""
        import time

        # Valid cache entry
        cache_entry = {"timestamp": time.time() - 100}  # 100 seconds ago
        assert firestore_client._is_cache_valid(cache_entry) is True

        # Expired cache entry
        cache_entry = {"timestamp": time.time() - 400}  # 400 seconds ago
        assert firestore_client._is_cache_valid(cache_entry) is False

    def test_cache_validity_disabled(self, firestore_client):
        """Test cache validity when caching is disabled."""
        firestore_client.enable_caching = False
        cache_entry = {"timestamp": time.time()}

        assert firestore_client._is_cache_valid(cache_entry) is False


class TestFirestoreClientDocumentOperations:
    """Test basic document CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_document_success(self, firestore_client, mock_firestore_client):
        """Test successful document creation."""
        mock_doc_ref = Mock()
        mock_doc_ref.id = "doc123"
        mock_firestore_client.collection.return_value.document.return_value = mock_doc_ref

        firestore_client._db = mock_firestore_client

        with patch.object(firestore_client, "_audit_log", new_callable=AsyncMock):
            result = await firestore_client.create_document(
                collection="test_collection",
                data={"name": "test"},
                document_id="doc123",
                user_id="user123",
            )

        assert result == "doc123"
        mock_doc_ref.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_document_auto_id(self, firestore_client, mock_firestore_client):
        """Test document creation with auto-generated ID."""
        mock_doc_ref = Mock()
        mock_doc_ref.id = "auto_id_123"
        mock_firestore_client.collection.return_value.document.return_value = mock_doc_ref

        firestore_client._db = mock_firestore_client

        with patch.object(firestore_client, "_audit_log", new_callable=AsyncMock):
            result = await firestore_client.create_document(
                collection="test_collection",
                data={"name": "test"},
            )

        assert result == "auto_id_123"

    @pytest.mark.asyncio
    async def test_create_document_validation_error(self, firestore_client):
        """Test document creation with validation error."""
        with patch.object(firestore_client, "_validate_health_data", side_effect=FirestoreValidationError("Invalid data")):
            with pytest.raises(FirestoreValidationError, match="Invalid data"):
                await firestore_client.create_document(
                    collection="health_data",
                    data={"invalid": "data"},
                )

    @pytest.mark.asyncio
    async def test_get_document_success(self, firestore_client, mock_firestore_client):
        """Test successful document retrieval."""
        mock_doc = Mock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {"name": "test", "value": 123}
        mock_firestore_client.collection.return_value.document.return_value.get.return_value = mock_doc

        firestore_client._db = mock_firestore_client

        result = await firestore_client.get_document("test_collection", "doc123")

        assert result == {"name": "test", "value": 123}

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, firestore_client, mock_firestore_client):
        """Test document retrieval when document doesn't exist."""
        mock_doc = Mock()
        mock_doc.exists = False
        mock_firestore_client.collection.return_value.document.return_value.get.return_value = mock_doc

        firestore_client._db = mock_firestore_client

        result = await firestore_client.get_document("test_collection", "doc123")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_document_cache_hit(self, firestore_client):
        """Test document retrieval with cache hit."""
        import time

        cache_key = "test_collection:doc123"
        firestore_client._cache[cache_key] = {
            "data": {"name": "cached"},
            "timestamp": time.time(),
        }

        result = await firestore_client.get_document("test_collection", "doc123")

        assert result == {"name": "cached"}

    @pytest.mark.asyncio
    async def test_update_document_success(self, firestore_client, mock_firestore_client):
        """Test successful document update."""
        mock_doc_ref = Mock()
        mock_firestore_client.collection.return_value.document.return_value = mock_doc_ref

        firestore_client._db = mock_firestore_client

        with patch.object(firestore_client, "_audit_log", new_callable=AsyncMock):
            result = await firestore_client.update_document(
                collection="test_collection",
                document_id="doc123",
                data={"name": "updated"},
                user_id="user123",
            )

        assert result is True
        mock_doc_ref.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_document_not_found(self, firestore_client, mock_firestore_client):
        """Test document update when document doesn't exist."""
        from google.cloud.exceptions import NotFound

        mock_doc_ref = Mock()
        mock_doc_ref.update.side_effect = NotFound("Document not found")
        mock_firestore_client.collection.return_value.document.return_value = mock_doc_ref

        firestore_client._db = mock_firestore_client

        with patch.object(firestore_client, "_audit_log", new_callable=AsyncMock):
            result = await firestore_client.update_document(
                collection="test_collection",
                document_id="doc123",
                data={"name": "updated"},
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_document_success(self, firestore_client, mock_firestore_client):
        """Test successful document deletion."""
        mock_doc_ref = Mock()
        mock_firestore_client.collection.return_value.document.return_value = mock_doc_ref

        firestore_client._db = mock_firestore_client

        with patch.object(firestore_client, "_audit_log", new_callable=AsyncMock):
            result = await firestore_client.delete_document(
                collection="test_collection",
                document_id="doc123",
                user_id="user123",
            )

        assert result is True
        mock_doc_ref.delete.assert_called_once()


class TestFirestoreClientHealthData:
    """Test health data specific operations."""

    @pytest.mark.asyncio
    async def test_validate_health_data_success(self):
        """Test successful health data validation."""
        valid_data = {
            "user_id": str(uuid4()),
            "metrics": [{"type": "heart_rate", "value": 72}],
            "upload_source": "apple_health",
        }

        await FirestoreClient._validate_health_data(valid_data)  # Should not raise

    @pytest.mark.asyncio
    async def test_validate_health_data_missing_field(self):
        """Test health data validation with missing required field."""
        invalid_data = {
            "metrics": [{"type": "heart_rate", "value": 72}],
            "upload_source": "apple_health",
        }

        with pytest.raises(FirestoreValidationError, match="Missing required field: user_id"):
            await FirestoreClient._validate_health_data(invalid_data)

    @pytest.mark.asyncio
    async def test_validate_health_data_invalid_user_id(self):
        """Test health data validation with invalid user_id format."""
        invalid_data = {
            "user_id": "invalid_uuid",
            "metrics": [{"type": "heart_rate", "value": 72}],
            "upload_source": "apple_health",
        }

        with pytest.raises(FirestoreValidationError, match="Invalid user_id format"):
            await FirestoreClient._validate_health_data(invalid_data)

    @pytest.mark.asyncio
    async def test_validate_health_data_empty_metrics(self):
        """Test health data validation with empty metrics."""
        invalid_data = {
            "user_id": str(uuid4()),
            "metrics": [],
            "upload_source": "apple_health",
        }

        with pytest.raises(FirestoreValidationError, match="Metrics must be a non-empty list"):
            await FirestoreClient._validate_health_data(invalid_data)

    @pytest.mark.asyncio
    async def test_store_health_data_success(self, firestore_client, sample_health_upload):
        """Test successful health data storage."""
        with patch.object(firestore_client, "create_document", return_value="proc123") as mock_create:
            result = await firestore_client.store_health_data(
                upload_data=sample_health_upload,
                processing_id="proc123",
            )

        assert result == "proc123"
        assert mock_create.call_count == 2  # health_data + processing_jobs

    @pytest.mark.asyncio
    async def test_store_health_data_auto_processing_id(self, firestore_client, sample_health_upload):
        """Test health data storage with auto-generated processing ID."""
        with patch.object(firestore_client, "create_document", return_value="auto_id") as mock_create:
            result = await firestore_client.store_health_data(upload_data=sample_health_upload)

        assert len(result) == 36  # UUID length
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_get_processing_status(self, firestore_client):
        """Test getting processing status."""
        expected_status = {
            "processing_id": "proc123",
            "status": "PROCESSING",
            "created_at": datetime.now(UTC),
        }

        with patch.object(firestore_client, "get_document", return_value=expected_status):
            result = await firestore_client.get_processing_status("proc123")

        assert result == expected_status


class TestFirestoreClientAuditLogging:
    """Test audit logging functionality."""

    @pytest.mark.asyncio
    async def test_audit_log_success(self, firestore_client, mock_firestore_client):
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

    @pytest.mark.asyncio
    async def test_audit_log_failure_no_exception(self, firestore_client, mock_firestore_client):
        """Test audit log failure doesn't raise exception."""
        mock_firestore_client.collection.side_effect = Exception("Audit failed")
        firestore_client._db = mock_firestore_client

        # Should not raise exception
        await firestore_client._audit_log(
            operation="CREATE",
            collection="test_collection",
            document_id="doc123",
        )


class TestFirestoreClientErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_document_operation_generic_error(self, firestore_client, mock_firestore_client):
        """Test generic error handling in document operations."""
        mock_firestore_client.collection.side_effect = Exception("Generic error")
        firestore_client._db = mock_firestore_client

        with pytest.raises(FirestoreError, match="Document creation failed"):
            await firestore_client.create_document("test", {"data": "test"})

    @pytest.mark.asyncio
    async def test_health_data_storage_error(self, firestore_client, sample_health_upload):
        """Test error handling in health data storage."""
        with patch.object(firestore_client, "create_document", side_effect=Exception("Storage failed")):
            with pytest.raises(FirestoreError, match="Health data storage failed"):
                await firestore_client.store_health_data(sample_health_upload)


class TestFirestoreHealthDataRepository:
    """Test FirestoreHealthDataRepository implementation."""

    @pytest.fixture
    def repository(self, mock_firebase_admin, mock_credentials):
        """Create FirestoreHealthDataRepository instance."""
        return FirestoreHealthDataRepository(
            project_id="test-project",
            credentials_path="/path/to/creds.json",
        )

    @pytest.mark.asyncio
    async def test_save_health_data_success(self, repository):
        """Test successful health data saving."""
        user_id = str(uuid4())
        processing_id = str(uuid4())
        metrics = [{"type": "heart_rate", "value": 72}]

        with patch.object(repository.client, "store_health_data", return_value=processing_id) as mock_store:
            result = await repository.save_health_data(
                user_id=user_id,
                processing_id=processing_id,
                metrics=metrics,
                upload_source="apple_health",
                client_timestamp=datetime.now(UTC),
            )

        assert result is True
        mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_health_data_error(self, repository):
        """Test health data saving error."""
        with patch.object(repository.client, "store_health_data", side_effect=Exception("Storage failed")):
            result = await repository.save_health_data(
                user_id=str(uuid4()),
                processing_id=str(uuid4()),
                metrics=[],
                upload_source="test",
                client_timestamp=datetime.now(UTC),
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_get_user_health_data_success(self, repository):
        """Test successful user health data retrieval."""
        user_id = str(uuid4())
        expected_data = {
            "data": [{"metric_type": "heart_rate", "value": 72}],
            "total_count": 1,
            "has_more": False,
        }

        with patch.object(repository.client, "query_documents", return_value=[{"metrics": []}]) as mock_query:
            result = await repository.get_user_health_data(user_id=user_id)

        assert "data" in result
        assert "total_count" in result
        assert "has_more" in result

    @pytest.mark.asyncio
    async def test_get_processing_status_success(self, repository):
        """Test successful processing status retrieval."""
        processing_id = str(uuid4())
        user_id = str(uuid4())
        expected_status = {"status": "PROCESSING", "processing_id": processing_id}

        with patch.object(repository.client, "get_processing_status", return_value=expected_status):
            result = await repository.get_processing_status(processing_id, user_id)

        assert result == expected_status

    @pytest.mark.asyncio
    async def test_delete_health_data_success(self, repository):
        """Test successful health data deletion."""
        user_id = str(uuid4())
        processing_id = str(uuid4())

        with patch.object(repository.client, "delete_document", return_value=True) as mock_delete:
            result = await repository.delete_health_data(user_id, processing_id)

        assert result is True
        assert mock_delete.call_count >= 1

    @pytest.mark.asyncio
    async def test_delete_health_data_error(self, repository):
        """Test health data deletion error."""
        with patch.object(repository.client, "delete_document", side_effect=Exception("Delete failed")):
            result = await repository.delete_health_data(str(uuid4()), str(uuid4()))

        assert result is False


class TestFirestoreClientAdvancedFeatures:
    """Test advanced Firestore client features."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, firestore_client, mock_firestore_client):
        """Test successful health check."""
        firestore_client._db = mock_firestore_client

        result = await firestore_client.health_check()

        assert result["status"] == "healthy"
        assert result["project_id"] == "test-project"
        assert "response_time_ms" in result

    @pytest.mark.asyncio
    async def test_health_check_failure(self, firestore_client):
        """Test health check failure."""
        with patch.object(firestore_client, "_get_db", side_effect=Exception("Connection failed")):
            result = await firestore_client.health_check()

        assert result["status"] == "unhealthy"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_close_cleanup(self, firestore_client, mock_firestore_client):
        """Test client cleanup on close."""
        firestore_client._db = mock_firestore_client

        await firestore_client.close()

        assert firestore_client._db is None
        assert firestore_client._cache == {}


class TestFirestoreClientConcurrency:
    """Test concurrent operations and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_db_access(self, firestore_client, mock_firestore_client):
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

    @pytest.mark.asyncio
    async def test_cache_thread_safety(self, firestore_client):
        """Test cache operations are thread-safe."""
        import time

        # Add item to cache
        cache_key = "test:doc1"
        firestore_client._cache[cache_key] = {
            "data": {"test": "data"},
            "timestamp": time.time(),
        }

        # Multiple cache validity checks
        tasks = [
            firestore_client._is_cache_valid(firestore_client._cache[cache_key])
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        assert all(isinstance(result, bool) for result in results)
