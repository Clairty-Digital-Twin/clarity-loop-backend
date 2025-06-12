"""Fake storage implementations for testing.

These fakes provide simple in-memory implementations of storage interfaces,
following the "Use Fakes Instead of Mocks" best practice from testing literature.

Fakes are preferred over mocks because they:
1. Test behavior rather than implementation details
2. Are more resilient to refactoring
3. Provide deterministic, fast tests
4. Don't require complex mock setup
"""

from __future__ import annotations

import json
from typing import Any
import uuid

from clarity.ports.storage import CloudStoragePort, CollectionPort, StoragePort


class FakeCollection(CollectionPort):
    """Fake collection implementation for testing.

    Provides a simple in-memory collection that mimics
    the behavior of a real document collection.
    """

    def __init__(self, data: dict[str, dict[str, Any]]) -> None:
        """Initialize with shared data storage.

        Args:
            data: Shared dictionary to store documents
        """
        self.data = data

    def add(self, data: dict[str, Any]) -> str:
        """Add a document to the collection.

        Args:
            data: Document data

        Returns:
            Generated document ID
        """
        doc_id = str(uuid.uuid4())
        self.data[doc_id] = data.copy()
        return doc_id

    def document(self, doc_id: str) -> FakeDocument:
        """Get a document reference.

        Args:
            doc_id: Document ID

        Returns:
            Fake document reference
        """
        return FakeDocument(self.data, doc_id)

    def where(self, field: str, operator: str, value: object) -> FakeQuery:
        """Create a query with a filter condition.

        Args:
            field: Field name
            operator: Comparison operator ('==', '>', '<', etc.)
            value: Value to compare against

        Returns:
            Fake query object
        """
        return FakeQuery(self.data, [(field, operator, value)])


class FakeDocument:
    """Fake document reference for testing."""

    def __init__(self, data: dict[str, dict[str, Any]], doc_id: str) -> None:
        """Initialize document reference.

        Args:
            data: Shared data storage
            doc_id: Document ID
        """
        self.data = data
        self.doc_id = doc_id

    def get(self) -> FakeDocumentSnapshot:
        """Get document snapshot.

        Returns:
            Fake document snapshot
        """
        return FakeDocumentSnapshot(
            self.data.get(self.doc_id), self.doc_id, exists=self.doc_id in self.data
        )

    def set(self, data: dict[str, Any]) -> None:
        """Set document data.

        Args:
            data: Document data to set
        """
        self.data[self.doc_id] = data.copy()

    def update(self, data: dict[str, Any]) -> None:
        """Update document data.

        Args:
            data: Fields to update
        """
        if self.doc_id in self.data:
            self.data[self.doc_id].update(data)

    def delete(self) -> None:
        """Delete the document."""
        if self.doc_id in self.data:
            del self.data[self.doc_id]


class FakeDocumentSnapshot:
    """Fake document snapshot for testing."""

    def __init__(
        self, data: dict[str, Any] | None, doc_id: str, *, exists: bool
    ) -> None:
        """Initialize document snapshot.

        Args:
            data: Document data
            doc_id: Document ID
            exists: Whether document exists
        """
        self._data = data or {}
        self.id = doc_id
        self.exists = exists

    def to_dict(self) -> dict[str, Any]:
        """Get document data as dictionary.

        Returns:
            Document data
        """
        return self._data.copy()


class FakeQuery:
    """Fake query for testing."""

    def __init__(
        self, data: dict[str, dict[str, Any]], filters: list[tuple[str, str, Any]]
    ) -> None:
        """Initialize query.

        Args:
            data: Data to query
            filters: List of filter conditions
        """
        self.data = data
        self.filters = filters
        self._limit: int | None = None

    def limit(self, count: int) -> FakeQuery:
        """Add limit to query.

        Args:
            count: Maximum number of results

        Returns:
            Self for chaining
        """
        self._limit = count
        return self

    def get(self) -> list[FakeDocumentSnapshot]:
        """Execute query and get results.

        Returns:
            List of document snapshots matching the query
        """
        results = []

        for doc_id, doc_data in self.data.items():
            if self._matches_filters(doc_data):
                results.append(FakeDocumentSnapshot(doc_data, doc_id, exists=True))

        if self._limit is not None:
            results = results[: self._limit]

        return results

    def _matches_filters(self, doc_data: dict[str, Any]) -> bool:
        """Check if document matches all filters.

        Args:
            doc_data: Document data to check

        Returns:
            True if document matches all filters
        """
        for field, operator, value in self.filters:
            if field not in doc_data:
                return False

            doc_value = doc_data[field]

            if operator == "==":
                if doc_value != value:
                    return False
            elif operator == ">":
                if not (doc_value > value):
                    return False
            elif operator == "<":
                if not (doc_value < value):
                    return False
            elif operator == ">=":
                if not (doc_value >= value):
                    return False
            elif operator == "<=":
                if not (doc_value <= value):
                    return False
            elif operator == "!=" and doc_value == value:
                return False
            # Add more operators as needed

        return True


class FakeStorage(StoragePort):
    """Fake storage implementation for testing.

    Provides a simple in-memory storage that implements the StoragePort
    interface. This allows tests to run quickly and deterministically
    without requiring external storage systems.

    Example:
        storage = FakeStorage()
        collection = storage.get_collection("users")
        doc_id = collection.add({"name": "John", "age": 30})
    """

    def __init__(self) -> None:
        """Initialize with empty storage."""
        self.collections: dict[str, dict[str, dict[str, Any]]] = {}

    def get_collection(self, name: str) -> FakeCollection:
        """Get a collection reference by name.

        Args:
            name: Collection name

        Returns:
            Fake collection reference
        """
        if name not in self.collections:
            self.collections[name] = {}
        return FakeCollection(self.collections[name])

    def create_document(self, collection: str, data: dict[str, Any]) -> str:
        """Create a new document in the specified collection.

        Args:
            collection: Collection name
            data: Document data

        Returns:
            Generated document ID
        """
        coll = self.get_collection(collection)
        return coll.add(data)

    def get_document(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a document by ID.

        Args:
            collection: Collection name
            doc_id: Document ID

        Returns:
            Document data if found, None otherwise
        """
        if collection not in self.collections:
            return None

        if doc_id not in self.collections[collection]:
            return None

        return self.collections[collection][doc_id].copy()

    def update_document(
        self, collection: str, doc_id: str, data: dict[str, Any]
    ) -> None:
        """Update an existing document.

        Args:
            collection: Collection name
            doc_id: Document ID
            data: Updated document data
        """
        if collection not in self.collections:
            self.collections[collection] = {}

        if doc_id in self.collections[collection]:
            self.collections[collection][doc_id].update(data)

    def delete_document(self, collection: str, doc_id: str) -> None:
        """Delete a document by ID.

        Args:
            collection: Collection name
            doc_id: Document ID
        """
        if collection in self.collections and doc_id in self.collections[collection]:
            del self.collections[collection][doc_id]

    def query_documents(
        self,
        collection: str,
        filters: list[dict[str, Any]] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query documents with optional filters.

        Args:
            collection: Collection name
            filters: Optional list of filter conditions
            limit: Optional limit on number of results

        Returns:
            List of matching documents
        """
        if collection not in self.collections:
            return []

        results = []
        collection_data = self.collections[collection]

        for doc_data in collection_data.values():
            if not filters:
                results.append(doc_data.copy())
            else:
                # Simple filter implementation - can be enhanced as needed
                matches = True
                for filter_dict in filters:
                    field = filter_dict.get("field")
                    operator = filter_dict.get("operator", "==")
                    value = filter_dict.get("value")

                    if field and field in doc_data:
                        doc_value = doc_data[field]
                        if operator == "==" and doc_value != value:
                            matches = False
                            break
                        # Add more operators as needed

                if matches:
                    results.append(doc_data.copy())

        if limit:
            results = results[:limit]

        return results


class FakeCloudStorage(CloudStoragePort):
    """Fake implementation of CloudStoragePort for testing.

    Provides a fast, in-memory implementation that doesn't require
    actual cloud credentials or network access.
    """

    def __init__(self, bucket_name: str = "test-raw-data-bucket") -> None:
        self._bucket_name = bucket_name
        self._stored_data: dict[str, dict[str, Any]] = {}

    def bucket(self, bucket_name: str) -> FakeBucket:
        """Get a fake bucket reference."""
        return FakeBucket(bucket_name, self._stored_data)

    def upload_json(
        self,
        bucket_name: str,
        blob_path: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Upload JSON data to fake storage."""
        full_path = f"{bucket_name}/{blob_path}"
        self._stored_data[full_path] = {
            "data": data,
            "metadata": metadata or {},
            "content_type": "application/json",
        }
        return f"gs://{full_path}"

    def get_raw_data_bucket_name(self) -> str:
        """Get the configured bucket name."""
        return self._bucket_name

    async def upload_file(
        self, file_data: bytes, file_path: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Upload a file to fake storage.

        Args:
            file_data: Binary data to upload
            file_path: Path for the file in cloud storage
            metadata: Optional metadata for the file

        Returns:
            Full path/URL of uploaded object
        """
        full_path = f"{self._bucket_name}/{file_path}"
        self._stored_data[full_path] = {
            "data": file_data,
            "metadata": metadata or {},
            "content_type": "application/octet-stream",
        }
        return f"gs://{full_path}"


class FakeBucket:
    """Fake bucket implementation."""

    def __init__(self, name: str, storage: dict[str, dict[str, Any]]) -> None:
        self.name = name
        self._storage = storage

    def blob(self, blob_path: str) -> FakeBlob:
        """Get a fake blob reference."""
        return FakeBlob(f"{self.name}/{blob_path}", self._storage)


class FakeBlob:
    """Fake blob implementation."""

    def __init__(self, full_path: str, storage: dict[str, dict[str, Any]]) -> None:
        self.full_path = full_path
        self._storage = storage

    def upload_from_string(
        self, data: str, content_type: str = "application/json"
    ) -> None:
        """Upload string data to fake storage."""
        parsed_data = json.loads(data) if content_type == "application/json" else data

        self._storage[self.full_path] = {
            "data": parsed_data,
            "content_type": content_type,
        }
