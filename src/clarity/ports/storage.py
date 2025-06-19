"""Storage port interface following Clean Architecture principles.

This module defines the abstract interface for storage operations,
following the Dependency Inversion Principle where high-level modules
depend on abstractions, not concrete implementations.
"""

# removed - breaks FastAPI

from abc import ABC, abstractmethod
from typing import Any


class StoragePort(ABC):
    """Abstract interface for storage operations.

    This interface defines the contract that all storage implementations
    must follow, enabling dependency injection and easier testing.

    Follows the Interface Segregation Principle by providing only
    the methods that clients actually need.
    """

    @abstractmethod
    def get_collection(self, name: str) -> object:
        """Get a collection reference by name.

        Args:
            name: The name of the collection

        Returns:
            A collection reference object
        """

    @abstractmethod
    def create_document(self, collection: str, data: dict[str, Any]) -> str:
        """Create a new document in the specified collection.

        Args:
            collection: Name of the collection
            data: Document data to store

        Returns:
            The ID of the created document
        """

    @abstractmethod
    def get_document(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a document by ID.

        Args:
            collection: Name of the collection
            doc_id: Document ID

        Returns:
            Document data if found, None otherwise
        """

    @abstractmethod
    def update_document(
        self, collection: str, doc_id: str, data: dict[str, Any]
    ) -> None:
        """Update an existing document.

        Args:
            collection: Name of the collection
            doc_id: Document ID
            data: Updated document data
        """

    @abstractmethod
    def delete_document(self, collection: str, doc_id: str) -> None:
        """Delete a document by ID.

        Args:
            collection: Name of the collection
            doc_id: Document ID
        """

    @abstractmethod
    def query_documents(
        self,
        collection: str,
        filters: list[dict[str, Any]] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query documents with optional filters.

        Args:
            collection: Name of the collection
            filters: Optional list of filter conditions
            limit: Optional limit on number of results

        Returns:
            List of matching documents
        """


class CollectionPort(ABC):
    """Abstract interface for collection operations.

    Represents a collection reference that can perform operations
    on documents within that collection.
    """

    @abstractmethod
    def add(self, data: dict[str, Any]) -> str:
        """Add a document to the collection.

        Args:
            data: Document data

        Returns:
            Document ID
        """

    @abstractmethod
    def document(self, doc_id: str) -> object:
        """Get a document reference.

        Args:
            doc_id: Document ID

        Returns:
            Document reference
        """

    @abstractmethod
    def where(self, field: str, operator: str, value: Any) -> object:
        """Create a query with a filter condition.

        Args:
            field: Field name
            operator: Comparison operator
            value: Value to compare against

        Returns:
            Query object
        """


class CloudStoragePort(ABC):
    """Abstract interface for cloud storage operations.

    This interface abstracts cloud storage operations (like GCS)
    to enable dependency injection and easier testing.
    """

    @abstractmethod
    def bucket(self, bucket_name: str) -> object:
        """Get a bucket reference.

        Args:
            bucket_name: Name of the bucket

        Returns:
            Bucket reference object
        """

    @abstractmethod
    def upload_json(
        self,
        bucket_name: str,
        blob_path: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Upload JSON data to cloud storage.

        Args:
            bucket_name: Name of the bucket
            blob_path: Path for the blob
            data: JSON data to upload
            metadata: Optional metadata

        Returns:
            Full path/URL of uploaded object
        """

    @abstractmethod
    def get_raw_data_bucket_name(self) -> str:
        """Get the name of the raw data bucket.

        Returns:
            Bucket name for raw data storage
        """

    @abstractmethod
    async def upload_file(
        self, file_data: bytes, file_path: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Upload a file to cloud storage.

        Args:
            file_data: Binary data to upload
            file_path: Path for the file in cloud storage
            metadata: Optional metadata for the file

        Returns:
            Full path/URL of uploaded object
        """
