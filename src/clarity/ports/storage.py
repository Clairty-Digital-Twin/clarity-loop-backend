"""Storage port interface following Clean Architecture principles.

This module defines the abstract interface for storage operations,
following the Dependency Inversion Principle where high-level modules
depend on abstractions, not concrete implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class StoragePort(ABC):
    """Abstract interface for storage operations.
    
    This interface defines the contract that all storage implementations
    must follow, enabling dependency injection and easier testing.
    
    Follows the Interface Segregation Principle by providing only
    the methods that clients actually need.
    """
    
    @abstractmethod
    def get_collection(self, name: str) -> Any:
        """Get a collection reference by name.
        
        Args:
            name: The name of the collection
            
        Returns:
            A collection reference object
        """
        pass
    
    @abstractmethod
    def create_document(self, collection: str, data: Dict[str, Any]) -> str:
        """Create a new document in the specified collection.
        
        Args:
            collection: Name of the collection
            data: Document data to store
            
        Returns:
            The ID of the created document
        """
        pass
    
    @abstractmethod
    def get_document(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID.
        
        Args:
            collection: Name of the collection
            doc_id: Document ID
            
        Returns:
            Document data if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update_document(self, collection: str, doc_id: str, data: Dict[str, Any]) -> None:
        """Update an existing document.
        
        Args:
            collection: Name of the collection
            doc_id: Document ID
            data: Updated document data
        """
        pass
    
    @abstractmethod
    def delete_document(self, collection: str, doc_id: str) -> None:
        """Delete a document by ID.
        
        Args:
            collection: Name of the collection
            doc_id: Document ID
        """
        pass
    
    @abstractmethod
    def query_documents(
        self, 
        collection: str, 
        filters: Optional[List[Dict[str, Any]]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query documents with optional filters.
        
        Args:
            collection: Name of the collection
            filters: Optional list of filter conditions
            limit: Optional limit on number of results
            
        Returns:
            List of matching documents
        """
        pass


class CollectionPort(ABC):
    """Abstract interface for collection operations.
    
    Represents a collection reference that can perform operations
    on documents within that collection.
    """
    
    @abstractmethod
    def add(self, data: Dict[str, Any]) -> str:
        """Add a document to the collection.
        
        Args:
            data: Document data
            
        Returns:
            Document ID
        """
        pass
    
    @abstractmethod
    def document(self, doc_id: str) -> Any:
        """Get a document reference.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document reference
        """
        pass
    
    @abstractmethod
    def where(self, field: str, operator: str, value: Any) -> Any:
        """Create a query with a filter condition.
        
        Args:
            field: Field name
            operator: Comparison operator
            value: Value to compare against
            
        Returns:
            Query object
        """
        pass 