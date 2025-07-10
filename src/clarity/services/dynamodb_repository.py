"""DynamoDB Repository Pattern Implementation - Following SOLID principles.

Repository pattern for data access, separating business logic from data access.
Each repository handles a specific entity type with focused responsibilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from datetime import UTC, datetime
import logging
from typing import Any, Generic, TypeVar
import uuid

from clarity.services.dynamodb_connection import DynamoDBConnection

logger = logging.getLogger(__name__)

# Generic type for entity
T = TypeVar("T")


class IRepository(ABC, Generic[T]):
    """Base repository interface following Interface Segregation Principle."""

    @abstractmethod
    async def create(self, entity: Any) -> str:
        """Create a new entity."""

    @abstractmethod
    async def get(self, id: str) -> Any:
        """Get entity by ID."""

    @abstractmethod
    async def update(self, id: str, entity: Any) -> bool:
        """Update an existing entity."""

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete an entity."""


class BaseRepository(IRepository[T]):
    """Base repository implementation with common functionality.

    Follows DRY principle by providing common operations.
    """

    def __init__(self, connection: DynamoDBConnection, table_name: str) -> None:
        """Initialize repository with connection and table name.

        Uses Dependency Injection for loose coupling.
        """
        self._connection = connection
        self._table_name = table_name
        self._resource = None

    def _get_table(self) -> Any:
        """Get DynamoDB table resource."""
        if self._resource is None:
            self._resource = self._connection.get_resource()
        return self._resource.Table(self._table_name)

    async def create(self, entity: dict[str, Any]) -> str:
        """Create a new entity with timestamps."""

        try:
            # Add timestamps
            entity["created_at"] = datetime.now(UTC).isoformat()
            entity["updated_at"] = datetime.now(UTC).isoformat()

            table = self._get_table()

            # Run synchronous DynamoDB operation in executor
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: table.put_item(Item=entity)
            )

            logger.info("Created entity in %s: %s", self._table_name, entity.get("id"))
            return entity.get("id", "")

        except Exception:
            logger.exception("Failed to create entity in %s", self._table_name)
            raise

    async def get(self, id: str) -> dict[str, Any] | None:
        """Get entity by ID."""

        try:
            table = self._get_table()

            # Run synchronous operation in executor
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: table.get_item(Key={"id": id})
            )

            if "Item" in response:
                return response["Item"]
            return None

        except Exception:
            logger.exception("Failed to get entity %s from %s", id, self._table_name)
            raise

    async def update(self, id: str, updates: dict[str, Any]) -> bool:
        """Update an existing entity."""

        try:
            # Build update expression
            update_parts = []
            expression_values = {}

            # Always update timestamp
            updates["updated_at"] = datetime.now(UTC).isoformat()

            for key, value in updates.items():
                update_parts.append(f"{key} = :{key}")
                expression_values[f":{key}"] = value

            update_expression = "SET " + ", ".join(update_parts)

            table = self._get_table()

            # Run synchronous operation in executor
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: table.update_item(
                    Key={"id": id},
                    UpdateExpression=update_expression,
                    ExpressionAttributeValues=expression_values,
                ),
            )

            logger.info("Updated entity %s in %s", id, self._table_name)
            return True

        except Exception:
            logger.exception("Failed to update entity %s in %s", id, self._table_name)
            return False

    async def delete(self, id: str) -> bool:
        """Delete an entity."""

        try:
            table = self._get_table()

            # Run synchronous operation in executor
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: table.delete_item(Key={"id": id})
            )

            logger.info("Deleted entity %s from %s", id, self._table_name)
            return True

        except Exception:
            logger.exception("Failed to delete entity %s from %s", id, self._table_name)
            return False

    async def batch_create(self, entities: list[dict[str, Any]]) -> None:
        """Create multiple entities in batch.

        Follows DynamoDB batch write limits (25 items per batch).
        """

        try:
            table = self._get_table()

            # DynamoDB batch write limit is 25 items
            batch_size = 25

            for i in range(0, len(entities), batch_size):
                batch_items = entities[i : i + batch_size]

                # Prepare batch items with timestamps
                for entity in batch_items:
                    if "created_at" not in entity:
                        entity["created_at"] = datetime.now(UTC).isoformat()
                    if "updated_at" not in entity:
                        entity["updated_at"] = datetime.now(UTC).isoformat()

                # Run batch write in executor
                def batch_write(items: list[dict[str, Any]]) -> None:
                    with table.batch_writer() as batch:
                        for entity in items:
                            batch.put_item(Item=entity)

                await asyncio.get_event_loop().run_in_executor(None, batch_write, batch_items)

            logger.info(
                "Batch created %s entities in %s", len(entities), self._table_name
            )

        except Exception:
            logger.exception("Failed to batch create entities in %s", self._table_name)
            raise


class HealthDataRepository(BaseRepository[dict[str, Any]]):
    """Repository for health data operations.

    Specific implementation for health data with additional methods.
    """

    def __init__(self, connection: DynamoDBConnection) -> None:
        """Initialize health data repository."""
        super().__init__(connection, "clarity_health_data")

    async def get_by_user(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get health data for a specific user."""
        try:
            table = self._get_table()
            response = table.query(
                KeyConditionExpression="user_id = :user_id",
                ExpressionAttributeValues={":user_id": user_id},
                Limit=limit,
                ScanIndexForward=False,  # Most recent first
            )

            return response.get("Items", [])

        except Exception:
            logger.exception("Failed to get health data for user %s", user_id)
            return []

    async def get_by_date_range(
        self, user_id: str, start_date: datetime, end_date: datetime
    ) -> list[dict[str, Any]]:
        """Get health data within a date range."""
        try:
            table = self._get_table()
            response = table.query(
                KeyConditionExpression="user_id = :user_id",
                FilterExpression="created_at BETWEEN :start AND :end",
                ExpressionAttributeValues={
                    ":user_id": user_id,
                    ":start": start_date.isoformat(),
                    ":end": end_date.isoformat(),
                },
            )

            return response.get("Items", [])

        except Exception:
            logger.exception("Failed to get health data by date range")
            return []


class ProcessingJobRepository(BaseRepository[dict[str, Any]]):
    """Repository for processing job operations."""

    def __init__(self, connection: DynamoDBConnection) -> None:
        """Initialize processing job repository."""
        super().__init__(connection, "clarity_processing_jobs")

    async def get_by_status(self, status: str) -> list[dict[str, Any]]:
        """Get processing jobs by status."""
        try:
            table = self._get_table()
            response = table.scan(
                FilterExpression="status = :status",
                ExpressionAttributeValues={":status": status},
            )

            return response.get("Items", [])

        except Exception:
            logger.exception("Failed to get processing jobs by status %s", status)
            return []

    async def update_status(self, job_id: str, status: str, progress: int = 0) -> bool:
        """Update job status and progress."""
        updates = {"status": status, "progress": progress}
        return await self.update(job_id, updates)


class UserProfileRepository(BaseRepository[dict[str, Any]]):
    """Repository for user profile operations."""

    def __init__(self, connection: DynamoDBConnection) -> None:
        """Initialize user profile repository."""
        super().__init__(connection, "clarity_user_profiles")

    async def get_by_email(self, email: str) -> dict[str, Any] | None:
        """Get user profile by email."""
        try:
            table = self._get_table()
            response = table.scan(
                FilterExpression="email = :email",
                ExpressionAttributeValues={":email": email},
            )

            items = response.get("Items", [])
            return items[0] if items else None

        except Exception:
            logger.exception("Failed to get user by email %s", email)
            return None


class AuditLogRepository(BaseRepository[dict[str, Any]]):
    """Repository for audit log operations.

    Follows Single Responsibility: Only handles audit logs.
    """

    def __init__(self, connection: DynamoDBConnection) -> None:
        """Initialize audit log repository."""
        super().__init__(connection, "clarity_audit_logs")

    async def create_audit_log(
        self,
        operation: str,
        table: str,
        item_id: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create an audit log entry."""
        audit_entry = {
            "id": str(uuid.uuid4()),
            "operation": operation,
            "table": table,
            "item_id": item_id,
            "user_id": user_id,
            "metadata": metadata or {},
            "source": "dynamodb_repository",
        }

        return await self.create(audit_entry)

    async def get_by_item(self, item_id: str) -> list[dict[str, Any]]:
        """Get audit logs for a specific item."""
        try:
            table = self._get_table()
            response = table.scan(
                FilterExpression="item_id = :item_id",
                ExpressionAttributeValues={":item_id": item_id},
            )

            return response.get("Items", [])

        except Exception:
            logger.exception("Failed to get audit logs for item %s", item_id)
            return []


class MLModelRepository(BaseRepository[dict[str, Any]]):
    """Repository for ML model metadata operations."""

    def __init__(self, connection: DynamoDBConnection) -> None:
        """Initialize ML model repository."""
        super().__init__(connection, "clarity_ml_models")

    async def get_latest_version(self, model_type: str) -> dict[str, Any] | None:
        """Get the latest version of a model type."""
        try:
            table = self._get_table()
            response = table.query(
                KeyConditionExpression="model_type = :type",
                ExpressionAttributeValues={":model_type": model_type},
                ScanIndexForward=False,  # Latest first
                Limit=1,
            )

            items = response.get("Items", [])
            return items[0] if items else None

        except Exception:
            logger.exception("Failed to get latest model version for %s", model_type)
            return None


class RepositoryFactory:
    """Factory for creating repository instances.

    Follows Factory pattern for object creation.
    """

    def __init__(self, connection: DynamoDBConnection) -> None:
        """Initialize factory with shared connection."""
        self._connection = connection

        # Repository instances (lazy loaded)
        self._repositories: dict[str, BaseRepository[Any]] = {}

    def get_health_data_repository(self) -> HealthDataRepository:
        """Get health data repository instance."""
        if "health_data" not in self._repositories:
            self._repositories["health_data"] = HealthDataRepository(self._connection)
        return self._repositories["health_data"]

    def get_processing_job_repository(self) -> ProcessingJobRepository:
        """Get processing job repository instance."""
        if "processing_job" not in self._repositories:
            self._repositories["processing_job"] = ProcessingJobRepository(
                self._connection
            )
        return self._repositories["processing_job"]

    def get_user_profile_repository(self) -> UserProfileRepository:
        """Get user profile repository instance."""
        if "user_profile" not in self._repositories:
            self._repositories["user_profile"] = UserProfileRepository(self._connection)
        return self._repositories["user_profile"]

    def get_audit_log_repository(self) -> AuditLogRepository:
        """Get audit log repository instance."""
        if "audit_log" not in self._repositories:
            self._repositories["audit_log"] = AuditLogRepository(self._connection)
        return self._repositories["audit_log"]

    def get_ml_model_repository(self) -> MLModelRepository:
        """Get ML model repository instance."""
        if "ml_model" not in self._repositories:
            self._repositories["ml_model"] = MLModelRepository(self._connection)
        return self._repositories["ml_model"]
