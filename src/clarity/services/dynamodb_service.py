"""CLARITY Digital Twin Platform - DynamoDB Service.

Enterprise-grade AWS DynamoDB client for health data operations.
Provides AWS-native NoSQL solution.
"""

# removed - breaks FastAPI

import asyncio
from datetime import UTC, datetime, timedelta
import json
import logging
import time
from typing import TYPE_CHECKING, Any
import uuid
from uuid import UUID

import boto3
from botocore.exceptions import ClientError
from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource

if TYPE_CHECKING:
    pass  # Only for type stubs now

from clarity.ports.data_ports import IHealthDataRepository

# Configure logger
logger = logging.getLogger(__name__)


class DynamoDBError(Exception):
    """Base exception for DynamoDB operations."""


class DocumentNotFoundError(DynamoDBError):
    """Raised when a requested document is not found."""


class DynamoDBPermissionError(DynamoDBError):
    """Raised when operation is not permitted."""


class DynamoDBValidationError(DynamoDBError):
    """Raised when data validation fails."""


class DynamoDBConnectionError(DynamoDBError):
    """Raised when connection to DynamoDB fails."""


class DynamoDBService:
    """Enterprise-grade DynamoDB service for health data operations.

    Provides comprehensive DynamoDB functionality for health data management.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        table_prefix: str = "clarity_",
        *,
        enable_caching: bool = True,
        cache_ttl: int = 300,  # 5 minutes
    ) -> None:
        """Initialize the DynamoDB service.

        Args:
            region: AWS region
            endpoint_url: Optional endpoint URL (for local DynamoDB)
            table_prefix: Prefix for all table names
            enable_caching: Enable in-memory caching for read operations
            cache_ttl: Cache time-to-live in seconds
        """
        self.region = region
        self.endpoint_url = endpoint_url
        self.table_prefix = table_prefix
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl

        # Initialize DynamoDB client
        self.dynamodb: DynamoDBServiceResource = boto3.resource(
            "dynamodb",
            region_name=region,
            endpoint_url=endpoint_url,
        )

        # Connection and caching
        self._cache: dict[str, dict[str, Any]] = {}
        self._connection_lock = asyncio.Lock()

        # Table names
        self.tables = {
            "health_data": f"{table_prefix}health_data",
            "processing_jobs": f"{table_prefix}processing_jobs",
            "user_profiles": f"{table_prefix}user_profiles",
            "audit_logs": f"{table_prefix}audit_logs",
            "ml_models": f"{table_prefix}ml_models",
            "insights": f"{table_prefix}insights",
            "analysis_results": f"{table_prefix}analysis_results",
        }

        logger.info("DynamoDB service initialized for region: %s", region)

    @staticmethod
    def _cache_key(table: str, item_id: str) -> str:
        """Generate cache key for item."""
        return f"{table}:{item_id}"

    def _is_cache_valid(self, cache_entry: dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if not self.enable_caching:
            return False

        timestamp: float = cache_entry.get("timestamp", 0.0)
        return time.time() - timestamp < self.cache_ttl

    @staticmethod
    async def _validate_health_data(data: dict[str, Any]) -> None:
        """Validate health data before storage."""
        required_fields = ["user_id", "metrics", "upload_source"]

        for field in required_fields:
            if field not in data:
                msg = f"Missing required field: {field}"
                raise DynamoDBValidationError(msg)

        # Validate user_id format
        try:
            if isinstance(data["user_id"], str):
                UUID(data["user_id"])
        except ValueError:
            msg = "Invalid user_id format"
            raise DynamoDBValidationError(msg) from None

        # Validate metrics
        if not isinstance(data["metrics"], list) or not data["metrics"]:
            msg = "Metrics must be a non-empty list"
            raise DynamoDBValidationError(msg) from None

    async def _audit_log(
        self,
        operation: str,
        table: str,
        item_id: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create audit log entry for HIPAA compliance."""
        try:
            audit_table = self.dynamodb.Table(self.tables["audit_logs"])
            audit_entry = {
                "audit_id": str(uuid.uuid4()),
                "operation": operation,
                "table": table,
                "item_id": item_id,
                "user_id": user_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "metadata": metadata or {},
                "source": "dynamodb_service",
            }

            await asyncio.get_event_loop().run_in_executor(
                None, lambda: audit_table.put_item(Item=audit_entry)
            )

            logger.debug("Audit log created: %s on %s/%s", operation, table, item_id)

        except Exception:
            logger.exception("Failed to create audit log")
            # Don't raise exception for audit failures to avoid breaking main operations

    # Item Operations

    async def put_item(
        self,
        table_name: str,
        item: dict[str, Any],
        user_id: str | None = None,
    ) -> str:
        """Put an item in the specified table.

        Args:
            table_name: Table name
            item: Item data
            user_id: User ID for audit logging

        Returns:
            str: Item ID

        Raises:
            DynamoDBValidationError: If data validation fails
            DynamoDBConnectionError: If DynamoDB connection fails
        """
        try:
            # Add timestamps
            item["created_at"] = datetime.now(UTC).isoformat()
            item["updated_at"] = datetime.now(UTC).isoformat()

            # Ensure item has an ID
            if "id" not in item and "user_id" in item:
                item["id"] = item["user_id"]
            elif "id" not in item:
                item["id"] = str(uuid.uuid4())

            # Validate health data if applicable
            if table_name == self.tables["health_data"]:
                await self._validate_health_data(item)

            table = self.dynamodb.Table(table_name)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: table.put_item(Item=item)
            )

            item_id: str = str(item["id"])

            # Cache the item
            if self.enable_caching:
                cache_key = self._cache_key(table_name, item_id)
                self._cache[cache_key] = {
                    "data": item,
                    "timestamp": time.time(),
                }

            # Audit log
            await self._audit_log(
                "CREATE",
                table_name,
                item_id,
                user_id,
                {"item_size": len(json.dumps(item))},
            )

            logger.info("Item created: %s/%s", table_name, item_id)
            return item_id

        except DynamoDBValidationError:
            raise
        except ClientError as e:
            logger.exception("Failed to create item in %s", table_name)
            msg = f"Item creation failed: {e}"
            raise DynamoDBError(msg) from e

    async def get_item(
        self,
        table_name: str,
        key: dict[str, Any],
        *,
        use_cache: bool = True,
    ) -> dict[str, Any] | None:
        """Retrieve an item by key.

        Args:
            table_name: Table name
            key: Primary key of the item
            use_cache: Whether to use cached data if available

        Returns:
            Dict containing item data or None if not found
        """
        try:
            # Get the ID from the key for caching
            item_id = key.get("id") or key.get("user_id") or str(key)

            # Check cache first
            cache_key = self._cache_key(table_name, item_id)
            if use_cache and cache_key in self._cache:
                cache_entry = self._cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    logger.debug("Cache hit for %s/%s", table_name, item_id)
                    return dict(cache_entry["data"])

            table = self.dynamodb.Table(table_name)
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: table.get_item(Key=key)
            )

            if "Item" not in response:
                logger.warning("Item not found: %s/%s", table_name, item_id)
                return None

            item = response["Item"]

            # Cache the result
            if self.enable_caching:
                self._cache[cache_key] = {"data": item, "timestamp": time.time()}

            logger.debug("Item retrieved: %s/%s", table_name, item_id)
            return item

        except Exception as e:
            logger.exception("Failed to get item %s/%s", table_name, key)
            msg = f"Item retrieval failed: {e}"
            raise DynamoDBError(msg) from e

    async def update_item(
        self,
        table_name: str,
        key: dict[str, Any],
        update_expression: str,
        expression_attribute_values: dict[str, Any],
        user_id: str | None = None,
    ) -> bool:
        """Update an existing item.

        Args:
            table_name: Table name
            key: Primary key of the item
            update_expression: Update expression
            expression_attribute_values: Values for the update expression
            user_id: User ID for audit logging

        Returns:
            bool: True if update was successful
        """
        try:
            # Add update timestamp
            update_expression += ", updated_at = :updated_at"
            expression_attribute_values[":updated_at"] = datetime.now(UTC).isoformat()

            table = self.dynamodb.Table(table_name)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: table.update_item(
                    Key=key,
                    UpdateExpression=update_expression,
                    ExpressionAttributeValues=expression_attribute_values,
                ),
            )

            # Clear cache
            item_id = key.get("id") or key.get("user_id") or str(key)
            cache_key = self._cache_key(table_name, item_id)
            self._cache.pop(cache_key, None)

            # Audit log
            await self._audit_log(
                "UPDATE",
                table_name,
                item_id,
                user_id,
                {"update_expression": update_expression},
            )

            logger.info("Item updated: %s/%s", table_name, item_id)
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.warning("Item not found for update: %s/%s", table_name, key)
                return False
            logger.exception("Failed to update item %s/%s", table_name, key)
            msg = f"Item update failed: {e}"
            raise DynamoDBError(msg) from e

    async def delete_item(
        self, table_name: str, key: dict[str, Any], user_id: str | None = None
    ) -> bool:
        """Delete an item.

        Args:
            table_name: Table name
            key: Primary key of the item
            user_id: User ID for audit logging

        Returns:
            bool: True if deletion was successful
        """
        try:
            table = self.dynamodb.Table(table_name)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: table.delete_item(Key=key)
            )

            # Clear cache
            item_id = key.get("id") or key.get("user_id") or str(key)
            cache_key = self._cache_key(table_name, item_id)
            self._cache.pop(cache_key, None)

            # Audit log
            await self._audit_log("DELETE", table_name, item_id, user_id)

            logger.info("Item deleted: %s/%s", table_name, item_id)
            return True

        except Exception as e:
            logger.exception("Failed to delete item %s/%s", table_name, key)
            msg = f"Item deletion failed: {e}"
            raise DynamoDBError(msg) from e

    async def query(
        self,
        table_name: str,
        key_condition_expression: str,
        expression_attribute_values: dict[str, Any],
        limit: int | None = None,
        *,
        scan_index_forward: bool = True,
    ) -> dict[str, Any]:
        """Query items from a table.

        Args:
            table_name: Table name
            key_condition_expression: Key condition expression
            expression_attribute_values: Values for the expression
            limit: Maximum number of items to return
            scan_index_forward: Sort order (True for ascending)

        Returns:
            Dict with Items and pagination info
        """
        try:
            table = self.dynamodb.Table(table_name)

            query_params: dict[str, Any] = {
                "KeyConditionExpression": key_condition_expression,
                "ExpressionAttributeValues": expression_attribute_values,
                "ScanIndexForward": scan_index_forward,
            }

            if limit:
                query_params["Limit"] = limit

            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: table.query(**query_params)
            )

            return {
                "Items": response.get("Items", []),
                "Count": response.get("Count", 0),
                "LastEvaluatedKey": response.get("LastEvaluatedKey"),
            }

        except Exception as e:
            logger.exception("Failed to query items in %s", table_name)
            msg = f"Query operation failed: {e}"
            raise DynamoDBError(msg) from e

    async def batch_write_items(
        self, table_name: str, items: list[dict[str, Any]]
    ) -> None:
        """Write multiple items in batch.

        Args:
            table_name: Table name
            items: List of items to write

        Raises:
            DynamoDBError: If batch write fails
        """
        try:
            table = self.dynamodb.Table(table_name)

            # DynamoDB batch write limit is 25 items
            batch_size = 25

            for i in range(0, len(items), batch_size):
                batch_items = items[i : i + batch_size]

                with table.batch_writer() as batch:
                    for item in batch_items:
                        # Add timestamps
                        item["created_at"] = datetime.now(UTC).isoformat()
                        item["updated_at"] = datetime.now(UTC).isoformat()

                        # Ensure ID
                        if "id" not in item:
                            item["id"] = str(uuid.uuid4())

                        batch.put_item(Item=item)

            await self._audit_log(
                operation="batch_write_items",
                table=table_name,
                item_id="batch_write",
                metadata={"item_count": len(items)},
            )

        except Exception as e:
            logger.exception("Failed to batch write items in %s", table_name)
            msg = f"Batch write operation failed: {e}"
            raise DynamoDBError(msg) from e

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on the DynamoDB connection.

        Returns:
            Dict with health status information
        """
        try:
            # Test connection by describing a table
            table = self.dynamodb.Table(self.tables["health_data"])
            await asyncio.get_event_loop().run_in_executor(None, table.load)

            return {
                "status": "healthy",
                "region": self.region,
                "cache_enabled": self.enable_caching,
                "cached_items": len(self._cache),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.exception("DynamoDB health check failed")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }


class DynamoDBHealthDataRepository(IHealthDataRepository):
    """Health Data Repository implementation using DynamoDB.

    Provides AWS DynamoDB-based health data repository.
    """

    def __init__(
        self, region: str = "us-east-1", endpoint_url: str | None = None
    ) -> None:
        """Initialize DynamoDB health data repository.

        Args:
            region: AWS region
            endpoint_url: Optional endpoint URL (for local DynamoDB)
        """
        self._dynamodb_service = DynamoDBService(
            region=region, endpoint_url=endpoint_url
        )

    @property
    def service(self) -> DynamoDBService:
        """Get the underlying DynamoDB service.

        Returns:
            The DynamoDB service instance
        """
        return self._dynamodb_service

    async def save_health_data(
        self,
        user_id: str,
        processing_id: str,
        metrics: list[Any],
        upload_source: str,
        client_timestamp: datetime,
    ) -> bool:
        """Save health data with processing metadata.

        Args:
            user_id: User identifier
            processing_id: Processing job identifier
            metrics: List of health metrics
            upload_source: Source of the upload
            client_timestamp: Client-side timestamp

        Returns:
            True if saved successfully
        """
        try:
            # Create processing document
            processing_doc = {
                "processing_id": processing_id,
                "user_id": user_id,
                "upload_source": upload_source,
                "client_timestamp": client_timestamp.isoformat(),
                "created_at": datetime.now(UTC).isoformat(),
                "status": "processing",
                "total_metrics": len(metrics),
                "processed_metrics": 0,
                "expires_at": (datetime.now(UTC) + timedelta(days=30)).isoformat(),
            }

            # Store processing document
            await self._dynamodb_service.put_item(
                table_name=self._dynamodb_service.tables["processing_jobs"],
                item=processing_doc,
                user_id=user_id,
            )

            # Store metrics in batch
            metric_items = []
            for i, metric in enumerate(metrics):
                # Handle both Pydantic models and dicts
                if hasattr(metric, "model_dump"):
                    metric_data = metric.model_dump()
                else:
                    metric_data = dict(metric)

                metric_doc = {
                    "id": f"{processing_id}#{i}",  # Composite key
                    "user_id": user_id,
                    "processing_id": processing_id,
                    "metric_index": i,
                    "metric_data": metric_data,
                    "created_at": datetime.now(UTC).isoformat(),
                    "expires_at": (datetime.now(UTC) + timedelta(days=30)).isoformat(),
                }
                metric_items.append(metric_doc)

            # Batch write metrics
            await self._dynamodb_service.batch_write_items(
                table_name=self._dynamodb_service.tables["health_data"],
                items=metric_items,
            )

            logger.info(
                "Health data saved: %s with %s metrics", processing_id, len(metrics)
            )
            return True

        except Exception:
            logger.exception(
                "Failed to save health data for processing %s", processing_id
            )
            return False

    async def get_user_health_data(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        metric_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Retrieve user health data with filtering and pagination.

        Args:
            user_id: User identifier
            limit: Maximum records to return
            offset: Records to skip
            metric_type: Filter by metric type
            start_date: Filter from date
            end_date: Filter to date

        Returns:
            Health data with pagination metadata
        """
        try:
            # Query health metrics for user
            response = await self._dynamodb_service.query(
                table_name=self._dynamodb_service.tables["health_data"],
                key_condition_expression="user_id = :user_id",
                expression_attribute_values={":user_id": user_id},
                limit=limit,
                scan_index_forward=False,  # Most recent first
            )

            metrics = response.get("Items", [])

            # Apply additional filters in memory (DynamoDB doesn't support complex queries)
            if metric_type:
                metrics = [
                    m
                    for m in metrics
                    if m.get("metric_data", {}).get("metric_type") == metric_type
                ]

            if start_date:
                metrics = [
                    m
                    for m in metrics
                    if datetime.fromisoformat(m.get("created_at", "")) >= start_date
                ]

            if end_date:
                metrics = [
                    m
                    for m in metrics
                    if datetime.fromisoformat(m.get("created_at", "")) <= end_date
                ]

            # Apply offset
            if offset > 0:
                metrics = metrics[offset:]

            return {
                "metrics": metrics[:limit],
                "pagination": {
                    "total": len(metrics),
                    "limit": limit,
                    "offset": offset,
                    "has_more": len(metrics) > limit,
                },
                "filters": {
                    "metric_type": metric_type,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                },
            }

        except Exception as e:
            logger.exception("Failed to get health data for user %s", user_id)
            msg = f"Health data retrieval failed: {e}"
            raise DynamoDBError(msg) from e

    async def get_processing_status(
        self, processing_id: str, user_id: str
    ) -> dict[str, Any] | None:
        """Get processing status for a health data upload.

        Args:
            processing_id: Processing job identifier
            user_id: User identifier for ownership verification

        Returns:
            Processing status info or None if not found
        """
        try:
            # Get processing document
            doc = await self._dynamodb_service.get_item(
                table_name=self._dynamodb_service.tables["processing_jobs"],
                key={"processing_id": processing_id},
            )

            if not doc:
                return None

            # Verify user ownership
            if doc.get("user_id") != user_id:
                return None

            # Calculate progress
            total_metrics = doc.get("total_metrics", 0)
            processed_metrics = doc.get("processed_metrics", 0)
            progress = (
                (processed_metrics / total_metrics * 100) if total_metrics > 0 else 0
            )

            return {
                "processing_id": processing_id,
                "status": doc.get("status"),
                "progress": progress,
                "total_metrics": total_metrics,
                "processed_metrics": processed_metrics,
                "created_at": doc.get("created_at"),
                "upload_source": doc.get("upload_source"),
            }

        except Exception:
            logger.exception("Failed to get processing status for %s", processing_id)
            return None

    async def delete_health_data(
        self, user_id: str, processing_id: str | None = None
    ) -> bool:
        """Delete user health data.

        Args:
            user_id: User identifier
            processing_id: Optional specific processing job to delete

        Returns:
            True if deletion was successful
        """
        try:
            if processing_id:
                # Delete specific processing job and related metrics
                response = await self._dynamodb_service.query(
                    table_name=self._dynamodb_service.tables["health_data"],
                    key_condition_expression="user_id = :user_id AND begins_with(id, :processing_id)",
                    expression_attribute_values={
                        ":user_id": user_id,
                        ":processing_id": processing_id,
                    },
                )

                # Delete each metric
                for item in response.get("Items", []):
                    await self._dynamodb_service.delete_item(
                        table_name=self._dynamodb_service.tables["health_data"],
                        key={"id": item["id"], "user_id": user_id},
                        user_id=user_id,
                    )

                # Delete processing record
                await self._dynamodb_service.delete_item(
                    table_name=self._dynamodb_service.tables["processing_jobs"],
                    key={"processing_id": processing_id},
                    user_id=user_id,
                )
            else:
                # Delete all user data
                response = await self._dynamodb_service.query(
                    table_name=self._dynamodb_service.tables["health_data"],
                    key_condition_expression="user_id = :user_id",
                    expression_attribute_values={":user_id": user_id},
                )

                # Delete each item
                for item in response.get("Items", []):
                    await self._dynamodb_service.delete_item(
                        table_name=self._dynamodb_service.tables["health_data"],
                        key={"id": item["id"], "user_id": user_id},
                        user_id=user_id,
                    )

            # Create audit log
            audit_record = {
                "user_id": user_id,
                "action": "data_deletion",
                "processing_id": processing_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "reason": "user_request",
            }

            await self._dynamodb_service.put_item(
                table_name=self._dynamodb_service.tables["audit_logs"],
                item=audit_record,
            )

            logger.info(
                "Deleted health data for user %s, processing %s", user_id, processing_id
            )
            return True

        except Exception:
            logger.exception("Failed to delete health data for user %s", user_id)
            return False

    async def save_data(self, user_id: str, data: dict[str, Any]) -> str:
        """Save health data for a user (legacy method).

        Args:
            user_id: User identifier
            data: Health data to save

        Returns:
            Document ID of saved data
        """
        try:
            # Add metadata
            enriched_data = {
                **data,
                "user_id": user_id,
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
            }

            # Create item in health_data table
            item_id = await self._dynamodb_service.put_item(
                table_name=self._dynamodb_service.tables["health_data"],
                item=enriched_data,
                user_id=user_id,
            )

            logger.info("Health data saved for user %s: %s", user_id, item_id)
            return item_id

        except Exception as e:
            logger.exception("Failed to save health data for user %s", user_id)
            msg = f"Health data save failed: {e}"
            raise DynamoDBError(msg) from e

    async def get_data(
        self, user_id: str, filters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Retrieve health data for a user (legacy method).

        Args:
            user_id: User identifier
            filters: Optional filters to apply

        Returns:
            Dictionary containing health data
        """
        try:
            # Query documents
            response = await self._dynamodb_service.query(
                table_name=self._dynamodb_service.tables["health_data"],
                key_condition_expression="user_id = :user_id",
                expression_attribute_values={":user_id": user_id},
                scan_index_forward=False,
            )

            documents = response.get("Items", [])

            # Apply filters in memory
            if filters:
                for key, value in filters.items():
                    if key != "user_id":
                        documents = [doc for doc in documents if doc.get(key) == value]

            result = {
                "user_id": user_id,
                "total_records": len(documents),
                "data": documents,
                "retrieved_at": datetime.now(UTC).isoformat(),
            }

            logger.info(
                "Retrieved %s health records for user %s", len(documents), user_id
            )
            return result

        except Exception as e:
            logger.exception("Failed to get health data for user %s", user_id)
            msg = f"Health data retrieval failed: {e}"
            raise DynamoDBError(msg) from e

    async def initialize(self) -> None:
        """Initialize the repository."""
        # Test connection
        health_status = await self._dynamodb_service.health_check()
        if health_status["status"] != "healthy":
            msg = "DynamoDB connection unhealthy"
            raise ConnectionError(msg)

        try:
            logger.info("DynamoDBHealthDataRepository initialized successfully")

        except Exception as e:
            logger.exception("Failed to initialize DynamoDBHealthDataRepository")
            msg = f"Repository initialization failed: {e}"
            raise ConnectionError(msg) from e

    async def cleanup(self) -> None:
        """Clean up repository resources."""
        try:
            # Clear cache
            self._dynamodb_service._cache.clear()  # noqa: SLF001
            logger.info("DynamoDBHealthDataRepository cleaned up successfully")

        except Exception:
            logger.exception("Failed to cleanup DynamoDBHealthDataRepository")

    async def get_user_health_summary(self, user_id: str) -> dict[str, Any]:
        """Get health data summary for a user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary containing health data summary
        """
        try:
            # Get recent data
            response = await self._dynamodb_service.query(
                table_name=self._dynamodb_service.tables["health_data"],
                key_condition_expression="user_id = :user_id",
                expression_attribute_values={":user_id": user_id},
                limit=100,
                scan_index_forward=False,
            )

            recent_data = response.get("Items", [])
            total_count = response.get("Count", 0)

            return {
                "user_id": user_id,
                "total_records": total_count,
                "recent_records": len(recent_data),
                "latest_record": recent_data[0] if recent_data else None,
                "summary_generated_at": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.exception("Failed to get health summary for user %s", user_id)
            msg = f"Health summary retrieval failed: {e}"
            raise DynamoDBError(msg) from e

    async def delete_user_data(self, user_id: str) -> int:
        """Delete all health data for a user (GDPR compliance).

        Args:
            user_id: User identifier

        Returns:
            Number of records deleted
        """
        try:
            # Query all user's records
            response = await self._dynamodb_service.query(
                table_name=self._dynamodb_service.tables["health_data"],
                key_condition_expression="user_id = :user_id",
                expression_attribute_values={":user_id": user_id},
            )

            items = response.get("Items", [])
            deleted_count = 0

            # Delete each item
            for item in items:
                await self._dynamodb_service.delete_item(
                    table_name=self._dynamodb_service.tables["health_data"],
                    key={"id": item["id"], "user_id": user_id},
                    user_id=user_id,
                )
                deleted_count += 1

            logger.info("Deleted %s health records for user %s", deleted_count, user_id)
            return deleted_count

        except Exception as e:
            logger.exception("Failed to delete health data for user %s", user_id)
            msg = f"Health data deletion failed: {e}"
            raise DynamoDBError(msg) from e
