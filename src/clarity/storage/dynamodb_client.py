"""AWS DynamoDB implementation for health data storage."""

# removed - breaks FastAPI

from datetime import UTC, datetime
from decimal import Decimal
import logging
from typing import TYPE_CHECKING, Any, TypeAlias

import boto3
from boto3.dynamodb.conditions import ConditionBase, Key
from botocore.exceptions import ClientError
from mypy_boto3_dynamodb import DynamoDBServiceResource
from mypy_boto3_dynamodb.service_resource import Table

from clarity.core.exceptions import ServiceError
from clarity.models.health_data import HealthMetric, ProcessingStatus
from clarity.ports.data_ports import IHealthDataRepository

if TYPE_CHECKING:
    pass  # Only for type stubs now

# Type aliases for clarity
DynamoDBItem: TypeAlias = dict[str, Any]
SerializedItem: TypeAlias = dict[str, Any]

logger = logging.getLogger(__name__)


class DynamoDBHealthDataRepository(IHealthDataRepository):
    """DynamoDB implementation of health data repository."""

    def __init__(
        self,
        table_name: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ) -> None:
        self.table_name = table_name
        self.region = region

        # Create DynamoDB resource with proper typing
        if endpoint_url:  # For local testing with DynamoDB Local
            self.dynamodb: DynamoDBServiceResource = boto3.resource(
                "dynamodb", region_name=region, endpoint_url=endpoint_url
            )
        else:
            self.dynamodb = boto3.resource("dynamodb", region_name=region)

        self.table: Table = self.dynamodb.Table(table_name)

    @staticmethod
    def _serialize_item(data: DynamoDBItem) -> SerializedItem:
        """Convert Python types to DynamoDB-compatible types."""

        def convert_value(v: Any) -> Any:
            if isinstance(v, float):
                return Decimal(str(v))
            if isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            if isinstance(v, list):
                return [convert_value(item) for item in v]
            if isinstance(v, datetime):
                return v.isoformat()
            return v

        return {k: convert_value(v) for k, v in data.items()}

    @staticmethod
    def _deserialize_item(item: SerializedItem) -> DynamoDBItem:
        """Convert DynamoDB types back to Python types."""

        def convert_value(v: Any) -> Any:
            if isinstance(v, Decimal):
                return float(v)
            if isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            if isinstance(v, list):
                return [convert_value(item) for item in v]
            return v

        return {k: convert_value(v) for k, v in item.items()}

    async def save_health_data(
        self,
        user_id: str,
        processing_id: str,
        metrics: list[HealthMetric],
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
            # Generate unique ID (timestamp-based)
            timestamp = datetime.now(UTC)
            item_id = f"{user_id}#{timestamp.isoformat()}"

            # Prepare item for DynamoDB
            item: DynamoDBItem = {
                "pk": f"USER#{user_id}",  # Partition key
                "sk": f"HEALTH#{timestamp.isoformat()}",  # Sort key
                "id": item_id,
                "user_id": user_id,
                "processing_id": processing_id,
                "timestamp": timestamp.isoformat(),
                "client_timestamp": client_timestamp.isoformat(),
                "upload_source": upload_source,
                "metrics": {
                    metric.metric_type.value: {
                        "metric_id": str(metric.metric_id),
                        "biometric_data": (
                            metric.biometric_data.model_dump()
                            if metric.biometric_data
                            else None
                        ),
                        "sleep_data": (
                            metric.sleep_data.model_dump()
                            if metric.sleep_data
                            else None
                        ),
                        "activity_data": (
                            metric.activity_data.model_dump()
                            if metric.activity_data
                            else None
                        ),
                        "mental_health_data": (
                            metric.mental_health_data.model_dump()
                            if metric.mental_health_data
                            else None
                        ),
                        "device_id": metric.device_id,
                        "raw_data": metric.raw_data or {},
                        "metadata": metric.metadata or {},
                        "created_at": metric.created_at.isoformat(),
                    }
                    for metric in metrics
                },
                "processing_status": ProcessingStatus.RECEIVED.value,
                "created_at": timestamp.isoformat(),
                "ttl": int(
                    (timestamp.timestamp()) + (90 * 24 * 60 * 60)
                ),  # 90 days TTL
            }

            # Serialize for DynamoDB
            serialized_item = self._serialize_item(item)

            # Save to DynamoDB
            self.table.put_item(Item=serialized_item)

        except ClientError as e:
            logger.exception("DynamoDB error saving health data")
            msg = f"Failed to save health data: {e!s}"
            raise ServiceError(msg) from e
        except Exception as e:
            logger.exception("Unexpected error saving health data")
            msg = f"Failed to save health data: {e!s}"
            raise ServiceError(msg) from e
        else:
            return True

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
            # Build query
            key_condition: ConditionBase = Key("pk").eq(f"USER#{user_id}")

            if start_date and end_date:
                key_condition &= Key("sk").between(
                    f"HEALTH#{start_date.isoformat()}", f"HEALTH#{end_date.isoformat()}"
                )
            elif start_date:
                key_condition &= Key("sk").gte(f"HEALTH#{start_date.isoformat()}")
            elif end_date:
                key_condition &= Key("sk").lte(f"HEALTH#{end_date.isoformat()}")
            else:
                key_condition &= Key("sk").begins_with("HEALTH#")

            # Query DynamoDB
            response = self.table.query(
                KeyConditionExpression=key_condition,
                Limit=limit + offset,  # Fetch extra for offset
                ScanIndexForward=False,  # Most recent first
            )

            items = response.get("Items", [])

            # Apply offset
            if offset > 0:
                items = items[offset:]

            # Limit results
            items = items[:limit]

            # Convert to response objects
            results = []
            for item in items:
                deserialized = self._deserialize_item(item)

                # Filter by metric type if specified
                if metric_type:
                    metrics = deserialized.get("metrics", {})
                    if metric_type not in metrics:
                        continue

                results.append(deserialized)

            return {
                "data": results,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": len(results),
                    "has_more": len(response.get("Items", [])) > limit + offset,
                },
            }

        except ClientError as e:
            logger.exception("DynamoDB error retrieving health data")
            msg = f"Failed to retrieve health data: {e!s}"
            raise ServiceError(msg) from e
        except Exception as e:
            logger.exception("Unexpected error retrieving health data")
            msg = f"Failed to retrieve health data: {e!s}"
            raise ServiceError(msg) from e

    async def get_processing_status(
        self, processing_id: str, user_id: str
    ) -> dict[str, str] | None:
        """Get processing status for a health data upload.

        Args:
            processing_id: Processing job identifier
            user_id: User identifier for ownership verification

        Returns:
            Processing status info or None if not found
        """
        try:
            # Query by processing_id
            response = self.table.query(
                IndexName="processing-id-index",  # Assumes GSI exists
                KeyConditionExpression=Key("processing_id").eq(processing_id),
            )

            items = response.get("Items", [])
            if not items:
                return None

            # Verify user owns this data
            item = items[0]
            if item.get("user_id") != user_id:
                return None

            return {
                "processing_id": processing_id,
                "status": str(item.get("processing_status", "unknown")),
                "created_at": str(item.get("created_at", "")),
                "updated_at": str(item.get("updated_at", "")),
            }

        except ClientError:
            logger.exception("DynamoDB error getting processing status")
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
                # Delete specific processing job
                # First, find the item by processing_id
                response = self.table.query(
                    IndexName="processing-id-index",
                    KeyConditionExpression=Key("processing_id").eq(processing_id),
                )
                items = response.get("Items", [])
                if items and items[0].get("user_id") == user_id:
                    self.table.delete_item(
                        Key={"pk": items[0]["pk"], "sk": items[0]["sk"]}
                    )
            else:
                # Delete all user data
                response = self.table.query(
                    KeyConditionExpression=Key("pk").eq(f"USER#{user_id}")
                    & Key("sk").begins_with("HEALTH#")
                )

                # Delete each item
                with self.table.batch_writer() as batch:
                    for item in response.get("Items", []):
                        batch.delete_item(Key={"pk": item["pk"], "sk": item["sk"]})

        except ClientError as e:
            logger.exception("DynamoDB error deleting health data")
            msg = f"Failed to delete health data: {e!s}"
            raise ServiceError(msg) from e
        else:
            return True

    async def save_data(self, user_id: str, data: dict[str, str]) -> str:
        """Save health data for a user (legacy method).

        Args:
            user_id: User identifier
            data: Health data to save

        Returns:
            Record identifier
        """
        try:
            timestamp = datetime.now(UTC)
            item_id = f"{user_id}#{timestamp.isoformat()}"

            item: DynamoDBItem = {
                "pk": f"USER#{user_id}",
                "sk": f"DATA#{timestamp.isoformat()}",
                "id": item_id,
                "user_id": user_id,
                "data": data,
                "created_at": timestamp.isoformat(),
            }

            self.table.put_item(Item=self._serialize_item(item))

        except ClientError as e:
            logger.exception("DynamoDB error saving data")
            msg = f"Failed to save data: {e!s}"
            raise ServiceError(msg) from e
        else:
            return item_id

    async def get_data(
        self, user_id: str, _filters: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Retrieve health data for a user (legacy method).

        Args:
            user_id: User identifier
            filters: Optional filters to apply

        Returns:
            Health data dictionary
        """
        try:
            response = self.table.query(
                KeyConditionExpression=Key("pk").eq(f"USER#{user_id}")
                & Key("sk").begins_with("DATA#"),
                Limit=1,
                ScanIndexForward=False,
            )

            items = response.get("Items", [])
            if items:
                item = self._deserialize_item(items[0])
                data = item.get("data", {})
                # Ensure we return dict[str, str]
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}

        except ClientError as e:
            logger.exception("DynamoDB error getting data")
            msg = f"Failed to get data: {e!s}"
            raise ServiceError(msg) from e
        else:
            return {}

    @staticmethod
    async def initialize() -> None:
        """Initialize the repository.

        Performs any necessary setup operations like connecting to database,
        creating indexes, etc.
        """
        # DynamoDB doesn't need explicit initialization
        # Table should already exist
        logger.info("DynamoDB repository initialized")

    @staticmethod
    async def cleanup() -> None:
        """Clean up repository resources.

        Performs cleanup operations like closing connections, releasing resources, etc.
        """
        # DynamoDB client doesn't need explicit cleanup
        logger.info("DynamoDB repository cleaned up")
