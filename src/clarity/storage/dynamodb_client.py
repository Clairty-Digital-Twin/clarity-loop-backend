"""AWS DynamoDB implementation for health data storage."""

from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal
import json
import logging
from typing import Any, Dict, List, Optional

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

from clarity.core.exceptions import ServiceError
from clarity.models.health_data import (
    HealthDataResponse,
    HealthDataUpload,
    HealthMetric,
    ProcessingStatus,
)
from clarity.models.user import User
from clarity.ports.data_ports import IHealthDataRepository

logger = logging.getLogger(__name__)


class DynamoDBHealthDataRepository(IHealthDataRepository):
    """DynamoDB implementation of health data repository."""

    def __init__(
        self,
        table_name: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ):
        self.table_name = table_name
        self.region = region

        # Create DynamoDB resource
        if endpoint_url:  # For local testing with DynamoDB Local
            self.dynamodb = boto3.resource(
                "dynamodb", region_name=region, endpoint_url=endpoint_url
            )
        else:
            self.dynamodb = boto3.resource("dynamodb", region_name=region)

        self.table = self.dynamodb.Table(table_name)

    def _serialize_item(self, data: dict) -> dict:
        """Convert Python types to DynamoDB-compatible types."""

        def convert_value(v):
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

    def _deserialize_item(self, item: dict) -> dict:
        """Convert DynamoDB types back to Python types."""

        def convert_value(v):
            if isinstance(v, Decimal):
                return float(v)
            if isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            if isinstance(v, list):
                return [convert_value(item) for item in v]
            return v

        return {k: convert_value(v) for k, v in item.items()}

    async def save_health_data(
        self, user_id: str, data: HealthDataUpload
    ) -> HealthDataResponse:
        """Save health data to DynamoDB."""
        try:
            # Generate unique ID (timestamp-based)
            timestamp = datetime.utcnow()
            item_id = f"{user_id}#{timestamp.isoformat()}"

            # Prepare item for DynamoDB
            item = {
                "pk": f"USER#{user_id}",  # Partition key
                "sk": f"HEALTH#{timestamp.isoformat()}",  # Sort key
                "id": item_id,
                "user_id": user_id,
                "timestamp": timestamp.isoformat(),
                "metrics": {
                    metric.metric_type.value: {
                        "value": metric.value,
                        "unit": metric.unit,
                        "metadata": metric.metadata or {},
                    }
                    for metric in data.metrics
                },
                "raw_data": data.raw_data or {},
                "processing_status": ProcessingStatus.PENDING.value,
                "created_at": timestamp.isoformat(),
                "ttl": int(
                    (timestamp.timestamp()) + (90 * 24 * 60 * 60)
                ),  # 90 days TTL
            }

            # Serialize for DynamoDB
            serialized_item = self._serialize_item(item)

            # Save to DynamoDB
            self.table.put_item(Item=serialized_item)

            return HealthDataResponse(
                id=item_id,
                user_id=user_id,
                timestamp=timestamp,
                metrics=[
                    HealthMetric(
                        metric_type=metric.metric_type,
                        value=metric.value,
                        unit=metric.unit,
                        timestamp=timestamp,
                        metadata=metric.metadata,
                    )
                    for metric in data.metrics
                ],
                processing_status=ProcessingStatus.PENDING,
                raw_data=data.raw_data,
            )

        except ClientError as e:
            logger.error(f"DynamoDB error saving health data: {e}")
            raise DatabaseError(f"Failed to save health data: {e!s}")
        except Exception as e:
            logger.error(f"Unexpected error saving health data: {e}")
            raise DatabaseError(f"Failed to save health data: {e!s}")

    async def get_health_data(
        self,
        user_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        metric_types: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[HealthDataResponse]:
        """Retrieve health data from DynamoDB."""
        try:
            # Build query
            key_condition = Key("pk").eq(f"USER#{user_id}")

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

                # Convert metrics
                metrics = []
                for metric_type, metric_data in deserialized.get("metrics", {}).items():
                    # Filter by metric types if specified
                    if metric_types and metric_type not in metric_types:
                        continue

                    metrics.append(
                        HealthMetric(
                            metric_type=metric_type,
                            value=metric_data["value"],
                            unit=metric_data["unit"],
                            timestamp=datetime.fromisoformat(deserialized["timestamp"]),
                            metadata=metric_data.get("metadata", {}),
                        )
                    )

                if metrics:  # Only include if has matching metrics
                    results.append(
                        HealthDataResponse(
                            id=deserialized["id"],
                            user_id=deserialized["user_id"],
                            timestamp=datetime.fromisoformat(deserialized["timestamp"]),
                            metrics=metrics,
                            processing_status=ProcessingStatus(
                                deserialized.get("processing_status", "pending")
                            ),
                            raw_data=deserialized.get("raw_data", {}),
                        )
                    )

            return results

        except ClientError as e:
            logger.error(f"DynamoDB error retrieving health data: {e}")
            raise DatabaseError(f"Failed to retrieve health data: {e!s}")
        except Exception as e:
            logger.error(f"Unexpected error retrieving health data: {e}")
            raise DatabaseError(f"Failed to retrieve health data: {e!s}")

    async def update_processing_status(
        self,
        data_id: str,
        status: ProcessingStatus,
        analysis_results: dict[str, Any] | None = None,
    ) -> None:
        """Update processing status in DynamoDB."""
        try:
            # Parse the composite ID
            user_id, timestamp = data_id.split("#", 1)

            update_expr = "SET processing_status = :status, updated_at = :updated"
            expr_values = {
                ":status": status.value,
                ":updated": datetime.utcnow().isoformat(),
            }

            if analysis_results:
                update_expr += ", analysis_results = :results"
                expr_values[":results"] = self._serialize_item(
                    {"data": analysis_results}
                )["data"]

            self.table.update_item(
                Key={"pk": f"USER#{user_id}", "sk": f"HEALTH#{timestamp}"},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_values,
            )

        except ClientError as e:
            logger.error(f"DynamoDB error updating status: {e}")
            raise DatabaseError(f"Failed to update status: {e!s}")

    async def get_user(self, user_id: str) -> User | None:
        """Get user from DynamoDB."""
        try:
            response = self.table.get_item(
                Key={"pk": f"USER#{user_id}", "sk": f"PROFILE#{user_id}"}
            )

            if "Item" in response:
                item = self._deserialize_item(response["Item"])
                return User(
                    id=user_id,
                    email=item.get("email", ""),
                    name=item.get("name", ""),
                    created_at=datetime.fromisoformat(item["created_at"]),
                    updated_at=datetime.fromisoformat(item["updated_at"]),
                )

            return None

        except ClientError as e:
            logger.error(f"DynamoDB error getting user: {e}")
            return None

    async def create_user(self, user_id: str, email: str, name: str) -> User:
        """Create user in DynamoDB."""
        try:
            now = datetime.utcnow()

            item = {
                "pk": f"USER#{user_id}",
                "sk": f"PROFILE#{user_id}",
                "user_id": user_id,
                "email": email,
                "name": name,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }

            self.table.put_item(Item=self._serialize_item(item))

            return User(
                id=user_id, email=email, name=name, created_at=now, updated_at=now
            )

        except ClientError as e:
            logger.error(f"DynamoDB error creating user: {e}")
            raise DatabaseError(f"Failed to create user: {e!s}")

    async def delete_user_data(self, user_id: str) -> None:
        """Delete all user data from DynamoDB."""
        try:
            # Query all items for the user
            response = self.table.query(
                KeyConditionExpression=Key("pk").eq(f"USER#{user_id}")
            )

            # Delete each item
            with self.table.batch_writer() as batch:
                for item in response.get("Items", []):
                    batch.delete_item(Key={"pk": item["pk"], "sk": item["sk"]})

        except ClientError as e:
            logger.error(f"DynamoDB error deleting user data: {e}")
            raise DatabaseError(f"Failed to delete user data: {e!s}")

    async def query_health_data_stream(
        self,
        user_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        batch_size: int = 100,
    ) -> AsyncIterator[list[HealthDataResponse]]:
        """Stream health data in batches (for large datasets)."""
        # For now, just yield a single batch
        # In production, this would implement pagination
        batch = await self.get_health_data(
            user_id=user_id, start_date=start_date, end_date=end_date, limit=batch_size
        )
        yield batch
