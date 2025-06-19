"""Mock Health Data Repository for Development.

Provides in-memory implementation of IHealthDataRepository for development
and testing without requiring external dependencies.

Following Clean Architecture and SOLID principles.
"""

# removed - breaks FastAPI

from datetime import UTC, datetime
import logging
from operator import itemgetter
from typing import Any
import uuid

from clarity.models.health_data import HealthMetric
from clarity.ports.data_ports import IHealthDataRepository

logger = logging.getLogger(__name__)


class MockHealthDataRepository(IHealthDataRepository):
    """Mock implementation of health data repository for development.

    Following Clean Architecture and SOLID principles:
    - Single Responsibility: Only handles mock health data storage
    - Open/Closed: Can be extended without modification
    - Liskov Substitution: Can substitute any IHealthDataRepository
    - Interface Segregation: Implements only needed methods
    - Dependency Inversion: Depends on abstractions
    """

    def __init__(self) -> None:
        """Initialize mock repository with in-memory storage."""
        self._health_data: dict[str, list[dict[str, Any]]] = {}
        self._processing_status: dict[str, dict[str, Any]] = {}
        logger.info("Mock health data repository initialized for development")

    async def save_health_data(
        self,
        user_id: str,
        processing_id: str,
        metrics: list[HealthMetric],
        upload_source: str,
        client_timestamp: datetime,
    ) -> bool:
        """Save health data to in-memory storage."""
        try:
            if user_id not in self._health_data:
                self._health_data[user_id] = []

            # Convert metrics to dict format for storage
            metrics_data = []
            for metric in metrics:
                metric_dict = {
                    "metric_id": str(metric.metric_id),
                    "metric_type": metric.metric_type.value,
                    "device_id": metric.device_id,
                    "raw_data": metric.raw_data,
                    "metadata": metric.metadata,
                    "created_at": metric.created_at.isoformat(),
                }

                # Add the specific data type if present
                if metric.biometric_data:
                    metric_dict["biometric_data"] = metric.biometric_data.model_dump()
                if metric.sleep_data:
                    metric_dict["sleep_data"] = metric.sleep_data.model_dump()
                if metric.activity_data:
                    metric_dict["activity_data"] = metric.activity_data.model_dump()
                if metric.mental_health_data:
                    metric_dict["mental_health_data"] = (
                        metric.mental_health_data.model_dump()
                    )

                metrics_data.append(metric_dict)

            # Store the health data entry
            entry: dict[str, Any] = {
                "processing_id": processing_id,
                "metrics": metrics_data,
                "upload_source": upload_source,
                "client_timestamp": client_timestamp.isoformat(),
                "server_timestamp": datetime.now(UTC).isoformat(),
            }

            self._health_data[user_id].append(entry)

            # Update processing status
            self._processing_status[processing_id] = {
                "processing_id": processing_id,
                "user_id": user_id,
                "status": "completed",
                "message": "Mock processing completed successfully",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "metrics_count": len(metrics),
                "upload_source": upload_source,
            }

            logger.info("Saved %s health metrics for user %s", len(metrics), user_id)
        except Exception:
            logger.exception("Failed to save health data")
            return False
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
        """Retrieve user health data with filtering and pagination."""
        if user_id not in self._health_data:
            return {
                "data": [],
                "total_count": 0,
                "page_info": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": False,
                },
            }

        # Get all entries for user
        all_entries = self._health_data[user_id]

        # Flatten metrics and apply filters
        all_metrics: list[dict[str, Any]] = []
        for entry in all_entries:
            for metric in entry["metrics"]:
                # Add entry metadata to metric
                metric_with_context = {
                    **metric,
                    "processing_id": entry["processing_id"],
                    "upload_source": entry["upload_source"],
                    "server_timestamp": entry["server_timestamp"],
                }

                # Apply filters
                if metric_type and metric["metric_type"] != metric_type:
                    continue

                metric_timestamp = datetime.fromisoformat(metric["created_at"])
                if start_date and metric_timestamp < start_date:
                    continue
                if end_date and metric_timestamp > end_date:
                    continue

                all_metrics.append(metric_with_context)

        # Sort by timestamp (newest first)
        all_metrics.sort(key=itemgetter("created_at"), reverse=True)

        # Apply pagination
        total_count = len(all_metrics)
        paginated_metrics = all_metrics[offset : offset + limit]

        return {
            "data": paginated_metrics,
            "total_count": total_count,
            "page_info": {
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(paginated_metrics) < total_count,
            },
        }

    async def get_processing_status(
        self, processing_id: str, user_id: str
    ) -> dict[str, Any] | None:
        """Get processing status for a health data upload."""
        status = self._processing_status.get(processing_id)
        if status and status["user_id"] == user_id:
            return status
        return None

    async def delete_health_data(
        self, user_id: str, processing_id: str | None = None
    ) -> bool:
        """Delete user health data."""
        try:
            if processing_id:
                # Delete specific processing job
                if user_id in self._health_data:
                    self._health_data[user_id] = [
                        entry
                        for entry in self._health_data[user_id]
                        if entry["processing_id"] != processing_id
                    ]

                # Remove from processing status
                if processing_id in self._processing_status:
                    del self._processing_status[processing_id]

                logger.info("Deleted health data for processing_id %s", processing_id)
            else:
                # Delete all data for user
                if user_id in self._health_data:
                    del self._health_data[user_id]

                # Remove all processing status for user
                to_remove = [
                    pid
                    for pid, status in self._processing_status.items()
                    if status["user_id"] == user_id
                ]
                for pid in to_remove:
                    del self._processing_status[pid]

                logger.info("Deleted all health data for user %s", user_id)

        except Exception:
            logger.exception("Failed to delete health data")
            return False
        else:
            return True

    async def save_data(self, user_id: str, data: dict[str, Any]) -> str:
        """Save health data for a user (legacy method)."""
        processing_id = str(uuid.uuid4())

        if user_id not in self._health_data:
            self._health_data[user_id] = []

        entry = {
            "processing_id": processing_id,
            "data": data,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self._health_data[user_id].append(entry)
        logger.info("Saved legacy health data for user %s", user_id)

        return processing_id

    async def get_data(
        self,
        user_id: str,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve health data for a user (legacy method)."""
        if user_id not in self._health_data:
            return {}

        return {
            "user_id": user_id,
            "entries": self._health_data[user_id],
            "count": len(self._health_data[user_id]),
        }

    async def initialize(self) -> None:
        """Initialize the mock repository."""
        logger.info("Mock health data repository initialized successfully")

    async def cleanup(self) -> None:
        """Clean up repository resources."""
        self._health_data.clear()
        self._processing_status.clear()
        logger.info("Mock health data repository cleaned up")
