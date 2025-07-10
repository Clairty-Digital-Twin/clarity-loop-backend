"""Performance testing configuration using Locust.

Defines user behavior scenarios and load patterns for
testing the CLARITY backend API performance.
"""

import json
import random
from typing import Any

from locust import HttpUser, between, task
from locust.exception import RescheduleTask


class ClarityAPIUser(HttpUser):
    """Simulates a typical CLARITY API user."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.access_token = None
        self.user_id = None
        self.health_data_ids = []

    def on_start(self):
        """Called when a simulated user starts."""
        self.login()

    def login(self):
        """Authenticate and obtain access token."""
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": f"test_user_{random.randint(1, 100)}@example.com",
                "password": "TestPassword123!",
            },
            catch_response=True,
        )

        if response.status_code == 200:
            data = response.json()
            self.access_token = data.get("access_token")
            self.user_id = data.get("user_id")
            response.success()
        else:
            response.failure(f"Login failed: {response.status_code}")
            raise RescheduleTask()

    @property
    def auth_headers(self) -> dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    @task(3)
    def check_health(self):
        """Health check endpoint - high frequency."""
        self.client.get("/health")

    @task(5)
    def upload_health_data(self):
        """Upload health metrics."""
        metrics = self._generate_health_metrics()

        with self.client.post(
            "/api/v1/health/metrics",
            json=metrics,
            headers=self.auth_headers,
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                data = response.json()
                self.health_data_ids.append(data.get("id"))
                response.success()
            else:
                response.failure(f"Upload failed: {response.status_code}")

    @task(4)
    def get_recent_metrics(self):
        """Retrieve recent health metrics."""
        with self.client.get(
            "/api/v1/health/metrics?limit=10",
            headers=self.auth_headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Get metrics failed: {response.status_code}")

    @task(2)
    def get_analysis(self):
        """Request health analysis."""
        if not self.health_data_ids:
            raise RescheduleTask()

        with self.client.get(
            "/api/v1/analysis/latest",
            headers=self.auth_headers,
            catch_response=True,
        ) as response:
            if response.status_code in [200, 202]:  # 202 for async processing
                response.success()
            else:
                response.failure(f"Analysis failed: {response.status_code}")

    @task(1)
    def get_user_profile(self):
        """Retrieve user profile."""
        with self.client.get(
            "/api/v1/users/me",
            headers=self.auth_headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Profile failed: {response.status_code}")

    def _generate_health_metrics(self) -> dict[str, Any]:
        """Generate random health metrics."""
        return {
            "metrics": [
                {
                    "type": "heart_rate",
                    "value": random.randint(60, 100),
                    "timestamp": "2024-01-01T12:00:00Z",
                    "unit": "bpm",
                },
                {
                    "type": "steps",
                    "value": random.randint(0, 2000),
                    "timestamp": "2024-01-01T12:00:00Z",
                    "unit": "count",
                },
                {
                    "type": "sleep_duration",
                    "value": round(random.uniform(6, 9), 1),
                    "timestamp": "2024-01-01T08:00:00Z",
                    "unit": "hours",
                },
            ]
        }


class ClarityWebSocketUser(HttpUser):
    """Simulates WebSocket connections for real-time features."""

    wait_time = between(5, 10)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = None
        self.access_token = None

    def on_start(self):
        """Initialize WebSocket connection."""
        # First authenticate via HTTP
        self.login()
        # Note: Locust doesn't natively support WebSocket
        # This is a placeholder for WebSocket load testing
        # Consider using locust-plugins for WebSocket support

    def login(self):
        """Authenticate to get WebSocket token."""
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": f"ws_user_{random.randint(1, 50)}@example.com",
                "password": "TestPassword123!",
            },
        )

        if response.status_code == 200:
            data = response.json()
            self.access_token = data.get("access_token")

    @task
    def simulate_websocket_activity(self):
        """Simulate WebSocket-like activity via HTTP."""
        # This would normally be WebSocket messages
        # Using HTTP endpoints as a proxy for load testing
        self.client.post(
            "/api/v1/health/realtime",
            json={
                "type": "heartbeat",
                "data": {"status": "active"},
            },
            headers={"Authorization": f"Bearer {self.access_token}"},
        )


class ClarityHeavyUser(HttpUser):
    """Simulates power users with heavy API usage."""

    wait_time = between(0.5, 2)  # More aggressive timing

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.access_token = None
        self.batch_size = 50  # Larger batches

    def on_start(self):
        """Initialize heavy user session."""
        self.login()

    def login(self):
        """Authenticate as power user."""
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": f"power_user_{random.randint(1, 10)}@example.com",
                "password": "TestPassword123!",
            },
        )

        if response.status_code == 200:
            data = response.json()
            self.access_token = data.get("access_token")

    @task(5)
    def batch_upload(self):
        """Upload large batches of data."""
        metrics = []
        for _ in range(self.batch_size):
            metrics.extend(self._generate_health_metrics()["metrics"])

        self.client.post(
            "/api/v1/health/metrics/batch",
            json={"metrics": metrics},
            headers={"Authorization": f"Bearer {self.access_token}"},
        )

    @task(3)
    def complex_query(self):
        """Execute complex data queries."""
        # Date range query with multiple filters
        params = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "metric_types": "heart_rate,steps,sleep_duration",
            "aggregation": "daily",
            "include_analysis": "true",
        }

        self.client.get(
            "/api/v1/health/metrics/aggregate",
            params=params,
            headers={"Authorization": f"Bearer {self.access_token}"},
        )

    @task(2)
    def export_data(self):
        """Request data export."""
        self.client.post(
            "/api/v1/export/request",
            json={
                "format": "csv",
                "date_range": "last_30_days",
                "include_predictions": True,
            },
            headers={"Authorization": f"Bearer {self.access_token}"},
        )

    @task(1)
    def ml_prediction(self):
        """Request ML predictions."""
        self.client.post(
            "/api/v1/ml/predict",
            json={
                "model": "pat_v2",
                "features": self._generate_ml_features(),
            },
            headers={"Authorization": f"Bearer {self.access_token}"},
        )

    def _generate_health_metrics(self) -> dict[str, Any]:
        """Generate random health metrics."""
        return {
            "metrics": [
                {
                    "type": random.choice(["heart_rate", "steps", "sleep_duration"]),
                    "value": random.randint(50, 150),
                    "timestamp": f"2024-01-{random.randint(1, 31):02d}T{random.randint(0, 23):02d}:00:00Z",
                    "unit": "various",
                }
                for _ in range(10)
            ]
        }

    def _generate_ml_features(self) -> list[float]:
        """Generate random ML features."""
        return [random.random() for _ in range(50)]
