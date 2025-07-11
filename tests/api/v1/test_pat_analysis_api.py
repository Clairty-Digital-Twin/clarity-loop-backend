"""Professional tests for PAT Analysis API endpoints.

Tests the REAL FastAPI HTTP layer for PAT analysis endpoints,
following the established architectural patterns from auth tests.
Uses dependency overrides instead of patches for clean, maintainable tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

import clarity.api.v1.pat_analysis
from clarity.api.v1.pat_analysis import (
    AnalysisResponse,
    DirectActigraphyRequest,
    HealthCheckResponse,
    PATAnalysisResponse,
    StepDataRequest,
    get_pat_inference_engine,
    router,
)
from clarity.auth.dependencies import get_authenticated_user, get_current_user
from clarity.ml.inference_engine import AsyncInferenceEngine
from clarity.ml.pat_service import ActigraphyAnalysis, PATModelService
from clarity.models.auth import UserContext
from clarity.storage.dynamodb_client import DynamoDBHealthDataRepository

# ===== FIXTURES FOLLOWING ESTABLISHED PATTERNS =====


@pytest.fixture
def mock_inference_engine() -> AsyncMock:
    """Create properly mocked inference engine."""
    engine = AsyncMock(spec=AsyncInferenceEngine)

    # Mock PAT service
    mock_pat_service = MagicMock(spec=PATModelService)
    mock_pat_service.is_loaded = True
    engine.pat_service = mock_pat_service

    # Mock inference response
    mock_analysis = ActigraphyAnalysis(
        user_id="test-user-123",
        analysis_timestamp=datetime.now(UTC).isoformat(),
        sleep_efficiency=0.85,
        sleep_onset_latency=10.0,
        wake_after_sleep_onset=15.0,
        total_sleep_time=8.0,
        circadian_rhythm_score=0.78,
        activity_fragmentation=0.25,
        depression_risk_score=0.2,
        sleep_stages=["awake", "light", "deep", "rem"],
        confidence_score=0.92,
        clinical_insights=["Good sleep efficiency", "Strong circadian rhythm"],
        embedding=[0.1] * 128,  # 128-dimensional embedding
    )

    engine.predict.return_value = AsyncMock(
        analysis=mock_analysis, processing_time_ms=1250.0, cached=False
    )

    # Mock stats
    engine.get_stats.return_value = {
        "requests_processed": 42,
        "average_processing_time": 1200,
        "cache_hit_rate": 0.65,
        "last_updated": datetime.now(UTC).isoformat(),
    }

    return engine


@pytest.fixture
def test_user() -> UserContext:
    """Create test user following established pattern."""
    return UserContext(
        user_id=str(uuid4()),
        email="test@example.com",
        role="patient",
        permissions=[],
        is_verified=True,
        is_active=True,
        custom_claims={},
        created_at=None,
        last_login=None,
    )


@pytest.fixture
def mock_dynamodb_repository() -> MagicMock:
    """Create mocked DynamoDB repository."""
    repo = MagicMock(spec=DynamoDBHealthDataRepository)

    # Mock table with query/get_item methods
    mock_table = MagicMock()
    mock_table.query.return_value = {"Items": []}
    mock_table.get_item.return_value = {"Item": None}
    repo.table = mock_table

    return repo


@pytest.fixture
def app(
    mock_inference_engine: AsyncMock,
    test_user: UserContext,
    mock_dynamodb_repository: MagicMock,
) -> FastAPI:
    """Create REAL FastAPI app following established pattern."""
    # Create app
    app = FastAPI()

    # Override ONLY the dependencies - everything else is REAL
    app.dependency_overrides[get_pat_inference_engine] = lambda: mock_inference_engine
    app.dependency_overrides[get_current_user] = lambda: test_user  # Router-level auth
    app.dependency_overrides[get_authenticated_user] = (
        lambda: test_user
    )  # Endpoint-level auth

    # Mock DynamoDB access at module level (since it's not dependency injected)
    clarity.api.v1.pat_analysis._get_analysis_repository = (
        lambda: mock_dynamodb_repository
    )

    # Include the REAL PAT analysis router
    app.include_router(router, prefix="/api/v1/pat")

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client with real app."""
    return TestClient(app)


@pytest.fixture
def valid_step_request_data() -> dict[str, Any]:
    """Create valid step data request."""
    base_time = datetime.now(UTC)
    # Create enough timestamps for proxy actigraphy expansion (1 week = 10080 minutes)
    timestamps = [
        (base_time + timedelta(minutes=i)).isoformat() for i in range(10080)  # 1 week
    ]

    return {
        "step_counts": list(range(100, 10180)),  # 10080 values to match timestamps
        "timestamps": timestamps,
        "user_metadata": {"age_group": "25-35", "sex": "F"},
    }


@pytest.fixture
def valid_actigraphy_request_data() -> dict[str, Any]:
    """Create valid actigraphy data request."""
    base_time = datetime.now(UTC)
    data_points = [
        {
            "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
            "value": float(50 + (i % 100)),  # Varying activity values
        }
        for i in range(1440)  # 24 hours of data
    ]

    return {"data_points": data_points, "sampling_rate": 1.0, "duration_hours": 24}


# ===== TESTS FOR STEP ANALYSIS ENDPOINT =====


class TestStepAnalysisEndpoint:
    """Test /step-analysis endpoint for Apple HealthKit data."""

    def test_step_analysis_success(
        self,
        client: TestClient,
        valid_step_request_data: dict[str, Any],
        mock_inference_engine: AsyncMock,
    ) -> None:
        """Test successful step data analysis."""
        response = client.post(
            "/api/v1/pat/step-analysis", json=valid_step_request_data
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "analysis_id" in data
        assert data["status"] == "completed"
        assert "analysis" in data
        assert "processing_time_ms" in data
        assert data["cached"] is False

        # Verify analysis data
        analysis = data["analysis"]
        assert "sleep_stages" in analysis
        assert "sleep_efficiency" in analysis
        assert analysis["sleep_efficiency"] == 0.85

    def test_step_analysis_validation_error_missing_data(
        self, client: TestClient
    ) -> None:
        """Test step analysis with missing required data."""
        response = client.post("/api/v1/pat/step-analysis", json={})

        assert response.status_code == 422  # Validation error

    def test_step_analysis_validation_error_mismatched_lengths(
        self, client: TestClient
    ) -> None:
        """Test step analysis with mismatched step_counts and timestamps."""
        response = client.post(
            "/api/v1/pat/step-analysis",
            json={
                "step_counts": [100, 200, 300],  # 3 values
                "timestamps": ["2024-01-01T00:00:00Z"],  # 1 timestamp
                "user_metadata": {"age_group": "25-35"},
            },
        )

        assert response.status_code == 422

    def test_step_analysis_inference_engine_error(
        self,
        client: TestClient,
        valid_step_request_data: dict[str, Any],
        mock_inference_engine: AsyncMock,
    ) -> None:
        """Test step analysis when inference engine fails."""
        # Make inference engine raise an exception
        mock_inference_engine.predict.side_effect = Exception("ML service unavailable")

        response = client.post(
            "/api/v1/pat/step-analysis", json=valid_step_request_data
        )

        assert response.status_code == 500
        data = response.json()
        assert "analysis_id" in data["detail"]
        assert "error" in data["detail"]


# ===== TESTS FOR DIRECT ACTIGRAPHY ENDPOINT =====


class TestDirectActigraphyEndpoint:
    """Test /analysis endpoint for direct actigraphy data."""

    def test_direct_actigraphy_success(
        self,
        client: TestClient,
        valid_actigraphy_request_data: dict[str, Any],
        mock_inference_engine: AsyncMock,
    ) -> None:
        """Test successful direct actigraphy analysis."""
        response = client.post(
            "/api/v1/pat/analysis", json=valid_actigraphy_request_data
        )

        assert response.status_code == 200
        data = response.json()

        assert "analysis_id" in data
        assert data["status"] == "completed"
        assert "analysis" in data
        assert data["cached"] is False

    def test_direct_actigraphy_validation_error(self, client: TestClient) -> None:
        """Test direct actigraphy with invalid data."""
        response = client.post(
            "/api/v1/pat/analysis",
            json={
                "data_points": [],  # Empty data points
                "sampling_rate": -1.0,  # Invalid sampling rate
                "duration_hours": 0,  # Invalid duration
            },
        )

        assert response.status_code == 422

    def test_direct_actigraphy_cached_result(
        self,
        client: TestClient,
        valid_actigraphy_request_data: dict[str, Any],
        mock_inference_engine: AsyncMock,
    ) -> None:
        """Test direct actigraphy with cached result."""
        # Configure inference engine to return cached result
        mock_inference_engine.predict.return_value.cached = True

        response = client.post(
            "/api/v1/pat/analysis", json=valid_actigraphy_request_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["cached"] is True


# ===== TESTS FOR RESULTS RETRIEVAL ENDPOINT =====


class TestAnalysisResultsEndpoint:
    """Test /analysis/{processing_id} endpoint for retrieving results."""

    def test_get_analysis_results_completed(
        self,
        client: TestClient,
        mock_dynamodb_repository: MagicMock,
    ) -> None:
        """Test retrieving completed analysis results."""
        # Configure DynamoDB to return completed analysis
        mock_dynamodb_repository.table.query.return_value = {
            "Items": [
                {
                    "status": "completed",
                    "analysis_date": "2024-01-01T12:00:00Z",
                    "pat_features": {"feature1": 0.85, "feature2": 0.92},
                    "activity_embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                }
            ]
        }

        processing_id = str(uuid4())
        response = client.get(f"/api/v1/pat/analysis/{processing_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["processing_id"] == processing_id
        assert data["status"] == "completed"
        assert "pat_features" in data
        assert "activity_embedding" in data

    def test_get_analysis_results_not_found(
        self,
        client: TestClient,
        mock_dynamodb_repository: MagicMock,
    ) -> None:
        """Test retrieving non-existent analysis results."""
        # Configure DynamoDB to return empty results
        mock_dynamodb_repository.table.query.return_value = {"Items": []}
        mock_dynamodb_repository.table.get_item.return_value = {"Item": None}

        processing_id = str(uuid4())
        response = client.get(f"/api/v1/pat/analysis/{processing_id}")

        assert response.status_code == 200  # Endpoint returns 200 with not_found status
        data = response.json()

        assert data["processing_id"] == processing_id
        assert data["status"] == "not_found"
        assert data["pat_features"] is None

    def test_get_analysis_results_processing(
        self,
        client: TestClient,
        mock_dynamodb_repository: MagicMock,
        test_user: UserContext,
    ) -> None:
        """Test retrieving processing job status."""
        # Configure DynamoDB: no analysis results, but processing job exists
        mock_dynamodb_repository.table.query.return_value = {"Items": []}
        mock_dynamodb_repository.table.get_item.return_value = {
            "Item": {
                "status": "processing",
                "user_id": test_user.user_id,
                "created_at": "2024-01-01T12:00:00Z",
            }
        }

        processing_id = str(uuid4())
        response = client.get(f"/api/v1/pat/analysis/{processing_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["processing_id"] == processing_id
        assert data["status"] == "processing"


# ===== TESTS FOR HEALTH CHECK ENDPOINT =====


class TestHealthCheckEndpoint:
    """Test /health endpoint."""

    def test_health_check_success(
        self,
        client: TestClient,
        mock_inference_engine: AsyncMock,
    ) -> None:
        """Test successful health check."""
        response = client.get("/api/v1/pat/health")

        assert response.status_code == 200
        data = response.json()

        assert data["service"] == "PAT Analysis API"
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "inference_engine" in data
        assert "pat_model" in data

    def test_health_check_unhealthy(
        self,
        client: TestClient,
        mock_inference_engine: AsyncMock,
    ) -> None:
        """Test health check when service is unhealthy."""
        # Make inference engine stats fail
        mock_inference_engine.get_stats.side_effect = Exception("Service unavailable")

        response = client.get("/api/v1/pat/health")

        assert response.status_code == 200  # Health endpoint always returns 200
        data = response.json()

        assert data["status"] == "unhealthy"
        assert "error" in data["inference_engine"]


# ===== TESTS FOR MODEL INFO ENDPOINT =====


class TestModelInfoEndpoint:
    """Test /models/info endpoint."""

    def test_model_info_success(
        self,
        client: TestClient,
        mock_inference_engine: AsyncMock,
    ) -> None:
        """Test successful model info retrieval."""
        response = client.get("/api/v1/pat/models/info")

        assert response.status_code == 200
        data = response.json()

        assert data["model_type"] == "PAT"
        assert data["version"] == "1.0.0"
        assert data["initialized"] is True
        assert "capabilities" in data
        assert "input_requirements" in data
        assert "performance" in data

    def test_model_info_service_error(
        self,
        client: TestClient,
        mock_inference_engine: AsyncMock,
    ) -> None:
        """Test model info when service fails."""
        # Make PAT service access fail
        mock_inference_engine.pat_service = None

        response = client.get("/api/v1/pat/models/info")

        assert response.status_code == 500


# ===== REQUEST/RESPONSE MODEL VALIDATION TESTS =====


class TestRequestValidation:
    """Test Pydantic model validation for requests."""

    def test_step_data_request_validation_success(
        self, valid_step_request_data: dict[str, Any]
    ) -> None:
        """Test valid step data request passes validation."""
        request = StepDataRequest(**valid_step_request_data)
        assert len(request.step_counts) == len(request.timestamps)
        assert request.user_metadata is not None

    def test_step_data_request_validation_length_mismatch(self) -> None:
        """Test step data request fails validation with mismatched lengths."""
        with pytest.raises(ValidationError, match="Timestamps length must match"):
            StepDataRequest(
                step_counts=[100, 200],
                timestamps=["2024-01-01T00:00:00Z"],
                user_metadata={"age_group": "25-35"},
            )

    def test_direct_actigraphy_request_validation_success(
        self, valid_actigraphy_request_data: dict[str, Any]
    ) -> None:
        """Test valid actigraphy request passes validation."""
        request = DirectActigraphyRequest(**valid_actigraphy_request_data)
        assert len(request.data_points) > 0
        assert request.sampling_rate > 0
        assert request.duration_hours > 0

    def test_direct_actigraphy_request_validation_errors(self) -> None:
        """Test actigraphy request validation catches errors."""
        with pytest.raises(ValueError, match="validation error"):
            DirectActigraphyRequest(
                data_points=[], sampling_rate=-1.0, duration_hours=0
            )


# ===== RESPONSE MODEL TESTS =====


class TestResponseModels:
    """Test response model validation."""

    def test_analysis_response_model(self) -> None:
        """Test AnalysisResponse model validation."""
        response = AnalysisResponse(
            analysis_id=str(uuid4()),
            status="completed",
            processing_time_ms=1200.5,
            cached=False,
        )
        assert response.analysis_id is not None
        assert response.status == "completed"

    def test_pat_analysis_response_model(self) -> None:
        """Test PATAnalysisResponse model validation."""
        response = PATAnalysisResponse(
            processing_id=str(uuid4()),
            status="completed",
            message="Analysis complete",
            analysis_date="2024-01-01T12:00:00Z",
            pat_features={"feature1": 0.85},
            activity_embedding=[0.1, 0.2, 0.3],
            metadata={"additional": "data"},
        )
        assert response.processing_id is not None
        assert response.status == "completed"

    def test_health_check_response_model(self) -> None:
        """Test HealthCheckResponse model validation."""
        response = HealthCheckResponse(
            service="PAT Analysis API",
            status="healthy",
            timestamp="2024-01-01T12:00:00Z",
            inference_engine={"requests": 42},
            pat_model={"initialized": True},
        )
        assert response.service == "PAT Analysis API"
        assert response.status == "healthy"
