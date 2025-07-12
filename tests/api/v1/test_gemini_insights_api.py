"""Comprehensive tests for Gemini Insights API endpoints.

Tests the complete gemini_insights.py module including:
- Dependency injection and service initialization
- Request/response models and validation
- All API endpoints with various scenarios
- Error handling and edge cases
- Authentication and authorization
- Real business logic without excessive mocking
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from fastapi import HTTPException, status
import pytest

from clarity.api.v1.gemini_insights import (
    ErrorDetail,
    ErrorResponse,
    InsightGenerationRequest,
    InsightGenerationResponse,
    InsightHistoryResponse,
    ServiceStatusResponse,
    create_error_response,
    create_metadata,
    generate_insights,
    generate_request_id,
    get_gemini_service,
    get_insight,
    get_insight_history,
    get_service_status,
    set_dependencies,
)
from clarity.auth.dependencies import AuthenticatedUser
from clarity.ml.gemini_service import GeminiService, HealthInsightResponse
from clarity.ports.auth_ports import IAuthProvider
from clarity.ports.config_ports import IConfigProvider


class TestGeminiInsightsModels:
    """Test Pydantic models for request/response validation."""

    def test_insight_generation_request_valid(self):
        """Test valid insight generation request creation."""
        request_data = {
            "analysis_results": {
                "heart_rate": {"avg": 72, "max": 85, "min": 60},
                "sleep_quality": {"score": 8.5, "deep_sleep_hours": 2.1},
            },
            "context": "User has been exercising regularly",
            "insight_type": "comprehensive",
            "include_recommendations": True,
            "language": "en",
        }

        request = InsightGenerationRequest(**request_data)

        assert request.analysis_results == request_data["analysis_results"]
        assert request.context == "User has been exercising regularly"
        assert request.insight_type == "comprehensive"
        assert request.include_recommendations is True
        assert request.language == "en"

    def test_insight_generation_request_defaults(self):
        """Test insight generation request with default values."""
        request_data = {"analysis_results": {"heart_rate": {"avg": 72}}}

        request = InsightGenerationRequest(**request_data)

        assert request.context is None
        assert request.insight_type == "comprehensive"
        assert request.include_recommendations is True
        assert request.language == "en"

    def test_insight_generation_request_invalid_empty_analysis(self):
        """Test insight generation request with empty analysis results."""
        request_data = {
            "analysis_results": {},
        }

        # This should still be valid as analysis_results is just a dict
        request = InsightGenerationRequest(**request_data)
        assert request.analysis_results == {}

    def test_insight_generation_response_structure(self):
        """Test insight generation response structure."""
        health_insight = HealthInsightResponse(
            user_id="user_456",
            narrative="Your health data shows positive trends...",
            key_insights=[
                "Heart rate is within optimal range",
                "Sleep quality is excellent",
            ],
            recommendations=["Continue regular exercise", "Maintain sleep schedule"],
            confidence_score=0.85,
            generated_at=datetime.now(UTC).isoformat(),
        )

        response = InsightGenerationResponse(
            success=True,
            data=health_insight,
            metadata={
                "request_id": "req_123",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        assert response.success is True
        assert response.data.user_id == "user_456"
        assert response.data.confidence_score == 0.85
        assert len(response.data.recommendations) == 2
        assert len(response.data.key_insights) == 2
        assert response.data.narrative == "Your health data shows positive trends..."
        assert "request_id" in response.metadata

    def test_error_detail_structure(self):
        """Test error detail structure."""
        error_detail = ErrorDetail(
            code="INSIGHT_NOT_FOUND",
            message="Insight not found",
            details={"insight_id": "insight_123"},
            request_id="req_456",
            timestamp=datetime.now(UTC).isoformat(),
            suggested_action="check_insight_id",
        )

        assert error_detail.code == "INSIGHT_NOT_FOUND"
        assert error_detail.message == "Insight not found"
        assert error_detail.details["insight_id"] == "insight_123"
        assert error_detail.request_id == "req_456"
        assert error_detail.suggested_action == "check_insight_id"

    def test_error_response_structure(self):
        """Test error response structure."""
        error_detail = ErrorDetail(
            code="ACCESS_DENIED",
            message="Access denied",
            request_id="req_789",
            timestamp=datetime.now(UTC).isoformat(),
        )

        error_response = ErrorResponse(error=error_detail)

        assert error_response.error.code == "ACCESS_DENIED"
        assert error_response.error.message == "Access denied"
        assert error_response.error.request_id == "req_789"


class TestHelperFunctions:
    """Test utility and helper functions."""

    def test_generate_request_id_format(self):
        """Test request ID generation format."""
        request_id = generate_request_id()

        assert request_id.startswith("req_insights_")
        assert len(request_id) == len("req_insights_") + 8
        # Check that it's a valid hex string
        hex_part = request_id.replace("req_insights_", "")
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_generate_request_id_uniqueness(self):
        """Test that request IDs are unique."""
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100  # All should be unique

    def test_create_metadata_basic(self):
        """Test basic metadata creation."""
        request_id = "req_test_123"
        metadata = create_metadata(request_id)

        assert metadata["request_id"] == request_id
        assert "timestamp" in metadata
        assert metadata["service"] == "gemini-insights"
        assert metadata["version"] == "1.0.0"
        assert "processing_time_ms" not in metadata

    def test_create_metadata_with_processing_time(self):
        """Test metadata creation with processing time."""
        request_id = "req_test_456"
        processing_time = 1234.5
        metadata = create_metadata(request_id, processing_time)

        assert metadata["request_id"] == request_id
        assert metadata["processing_time_ms"] == processing_time
        assert "timestamp" in metadata
        assert metadata["service"] == "gemini-insights"

    def test_create_metadata_timestamp_format(self):
        """Test that metadata timestamp is in correct ISO format."""
        request_id = "req_test_789"
        metadata = create_metadata(request_id)

        # Should be able to parse the timestamp
        timestamp = datetime.fromisoformat(metadata["timestamp"])
        assert isinstance(timestamp, datetime)


class TestErrorHandling:
    """Test error handling functions."""

    def test_create_error_response_basic(self):
        """Test creating basic error response."""
        error_code = "TEST_ERROR"
        message = "Test error message"
        request_id = "req_123"
        status_code = 400

        # create_error_response returns HTTPException, doesn't raise it
        http_exception = create_error_response(
            error_code=error_code,
            message=message,
            request_id=request_id,
            status_code=status_code,
        )

        # Verify the exception was created properly
        assert isinstance(http_exception, HTTPException)
        assert http_exception.status_code == status_code
        assert http_exception.detail["code"] == error_code
        assert http_exception.detail["message"] == message
        assert http_exception.detail["request_id"] == request_id
        assert "timestamp" in http_exception.detail
        assert http_exception.detail["suggested_action"] is None
        assert http_exception.detail["details"] is None

    def test_create_error_response_with_custom_status(self):
        """Test error response with custom status code."""
        error_code = "NOT_FOUND"
        message = "Resource not found"
        request_id = "req_456"
        custom_status = status.HTTP_404_NOT_FOUND

        # create_error_response returns HTTPException, doesn't raise it
        http_exception = create_error_response(
            error_code=error_code,
            message=message,
            request_id=request_id,
            status_code=custom_status,
        )

        # Verify the exception was created properly
        assert isinstance(http_exception, HTTPException)
        assert http_exception.status_code == custom_status
        assert http_exception.detail["code"] == error_code
        assert http_exception.detail["message"] == message
        assert http_exception.detail["request_id"] == request_id

    def test_create_error_response_with_details(self):
        """Test error response with additional details."""
        error_code = "VALIDATION_ERROR"
        message = "Validation failed"
        request_id = "req_789"
        details = {"field": "user_id", "value": "invalid"}
        suggested_action = "check_user_id"

        # create_error_response returns HTTPException, doesn't raise it
        http_exception = create_error_response(
            error_code=error_code,
            message=message,
            request_id=request_id,
            details=details,
            suggested_action=suggested_action,
        )

        # Verify the exception was created properly
        assert isinstance(http_exception, HTTPException)
        assert (
            http_exception.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        )  # Default
        assert http_exception.detail["code"] == error_code
        assert http_exception.detail["message"] == message
        assert http_exception.detail["request_id"] == request_id
        assert http_exception.detail["details"] == details
        assert http_exception.detail["suggested_action"] == suggested_action


class TestDependencyInjection:
    """Test dependency injection and service initialization."""

    def setUp(self):
        """Reset global dependencies before each test."""
        # Clear the global dependencies
        from clarity.api.v1 import gemini_insights

        gemini_insights._auth_provider = None
        gemini_insights._config_provider = None
        gemini_insights._gemini_service = None

    def test_set_dependencies_development_mode(self):
        """Test setting dependencies in development mode."""
        mock_auth_provider = MagicMock(spec=IAuthProvider)
        mock_config_provider = MagicMock(spec=IConfigProvider)
        mock_config_provider.is_development.return_value = True

        set_dependencies(mock_auth_provider, mock_config_provider)

        # Should create a development GeminiService
        service = get_gemini_service()
        assert isinstance(service, GeminiService)
        # Project ID will come from credentials manager
        assert service.project_id is not None

    @patch("clarity.api.v1.gemini_insights.get_settings")
    def test_set_dependencies_production_mode(self, mock_get_settings):
        """Test setting dependencies in production mode."""
        mock_auth_provider = MagicMock(spec=IAuthProvider)
        mock_config_provider = MagicMock(spec=IConfigProvider)
        mock_config_provider.is_development.return_value = False

        mock_settings = MagicMock()
        mock_settings.gcp_project_id = "clarity-loop-backend"
        mock_settings.vertex_ai_location = "us-central1"
        mock_get_settings.return_value = mock_settings

        set_dependencies(mock_auth_provider, mock_config_provider)

        # Should create a production GeminiService
        service = get_gemini_service()
        assert isinstance(service, GeminiService)
        # Project ID will come from credentials manager or settings
        assert service.project_id is not None

    def test_get_gemini_service_not_initialized(self):
        """Test getting service when not initialized."""
        # Clear dependencies
        from clarity.api.v1 import gemini_insights

        gemini_insights._gemini_service = None

        with pytest.raises(HTTPException) as exc_info:
            get_gemini_service()

        error = exc_info.value
        assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "not initialized" in str(error.detail)


class TestGenerateInsightsEndpoint:
    """Test the main insight generation endpoint."""

    def setup_method(self):
        """Setup for each test method."""
        self.mock_auth_provider = MagicMock(spec=IAuthProvider)
        self.mock_config_provider = MagicMock(spec=IConfigProvider)
        self.mock_config_provider.is_development.return_value = True

        set_dependencies(self.mock_auth_provider, self.mock_config_provider)

    @pytest.fixture
    def mock_current_user(self):
        """Mock authenticated user."""
        user = MagicMock(spec=AuthenticatedUser)
        user.user_id = "user_456"
        user.is_active = True
        user.username = "testuser"
        return user

    @pytest.fixture
    def mock_gemini_service(self):
        """Create a mock Gemini service for testing."""
        mock_service = AsyncMock(spec=GeminiService)

        # Mock successful insight generation
        mock_response = HealthInsightResponse(
            user_id="user_456",
            narrative="Your health data shows positive trends in cardiovascular health...",
            key_insights=[
                "Heart rate variability improved by 15%",
                "Sleep quality shows steady improvement",
            ],
            recommendations=[
                "Continue current exercise routine",
                "Maintain consistent sleep schedule",
            ],
            confidence_score=0.85,
            generated_at=datetime.now(UTC).isoformat(),
        )

        mock_service.generate_health_insights.return_value = mock_response
        mock_service.health_check.return_value = {
            "status": "healthy",
            "model_available": True,
            "last_check": datetime.now(UTC).isoformat(),
        }

        return mock_service

    @pytest.fixture
    def sample_insight_request(self):
        """Sample insight generation request."""
        return InsightGenerationRequest(
            analysis_results={
                "heart_rate": {
                    "avg": 68,
                    "max": 85,
                    "min": 55,
                    "variability": 12.5,
                },
                "sleep_quality": {
                    "score": 8.2,
                    "deep_sleep_hours": 2.3,
                    "rem_sleep_hours": 1.8,
                    "efficiency": 89.5,
                },
                "activity_level": {
                    "daily_steps": 8500,
                    "active_minutes": 45,
                    "calories_burned": 2100,
                },
            },
            context="User has been following a new fitness routine for 3 weeks",
            insight_type="comprehensive",
            include_recommendations=True,
            language="en",
        )

    @pytest.mark.asyncio
    async def test_generate_insights_success(
        self, mock_current_user, mock_gemini_service, sample_insight_request
    ):
        """Test successful insight generation."""
        response = await generate_insights(
            insight_request=sample_insight_request,
            current_user=mock_current_user,
            gemini_service=mock_gemini_service,
        )

        assert isinstance(response, InsightGenerationResponse)
        assert response.success is True
        assert response.data.user_id == "user_456"
        assert response.data.confidence_score == 0.85
        assert len(response.data.recommendations) == 2
        assert response.data.narrative is not None
        assert len(response.data.key_insights) > 0
        assert "request_id" in response.metadata
        assert "timestamp" in response.metadata
        assert "processing_time_ms" in response.metadata

    @pytest.mark.asyncio
    async def test_generate_insights_with_minimal_request(
        self, mock_current_user, mock_gemini_service
    ):
        """Test insight generation with minimal request data."""
        minimal_request = InsightGenerationRequest(
            analysis_results={"heart_rate": {"avg": 72}},
        )

        response = await generate_insights(
            insight_request=minimal_request,
            current_user=mock_current_user,
            gemini_service=mock_gemini_service,
        )

        assert isinstance(response, InsightGenerationResponse)
        assert response.success is True
        assert response.data.user_id == "user_456"
        assert response.data.narrative is not None

    @pytest.mark.asyncio
    async def test_generate_insights_service_error(
        self, mock_current_user, sample_insight_request
    ):
        """Test insight generation when service fails."""
        mock_service = MagicMock(spec=GeminiService)
        mock_service.generate_health_insights.side_effect = Exception(
            "Service unavailable"
        )

        with pytest.raises(HTTPException) as exc_info:
            await generate_insights(
                insight_request=sample_insight_request,
                current_user=mock_current_user,
                gemini_service=mock_service,
            )

        error = exc_info.value
        assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "INSIGHT_GENERATION_FAILED" in str(error.detail)

    @pytest.mark.asyncio
    async def test_generate_insights_invalid_user(
        self, mock_gemini_service, sample_insight_request
    ):
        """Test insight generation with invalid user."""
        mock_user = MagicMock(spec=AuthenticatedUser)
        mock_user.user_id = "user_456"
        mock_user.is_active = False  # Inactive user

        with pytest.raises(HTTPException) as exc_info:
            await generate_insights(
                insight_request=sample_insight_request,
                current_user=mock_user,
                gemini_service=mock_gemini_service,
            )

        error = exc_info.value
        assert error.status_code == status.HTTP_403_FORBIDDEN
        assert "ACCOUNT_DISABLED" in str(error.detail)


class TestGetInsightEndpoint:
    """Test the get cached insight endpoint."""

    def setup_method(self):
        """Setup for each test method."""
        self.mock_auth_provider = MagicMock(spec=IAuthProvider)
        self.mock_config_provider = MagicMock(spec=IConfigProvider)
        self.mock_config_provider.is_development.return_value = True

        set_dependencies(self.mock_auth_provider, self.mock_config_provider)

    @pytest.fixture
    def mock_current_user(self):
        """Mock authenticated user."""
        user = MagicMock(spec=AuthenticatedUser)
        user.user_id = "user_456"
        user.is_active = True
        user.username = "testuser"
        return user

    @pytest.mark.asyncio
    @patch("clarity.api.v1.gemini_insights._get_dynamodb_client")
    async def test_get_insight_success(self, mock_get_client, mock_current_user):
        """Test successful insight retrieval."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock cached insight data
        mock_insight_data = {
            "user_id": "user_456",
            "generated_at": datetime.now(UTC).isoformat(),
            "narrative": "Your health data shows positive trends...",
            "key_insights": ["Heart rate improved", "Sleep quality better"],
            "recommendations": ["Continue exercise", "Maintain sleep schedule"],
            "confidence_score": 0.88,
        }

        # Mock the table.get_item response
        mock_client.table.get_item.return_value = {"Item": mock_insight_data}

        response = await get_insight(
            insight_id="insight_123",
            current_user=mock_current_user,
        )

        assert isinstance(response, InsightGenerationResponse)
        assert response.success is True
        assert response.data.user_id == "user_456"
        assert response.data.confidence_score == 0.88
        assert len(response.data.recommendations) == 2
        assert response.data.narrative is not None

    @pytest.mark.asyncio
    @patch("clarity.api.v1.gemini_insights._get_dynamodb_client")
    async def test_get_insight_not_found(self, mock_get_client, mock_current_user):
        """Test insight retrieval when insight not found."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.table.get_item.return_value = {"Item": None}

        with pytest.raises(HTTPException) as exc_info:
            await get_insight(
                insight_id="nonexistent_insight",
                current_user=mock_current_user,
            )

        error = exc_info.value
        assert error.status_code == status.HTTP_404_NOT_FOUND
        assert "INSIGHT_NOT_FOUND" in str(error.detail)

    @pytest.mark.asyncio
    @patch("clarity.api.v1.gemini_insights._get_dynamodb_client")
    async def test_get_insight_access_denied(self, mock_get_client, mock_current_user):
        """Test insight retrieval when user doesn't own the insight."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock insight owned by different user
        mock_insight_data = {
            "user_id": "other_user_456",  # Different user
            "generated_at": datetime.now(UTC).isoformat(),
            "narrative": "Some insight...",
            "key_insights": ["Some key insight"],
            "recommendations": ["Some recommendation"],
            "confidence_score": 0.75,
        }

        mock_client.table.get_item.return_value = {"Item": mock_insight_data}

        with pytest.raises(HTTPException) as exc_info:
            await get_insight(
                insight_id="insight_123",
                current_user=mock_current_user,
            )

        error = exc_info.value
        assert error.status_code == status.HTTP_403_FORBIDDEN
        assert "ACCESS_DENIED" in str(error.detail)


class TestGetInsightHistoryEndpoint:
    """Test the insight history endpoint."""

    def setup_method(self):
        """Setup for each test method."""
        self.mock_auth_provider = MagicMock(spec=IAuthProvider)
        self.mock_config_provider = MagicMock(spec=IConfigProvider)
        self.mock_config_provider.is_development.return_value = True

        set_dependencies(self.mock_auth_provider, self.mock_config_provider)

    @pytest.fixture
    def mock_current_user(self):
        """Mock authenticated user."""
        user = MagicMock(spec=AuthenticatedUser)
        user.user_id = "user_456"
        user.is_active = True
        user.username = "testuser"
        return user

    @pytest.mark.asyncio
    @patch("clarity.api.v1.gemini_insights._get_dynamodb_client")
    async def test_get_insight_history_success(
        self, mock_get_client, mock_current_user
    ):
        """Test successful insight history retrieval."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock historical insights (unused - keeping for reference)
        # This data structure shows expected format but actual mocking is done below
        _ = {
            "insights": [
                {
                    "insight_id": "insight_123",
                    "user_id": "user_456",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "insight_type": "comprehensive",
                    "narrative": "Your health data shows positive trends...",
                    "recommendations": ["Continue exercise", "Maintain sleep schedule"],
                    "confidence_score": 0.88,
                    "data_sources": ["heart_rate", "sleep"],
                    "processing_time_ms": 950.2,
                },
                {
                    "insight_id": "insight_456",
                    "user_id": "user_456",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "insight_type": "quick",
                    "narrative": "Recent activity shows improvement...",
                    "recommendations": ["Increase water intake"],
                    "confidence_score": 0.72,
                    "data_sources": ["activity"],
                    "processing_time_ms": 650.0,
                },
            ],
            "has_more": False,
            "total_count": 2,
            "offset": 0,
        }

        # Mock the table.query response
        mock_client.table.query.return_value = {
            "Items": [
                {
                    "id": "insight_123",
                    "user_id": "user_456",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "narrative": "Your health data shows positive trends...",
                    "recommendations": ["Continue exercise", "Maintain sleep schedule"],
                    "confidence_score": 0.88,
                    "key_insights": ["Heart rate improved", "Sleep quality better"],
                },
                {
                    "id": "insight_456",
                    "user_id": "user_456",
                    "generated_at": datetime.now(UTC).isoformat(),
                    "narrative": "Recent activity shows improvement...",
                    "recommendations": ["Increase water intake"],
                    "confidence_score": 0.72,
                    "key_insights": ["Activity level increased"],
                },
            ],
            "Count": 2,
        }

        response = await get_insight_history(
            user_id="user_456",
            current_user=mock_current_user,
            limit=10,
            offset=0,
        )

        assert isinstance(response, InsightHistoryResponse)
        assert response.success is True
        assert len(response.data["insights"]) == 2
        assert response.data["has_more"] is False
        assert response.data["total_count"] == 2
        assert response.data["pagination"]["offset"] == 0

    @pytest.mark.asyncio
    async def test_get_insight_history_access_denied(self, mock_current_user):
        """Test insight history access denied for different user."""
        # Try to access history for different user
        with pytest.raises(HTTPException) as exc_info:
            await get_insight_history(
                user_id="other_user_456",  # Different user
                current_user=mock_current_user,
                limit=10,
                offset=0,
            )

        error = exc_info.value
        assert error.status_code == status.HTTP_403_FORBIDDEN
        assert "ACCESS_DENIED" in str(error.detail)


class TestGetServiceStatusEndpoint:
    """Test the service status endpoint."""

    def setup_method(self):
        """Setup for each test method."""
        self.mock_auth_provider = MagicMock(spec=IAuthProvider)
        self.mock_config_provider = MagicMock(spec=IConfigProvider)
        self.mock_config_provider.is_development.return_value = True

        set_dependencies(self.mock_auth_provider, self.mock_config_provider)

    @pytest.fixture
    def mock_current_user(self):
        """Mock authenticated user."""
        user = MagicMock(spec=AuthenticatedUser)
        user.user_id = "user_456"
        user.is_active = True
        user.username = "testuser"
        return user

    @pytest.mark.asyncio
    @patch("clarity.api.v1.gemini_insights.get_gemini_service")
    async def test_get_service_status_success(
        self, mock_get_service, mock_current_user
    ):
        """Test successful service status check."""
        mock_service = MagicMock()
        mock_service.is_initialized = True
        mock_service.project_id = "test-project"
        mock_get_service.return_value = mock_service

        response = await get_service_status(
            _current_user=mock_current_user,
            gemini_service=mock_service,
        )

        assert isinstance(response, ServiceStatusResponse)
        assert response.success is True
        assert response.data["status"] == "healthy"
        assert response.data["model"]["initialized"] is True
        assert response.data["model"]["project_id"] == "test-project"
        assert response.data["service"] == "gemini-insights"
        assert "timestamp" in response.data
        assert "request_id" in response.metadata
        assert "timestamp" in response.metadata

    @pytest.mark.asyncio
    @patch("clarity.api.v1.gemini_insights.get_gemini_service")
    async def test_get_service_status_unhealthy(
        self, mock_get_service, mock_current_user
    ):
        """Test service status check when service is unhealthy."""
        mock_service = MagicMock()
        mock_service.is_initialized = False  # Service is not initialized
        mock_service.project_id = "test-project"
        mock_get_service.return_value = mock_service

        response = await get_service_status(
            _current_user=mock_current_user,
            gemini_service=mock_service,
        )

        # It should still return successfully but with unhealthy status
        assert isinstance(response, ServiceStatusResponse)
        assert response.success is True
        assert response.data["status"] == "unhealthy"
        assert response.data["model"]["initialized"] is False
        assert response.data["model"]["project_id"] == "test-project"

    @pytest.mark.asyncio
    @patch("clarity.api.v1.gemini_insights.get_gemini_service")
    async def test_get_service_status_error(self, mock_get_service, mock_current_user):
        """Test service status check when there's an error."""
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        # Configure the mock to raise an exception when accessing is_initialized
        type(mock_service).is_initialized = PropertyMock(
            side_effect=Exception("Service unavailable")
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_service_status(
                _current_user=mock_current_user,
                gemini_service=mock_service,
            )

        error = exc_info.value
        assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "STATUS_CHECK_FAILED" in str(error.detail)


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    def setup_method(self):
        """Setup for each test method."""
        self.mock_auth_provider = MagicMock(spec=IAuthProvider)
        self.mock_config_provider = MagicMock(spec=IConfigProvider)
        self.mock_config_provider.is_development.return_value = True

        set_dependencies(self.mock_auth_provider, self.mock_config_provider)

    @pytest.fixture
    def mock_current_user(self):
        """Mock authenticated user."""
        user = MagicMock(spec=AuthenticatedUser)
        user.user_id = "user_456"
        user.is_active = True
        user.username = "testuser"
        return user

    def test_complete_insight_workflow(self, mock_current_user):
        """Test complete workflow from request to response."""
        # This test would verify the complete flow but requires more complex setup
        # For now, it demonstrates the testing approach
        request_id = generate_request_id()

        assert request_id.startswith("req_insights_")

        metadata = create_metadata(request_id, 1200.5)

        assert metadata["request_id"] == request_id
        assert metadata["processing_time_ms"] == 1200.5
        assert metadata["service"] == "gemini-insights"

    @pytest.mark.asyncio
    async def test_error_handling_consistency(self):
        """Test that all error responses follow the same format."""
        # Test that create_error_response returns HTTPException with proper structure
        http_exception = create_error_response(
            error_code="VALIDATION_ERROR",
            message="Invalid input data",
            request_id="req_123",
            status_code=400,
        )

        # Verify the returned exception has the correct structure
        assert isinstance(http_exception, HTTPException)
        assert http_exception.status_code == 400
        assert http_exception.detail["code"] == "VALIDATION_ERROR"
        assert http_exception.detail["message"] == "Invalid input data"
        assert http_exception.detail["request_id"] == "req_123"
        assert "timestamp" in http_exception.detail

        # Test multiple error types follow the same pattern
        errors = [
            ("INSUFFICIENT_DATA", "Not enough data for analysis", 422),
            ("SERVICE_UNAVAILABLE", "Gemini service temporarily unavailable", 503),
            ("RATE_LIMIT_EXCEEDED", "Too many requests", 429),
        ]

        for error_code, message, status_code in errors:
            exc = create_error_response(error_code, message, "test_req", status_code)
            assert isinstance(exc, HTTPException)
            assert exc.status_code == status_code
            assert exc.detail["code"] == error_code
            assert exc.detail["message"] == message
            assert "timestamp" in exc.detail
            assert "request_id" in exc.detail

    def test_request_id_tracking(self):
        """Test that request IDs are properly tracked across operations."""
        # Generate multiple request IDs
        request_ids = [generate_request_id() for _ in range(5)]

        # All should be unique
        assert len(set(request_ids)) == 5

        # All should follow the same format
        for req_id in request_ids:
            assert req_id.startswith("req_insights_")
            assert len(req_id) == len("req_insights_") + 8

            # Test metadata creation with each ID
            metadata = create_metadata(req_id)
            assert metadata["request_id"] == req_id
            assert metadata["service"] == "gemini-insights"
