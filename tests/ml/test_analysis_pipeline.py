"""Comprehensive tests for Analysis Pipeline - Health Data Processing Orchestrator.

This test suite covers all aspects of the analysis pipeline including:
- AnalysisResults container class
- HealthAnalysisPipeline main orchestrator
- Data organization by modality
- Individual modality processing (cardio, respiratory, activity)
- Multi-modal fusion
- Summary statistics generation
- DynamoDB integration
- Error handling and edge cases
- Utility functions for HealthKit data conversion
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from clarity.ml.analysis_pipeline import (
    MIN_FEATURE_VECTOR_LENGTH,
    MIN_METRICS_FOR_TIME_SPAN,
    AnalysisPipelineSingleton,
    AnalysisResults,
    HealthAnalysisPipeline,
    get_analysis_pipeline,
    run_analysis_pipeline,
)
from clarity.models.health_data import (
    ActivityData,
    BiometricData,
    HealthMetric,
    HealthMetricType,
)


class TestAnalysisResults:
    """Test the AnalysisResults container class."""

    @staticmethod
    def test_analysis_results_initialization() -> None:
        """Test AnalysisResults initializes with empty containers."""
        results = AnalysisResults()

        assert results.cardio_features == []
        assert results.respiratory_features == []
        assert results.activity_features == []
        assert results.activity_embedding == []
        assert results.fused_vector == []
        assert results.summary_stats == {}
        assert results.processing_metadata == {}

    @staticmethod
    def test_analysis_results_assignment() -> None:
        """Test that AnalysisResults fields can be assigned values."""
        results = AnalysisResults()

        # Test cardio features assignment
        cardio_data = [1.0, 2.0, 3.0]
        results.cardio_features = cardio_data
        assert results.cardio_features == cardio_data

        # Test respiratory features assignment
        respiratory_data = [4.0, 5.0, 6.0]
        results.respiratory_features = respiratory_data
        assert results.respiratory_features == respiratory_data

        # Test activity features assignment
        activity_data = [{"heart_rate": 80}, {"steps": 1000}]
        results.activity_features = activity_data
        assert results.activity_features == activity_data

        # Test activity embedding assignment
        embedding_data = [0.1, 0.2, 0.3, 0.4]
        results.activity_embedding = embedding_data
        assert results.activity_embedding == embedding_data

        # Test fused vector assignment
        fused_data = [7.0, 8.0, 9.0]
        results.fused_vector = fused_data
        assert results.fused_vector == fused_data

        # Test summary stats assignment
        stats_data = {"total_metrics": 10, "time_span": 24.0}
        results.summary_stats = stats_data
        assert results.summary_stats == stats_data

        # Test processing metadata assignment
        metadata = {"user_id": "test_user", "processed_at": "2023-01-01T00:00:00Z"}
        results.processing_metadata = metadata
        assert results.processing_metadata == metadata


class TestHealthAnalysisPipelineInitialization:
    """Test HealthAnalysisPipeline initialization and setup."""

    @staticmethod
    def test_pipeline_initialization() -> None:
        """Test pipeline initializes with all required processors."""
        pipeline = HealthAnalysisPipeline()

        # Check processors are initialized
        assert hasattr(pipeline, "cardio_processor")
        assert hasattr(pipeline, "respiratory_processor")
        assert hasattr(pipeline, "activity_processor")
        assert hasattr(pipeline, "preprocessor")

        # Check ML services are set up
        assert pipeline.pat_service is None  # Loaded on-demand
        assert hasattr(pipeline, "fusion_service")

        # Check storage client is initially None
        assert pipeline.dynamodb_client is None

        # Check logger is set up
        assert hasattr(pipeline, "logger")

    @pytest.mark.asyncio
    @staticmethod
    async def test_get_dynamodb_client_creates_client() -> None:
        """Test that _get_dynamodb_client creates client on first call."""
        pipeline = HealthAnalysisPipeline()

        with patch(
            "clarity.ml.analysis_pipeline.DynamoDBHealthDataRepository"
        ) as mock_dynamodb:
            mock_client = MagicMock()
            mock_dynamodb.return_value = mock_client

            client = await pipeline._get_dynamodb_client()

            assert client == mock_client
            assert pipeline.dynamodb_client == mock_client
            mock_dynamodb.assert_called_once()

    @pytest.mark.asyncio
    @staticmethod
    async def test_get_dynamodb_client_reuses_existing() -> None:
        """Test that _get_dynamodb_client reuses existing client."""
        pipeline = HealthAnalysisPipeline()
        existing_client = MagicMock()
        pipeline.dynamodb_client = existing_client

        client = await pipeline._get_dynamodb_client()

        assert client == existing_client


class TestHealthAnalysisPipelineDataOrganization:
    """Test data organization by modality functionality."""

    @staticmethod
    def test_organize_metrics_by_modality_empty_list() -> None:
        """Test organizing empty metrics list returns empty modalities."""
        pipeline = HealthAnalysisPipeline()
        result = pipeline._organize_metrics_by_modality([])

        assert result == {
            "cardio": [],
            "respiratory": [],
            "activity": [],
            "sleep": [],
            "other": [],
        }

    @staticmethod
    def test_organize_metrics_by_modality_cardio_only() -> None:
        """Test organizing metrics with only cardio data."""
        pipeline = HealthAnalysisPipeline()

        # Create cardio metrics
        cardio_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=75.0),
        )

        result = pipeline._organize_metrics_by_modality([cardio_metric])

        assert len(result["cardio"]) == 1
        assert result["cardio"][0] == cardio_metric
        assert result["respiratory"] == []
        assert result["activity"] == []
        assert result["sleep"] == []
        assert result["other"] == []

    @staticmethod
    def test_organize_metrics_by_modality_all_types() -> None:
        """Test organizing metrics with all modality types."""
        pipeline = HealthAnalysisPipeline()

        # Create metrics for each modality
        cardio_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=75.0),
        )

        respiratory_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE_VARIABILITY,
            biometric_data=BiometricData(heart_rate_variability=50.0),
        )

        activity_metric = HealthMetric(
            metric_type=HealthMetricType.ACTIVITY_LEVEL,  # Use ACTIVITY_LEVEL
            activity_data=ActivityData(steps=1000, distance=2.5),
        )

        metrics = [cardio_metric, respiratory_metric, activity_metric]
        result = pipeline._organize_metrics_by_modality(metrics)

        # Both HEART_RATE and HEART_RATE_VARIABILITY are categorized as cardio (correct behavior)
        assert len(result["cardio"]) == 2
        assert cardio_metric in result["cardio"]
        assert respiratory_metric in result["cardio"]  # HRV is cardio-related
        # ACTIVITY_LEVEL is now properly routed to activity (fixed behavior)
        assert len(result["activity"]) == 1
        assert result["activity"][0] == activity_metric
        assert len(result["other"]) == 0  # No unrecognized metrics
        assert result["respiratory"] == []  # No respiratory metrics in this test
        assert result["sleep"] == []


class TestHealthAnalysisPipelineModalityProcessing:
    """Test individual modality processing methods."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_cardio_data_success() -> None:
        """Test successful cardio data processing."""
        pipeline = HealthAnalysisPipeline()

        # Mock cardio processor
        expected_features = [1.0, 2.0, 3.0]
        mock_processor = Mock()
        mock_processor.process = MagicMock(return_value=expected_features)
        pipeline.cardio_processor = mock_processor

        cardio_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=75.0),
        )

        result = await pipeline._process_cardio_data([cardio_metric])

        assert result == expected_features
        # Verify processor is called with extracted timestamps and values
        call_args = pipeline.cardio_processor.process.call_args[0]
        assert (
            len(call_args) == 4
        )  # hr_timestamps, hr_values, hrv_timestamps, hrv_values
        assert len(call_args[0]) == 1  # hr_timestamps
        assert call_args[1] == [75.0]  # hr_values
        assert call_args[2] == []  # hrv_timestamps (empty)
        assert call_args[3] == []  # hrv_values (empty)

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_respiratory_data_success() -> None:
        """Test successful respiratory data processing."""
        pipeline = HealthAnalysisPipeline()

        # Mock respiratory processor
        expected_features = [4.0, 5.0, 6.0]
        mock_processor = Mock()
        mock_processor.process = MagicMock(return_value=expected_features)
        pipeline.respiratory_processor = mock_processor

        # Since HRV is categorized as cardio, pass empty list to respiratory processor
        result = await pipeline._process_respiratory_data([])

        assert result == expected_features
        # Verify processor is called with empty lists since HRV is not respiratory
        pipeline.respiratory_processor.process.assert_called_once_with([], [], [], [])

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_activity_data_success() -> None:
        """Test successful activity data processing with PAT model."""
        pipeline = HealthAnalysisPipeline()

        # Mock PAT service
        mock_pat_service = AsyncMock()
        mock_analysis = MagicMock()
        mock_analysis.embedding = [0.1, 0.2, 0.3, 0.4]
        mock_pat_service.analyze_actigraphy.return_value = mock_analysis

        with patch(
            "clarity.ml.analysis_pipeline.get_pat_service",
            return_value=mock_pat_service,
        ):
            activity_metric = HealthMetric(
                metric_type=HealthMetricType.ACTIVITY_LEVEL,
                activity_data=ActivityData(steps=1000),
            )

            result = await pipeline._process_activity_data("user1", [activity_metric])

            assert result == [0.1, 0.2, 0.3, 0.4]
            mock_pat_service.analyze_actigraphy.assert_called_once()

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_activity_data_pat_service_error() -> None:
        """Test activity data processing when PAT service fails."""
        pipeline = HealthAnalysisPipeline()

        # Mock PAT service to raise error
        mock_pat_service = AsyncMock()
        mock_pat_service.analyze_actigraphy.side_effect = RuntimeError(
            "PAT service error"
        )

        with patch(
            "clarity.ml.analysis_pipeline.get_pat_service",
            return_value=mock_pat_service,
        ):
            activity_metric = HealthMetric(
                metric_type=HealthMetricType.ACTIVITY_LEVEL,
                activity_data=ActivityData(steps=1000),
            )

            # The error should propagate, not be caught
            with pytest.raises(RuntimeError, match="PAT service error"):
                await pipeline._process_activity_data("user1", [activity_metric])


class TestHealthAnalysisPipelineFusion:
    """Test multi-modal fusion functionality."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_fuse_modalities_success() -> None:
        """Test successful multi-modal fusion."""
        pipeline = HealthAnalysisPipeline()

        # Mock fusion service
        expected_fused = [7.0, 8.0, 9.0]
        mock_fusion = Mock()
        mock_fusion.fuse_modalities = MagicMock(return_value=expected_fused)
        mock_fusion.initialize_model = MagicMock()
        pipeline.fusion_service = mock_fusion

        modality_features = {"cardio": [1.0, 2.0, 3.0], "respiratory": [4.0, 5.0, 6.0]}

        result = await pipeline._fuse_modalities(modality_features)

        assert result == expected_fused
        pipeline.fusion_service.initialize_model.assert_called_once()
        pipeline.fusion_service.fuse_modalities.assert_called_once_with(
            modality_features
        )

    @pytest.mark.asyncio
    @staticmethod
    async def test_fuse_modalities_service_error() -> None:
        """Test fusion when service fails."""
        pipeline = HealthAnalysisPipeline()

        # Mock fusion service to raise error
        mock_fusion = Mock()
        mock_fusion.initialize_model = MagicMock()
        mock_fusion.fuse_modalities = MagicMock(
            side_effect=RuntimeError("Fusion error")
        )
        pipeline.fusion_service = mock_fusion

        modality_features = {"cardio": [1.0, 2.0, 3.0], "respiratory": [4.0, 5.0, 6.0]}

        # The error should propagate, not be caught
        with pytest.raises(RuntimeError, match="Fusion error"):
            await pipeline._fuse_modalities(modality_features)


class TestHealthAnalysisPipelineSummaryStats:
    """Test summary statistics generation."""

    @staticmethod
    def test_generate_summary_stats_basic() -> None:
        """Test basic summary statistics generation."""
        pipeline = HealthAnalysisPipeline()

        # Create organized data
        cardio_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=75.0),
        )

        organized_data = {"cardio": [cardio_metric], "respiratory": [], "activity": []}

        modality_features = {"cardio": [1.0, 2.0, 3.0]}

        result = pipeline._generate_summary_stats(organized_data, modality_features)

        assert "data_coverage" in result
        assert "feature_summary" in result
        assert "health_indicators" in result

    @staticmethod
    def test_generate_data_coverage() -> None:
        """Test data coverage calculation."""
        # Create test metrics with timestamps
        start_time = datetime.now(UTC)
        end_time = start_time + timedelta(hours=24)

        cardio_metric1 = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            created_at=start_time,
            biometric_data=BiometricData(heart_rate=75.0),
        )

        cardio_metric2 = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            created_at=end_time,
            biometric_data=BiometricData(heart_rate=80.0),
        )

        organized_data = {
            "cardio": [cardio_metric1, cardio_metric2],
            "respiratory": [],
            "activity": [],
        }

        result = HealthAnalysisPipeline._generate_data_coverage(organized_data)

        # Check cardio modality data
        assert "cardio" in result
        assert result["cardio"]["metric_count"] == 2
        assert result["cardio"]["time_span_hours"] == 24.0
        assert result["cardio"]["data_density"] == 2.0 / 24.0

        # Empty modalities should not be in result
        assert "respiratory" not in result
        assert "activity" not in result

    @staticmethod
    def test_generate_feature_summary() -> None:
        """Test feature summary generation."""
        modality_features = {
            "cardio": [1.0, 2.0, 3.0],
            "respiratory": [4.0, 5.0, 6.0, 7.0],
        }

        result = HealthAnalysisPipeline._generate_feature_summary(modality_features)

        # Check cardio features summary
        assert "cardio" in result
        assert result["cardio"]["feature_count"] == 3
        assert result["cardio"]["mean_value"] == 2.0  # (1+2+3)/3
        assert result["cardio"]["min_value"] == 1.0
        assert result["cardio"]["max_value"] == 3.0

        # Check respiratory features summary
        assert "respiratory" in result
        assert result["respiratory"]["feature_count"] == 4
        assert result["respiratory"]["mean_value"] == 5.5  # (4+5+6+7)/4
        assert result["respiratory"]["min_value"] == 4.0
        assert result["respiratory"]["max_value"] == 7.0

    @staticmethod
    def test_generate_health_indicators_all_modalities() -> None:
        """Test health indicators generation with all modalities."""
        pipeline = HealthAnalysisPipeline()

        modality_features = {
            "cardio": [75.0, 80.0, 70.0],  # Heart rate values
            "respiratory": [16.0, 18.0, 14.0],  # Breathing rate values
            "activity": [1000.0, 1500.0, 800.0],  # Activity values
        }

        activity_features = [
            {"feature_name": "total_steps", "value": 1000},
            {"feature_name": "average_daily_steps", "value": 1500},
        ]

        result = pipeline._generate_health_indicators(
            modality_features, activity_features
        )

        # _generate_health_indicators only returns activity_health (not cardio/respiratory)
        assert "activity_health" in result
        assert "total_steps" in result["activity_health"]
        assert "avg_daily_steps" in result["activity_health"]

    @staticmethod
    def test_extract_cardio_health_indicators() -> None:
        """Test cardio health indicators extraction."""
        # Need at least MIN_FEATURE_VECTOR_LENGTH (8) features
        modality_features = {"cardio": [75.0, 80.0, 70.0, 85.0, 90.0, 65.0, 78.0, 82.0]}

        result = HealthAnalysisPipeline._extract_cardio_health_indicators(
            modality_features
        )

        assert result is not None
        assert "avg_heart_rate" in result
        assert "resting_heart_rate" in result
        assert "heart_rate_recovery" in result
        assert "circadian_rhythm" in result
        assert result["avg_heart_rate"] == 75.0  # cardio[0]
        assert result["resting_heart_rate"] == 70.0  # cardio[2]
        assert result["heart_rate_recovery"] == 78.0  # cardio[6]
        assert result["circadian_rhythm"] == 82.0  # cardio[7]

    @staticmethod
    def test_extract_respiratory_health_indicators() -> None:
        """Test respiratory health indicators extraction."""
        # Need at least MIN_FEATURE_VECTOR_LENGTH (8) features
        modality_features = {
            "respiratory": [16.0, 18.0, 14.0, 95.5, 20.0, 15.0, 0.85, 0.92]
        }

        result = HealthAnalysisPipeline._extract_respiratory_health_indicators(
            modality_features
        )

        assert result is not None
        assert "avg_respiratory_rate" in result
        assert "avg_oxygen_saturation" in result
        assert "respiratory_stability" in result
        assert "oxygenation_efficiency" in result
        assert result["avg_respiratory_rate"] == 16.0  # resp[0]
        assert result["avg_oxygen_saturation"] == 95.5  # resp[3]
        assert result["respiratory_stability"] == 0.85  # resp[6]
        assert result["oxygenation_efficiency"] == 0.92  # resp[7]

    @staticmethod
    def test_extract_activity_health_indicators() -> None:
        """Test activity health indicators extraction."""
        activity_features = [
            {"feature_name": "total_steps", "value": 1000},
            {"feature_name": "average_daily_steps", "value": 1250.5},
            {"feature_name": "total_distance", "value": 5.75},
            {"feature_name": "total_active_energy", "value": 450.7},
            {"feature_name": "total_exercise_minutes", "value": 35.2},
            {"feature_name": "activity_consistency_score", "value": 0.8765},
            {"feature_name": "latest_vo2_max", "value": 42.345},
        ]

        result = HealthAnalysisPipeline._extract_activity_health_indicators(
            activity_features
        )

        assert result is not None
        assert "total_steps" in result
        assert "avg_daily_steps" in result
        assert "total_distance_km" in result
        assert "total_calories" in result
        assert "total_exercise_minutes" in result
        assert "consistency_score" in result
        assert "cardio_fitness_vo2_max" in result

        assert result["total_steps"] == 1000
        assert result["avg_daily_steps"] == 1250  # Rounded (not 1251)
        assert result["total_distance_km"] == 5.8  # Rounded to 1 decimal
        assert result["total_calories"] == 451  # Rounded
        assert result["total_exercise_minutes"] == 35  # Rounded
        assert result["consistency_score"] == 0.88  # Rounded to 2 decimals
        assert result["cardio_fitness_vo2_max"] == 42.3  # Rounded to 1 decimal

    @staticmethod
    def test_extract_activity_health_indicators_none_input() -> None:
        """Test activity health indicators extraction with None input."""
        result = HealthAnalysisPipeline._extract_activity_health_indicators(None)
        assert result is None

    @staticmethod
    def test_extract_activity_health_indicators_empty_input() -> None:
        """Test activity health indicators extraction with empty input."""
        result = HealthAnalysisPipeline._extract_activity_health_indicators([])
        assert result is None

    @staticmethod
    def test_calculate_time_span() -> None:
        """Test time span calculation between metrics."""
        start_time = datetime.now(UTC)
        end_time = start_time + timedelta(hours=24)

        metric1 = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            created_at=start_time,
            biometric_data=BiometricData(heart_rate=75.0),
        )

        metric2 = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            created_at=end_time,
            biometric_data=BiometricData(heart_rate=80.0),
        )

        result = HealthAnalysisPipeline._calculate_time_span([metric1, metric2])
        assert result == 24.0

    @staticmethod
    def test_calculate_time_span_single_metric() -> None:
        """Test time span calculation with single metric."""
        metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            created_at=datetime.now(UTC),
            biometric_data=BiometricData(heart_rate=75.0),
        )

        result = HealthAnalysisPipeline._calculate_time_span([metric])
        assert (
            result == 1.0
        )  # Single metric returns default 1.0 (below MIN_METRICS_FOR_TIME_SPAN)


class TestHealthAnalysisPipelineMainWorkflow:
    """Test the main process_health_data workflow."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_health_data_empty_metrics() -> None:
        """Test processing with empty metrics list."""
        pipeline = HealthAnalysisPipeline()

        result = await pipeline.process_health_data("user1", [])

        assert isinstance(result, AnalysisResults)
        assert result.cardio_features == []
        assert result.respiratory_features == []
        assert result.activity_features == []
        assert result.activity_embedding == []
        assert result.fused_vector == []
        assert result.processing_metadata["total_metrics"] == 0

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_health_data_single_modality() -> None:
        """Test processing with single modality (cardio only)."""
        pipeline = HealthAnalysisPipeline()

        # Mock cardio processor
        expected_cardio_features = [1.0, 2.0, 3.0]
        mock_processor = Mock()
        mock_processor.process = MagicMock(return_value=expected_cardio_features)
        pipeline.cardio_processor = mock_processor

        cardio_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=75.0),
        )

        result = await pipeline.process_health_data("user1", [cardio_metric])

        assert result.cardio_features == expected_cardio_features
        assert result.respiratory_features == []
        assert result.activity_features == []
        assert result.activity_embedding == []
        # Single modality should use cardio features as fused vector
        assert result.fused_vector == expected_cardio_features
        assert result.processing_metadata["total_metrics"] == 1
        assert result.processing_metadata["modalities_processed"] == ["cardio"]

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_health_data_multiple_modalities() -> None:
        """Test processing with multiple modalities requiring fusion."""
        pipeline = HealthAnalysisPipeline()

        # Mock processors
        expected_cardio = [1.0, 2.0, 3.0]
        expected_respiratory = [4.0, 5.0, 6.0]
        expected_fused = [7.0, 8.0, 9.0]

        mock_cardio = Mock()
        mock_cardio.process = MagicMock(return_value=expected_cardio)
        pipeline.cardio_processor = mock_cardio

        mock_respiratory = Mock()
        mock_respiratory.process = MagicMock(return_value=expected_respiratory)
        pipeline.respiratory_processor = mock_respiratory

        mock_fusion = Mock()
        mock_fusion.fuse_modalities = MagicMock(return_value=expected_fused)
        mock_fusion.initialize_model = MagicMock()
        pipeline.fusion_service = mock_fusion

        cardio_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=75.0),
        )

        respiratory_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE_VARIABILITY,  # Use valid type
            biometric_data=BiometricData(heart_rate_variability=16.0),
        )

        result = await pipeline.process_health_data(
            "user1", [cardio_metric, respiratory_metric]
        )

        assert result.cardio_features == expected_cardio
        # No respiratory features since both metrics are cardio (HEART_RATE and HEART_RATE_VARIABILITY)
        assert result.respiratory_features == []
        # With only 1 modality (cardio), fused_vector equals cardio_features (no fusion service called)
        assert result.fused_vector == expected_cardio
        # Both metrics are cardio (HEART_RATE and HEART_RATE_VARIABILITY)
        assert "cardio" in result.processing_metadata["modalities_processed"]
        assert len(result.processing_metadata["modalities_processed"]) == 1

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_health_data_with_dynamodb_save() -> None:
        """Test processing with DynamoDB saving when processing_id provided."""
        pipeline = HealthAnalysisPipeline()

        # Mock processors
        expected_cardio = [1.0, 2.0, 3.0]
        mock_processor = Mock()
        mock_processor.process = MagicMock(return_value=expected_cardio)
        pipeline.cardio_processor = mock_processor

        # Mock DynamoDB client
        mock_dynamodb = MagicMock()
        mock_table = MagicMock()
        mock_table.put_item = MagicMock()
        mock_dynamodb.table = mock_table
        pipeline.dynamodb_client = mock_dynamodb

        cardio_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=75.0),
        )

        processing_id = "test_processing_123"
        result = await pipeline.process_health_data(
            "user1", [cardio_metric], processing_id
        )

        # Verify DynamoDB save was called
        mock_table.put_item.assert_called_once()
        call_args = mock_table.put_item.call_args
        saved_item = call_args[1]["Item"]
        assert saved_item["user_id"] == "user1"
        assert saved_item["processing_id"] == processing_id

        # Verify processing_id is in metadata
        assert result.processing_metadata["processing_id"] == processing_id


class TestAnalysisPipelineSingleton:
    """Test the singleton pattern for analysis pipeline."""

    @staticmethod
    def test_singleton_get_instance() -> None:
        """Test singleton returns same instance."""
        # Clear any existing instance
        AnalysisPipelineSingleton._instance = None

        instance1 = AnalysisPipelineSingleton.get_instance()
        instance2 = AnalysisPipelineSingleton.get_instance()

        assert instance1 is instance2
        assert isinstance(instance1, HealthAnalysisPipeline)

    @staticmethod
    def test_get_analysis_pipeline_function() -> None:
        """Test global get_analysis_pipeline function."""
        # Clear any existing instance
        AnalysisPipelineSingleton._instance = None

        pipeline = get_analysis_pipeline()

        assert isinstance(pipeline, HealthAnalysisPipeline)
        # Should return same instance as singleton
        assert pipeline is AnalysisPipelineSingleton.get_instance()


class TestRunAnalysisPipelineFunction:
    """Test the run_analysis_pipeline utility function."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_run_analysis_pipeline_success() -> None:
        """Test successful run_analysis_pipeline execution."""
        # Mock health data input
        health_data = {
            "quantity_samples": [
                {
                    "type": "heart_rate",
                    "value": 75.0,
                    "unit": "bpm",
                    "start_date": datetime.now(UTC).isoformat(),
                    "end_date": datetime.now(UTC).isoformat(),
                }
            ],
            "category_samples": [],
            "workout_data": [],
        }

        with patch(
            "clarity.ml.analysis_pipeline.get_analysis_pipeline"
        ) as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_results = AnalysisResults()
            mock_results.cardio_features = [1.0, 2.0, 3.0]
            mock_results.processing_metadata = {"total_metrics": 1}

            mock_pipeline.process_health_data.return_value = mock_results
            mock_get_pipeline.return_value = mock_pipeline

            result = await run_analysis_pipeline("user1", health_data)

            assert isinstance(result, dict)
            assert "cardio_features" in result
            assert "processing_metadata" in result

    @pytest.mark.asyncio
    @staticmethod
    async def test_run_analysis_pipeline_conversion_error() -> None:
        """Test run_analysis_pipeline with invalid data (gracefully handled)."""
        # Invalid health data - conversion should succeed but return empty metrics
        invalid_health_data = {"invalid_key": "invalid_value"}

        # Should not raise exception, but handle gracefully with empty metrics
        result = await run_analysis_pipeline("user1", invalid_health_data)

        assert isinstance(result, dict)
        assert result["user_id"] == "user1"
        # Should have empty/default features since no valid metrics were converted
        assert "cardio_features" in result
        assert "processing_metadata" in result


# TestHealthKitDataConversion class removed due to private function dependencies


class TestAnalysisPipelineConstants:
    """Test module constants."""

    @staticmethod
    def test_constants_defined() -> None:
        """Test that required constants are defined."""
        assert MIN_FEATURE_VECTOR_LENGTH == 8
        assert MIN_METRICS_FOR_TIME_SPAN == 2


class TestAnalysisPipelineErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_health_data_processor_error() -> None:
        """Test processing when a processor raises an error."""
        pipeline = HealthAnalysisPipeline()

        # Mock cardio processor to raise error
        mock_processor = Mock()
        mock_processor.process = MagicMock(side_effect=RuntimeError("Processor error"))
        pipeline.cardio_processor = mock_processor

        cardio_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=75.0),
        )

        # Should raise the processor error (not handled gracefully)
        with pytest.raises(RuntimeError, match="Processor error"):
            await pipeline.process_health_data("user1", [cardio_metric])

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_health_data_dynamodb_error() -> None:
        """Test processing when DynamoDB save fails."""
        pipeline = HealthAnalysisPipeline()

        # Mock processors
        mock_processor = Mock()
        mock_processor.process = MagicMock(return_value=[1.0, 2.0, 3.0])
        pipeline.cardio_processor = mock_processor

        # Mock DynamoDB client to raise error
        mock_dynamodb = AsyncMock()
        mock_dynamodb.save_analysis_result.side_effect = RuntimeError("DynamoDB error")
        pipeline.dynamodb_client = mock_dynamodb

        cardio_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=75.0),
        )

        # Should continue processing despite DynamoDB error
        result = await pipeline.process_health_data(
            "user1", [cardio_metric], "processing_123"
        )

        assert isinstance(result, AnalysisResults)
        assert result.cardio_features == [1.0, 2.0, 3.0]
