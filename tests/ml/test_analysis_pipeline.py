"""Comprehensive tests for Analysis Pipeline - Health Data Processing Orchestrator.

This test suite covers all aspects of the analysis pipeline including:
- AnalysisResults container class
- HealthAnalysisPipeline main orchestrator
- Data organization by modality
- Individual modality processing (cardio, respiratory, activity)
- Multi-modal fusion
- Summary statistics generation
- Firestore integration
- Error handling and edge cases
- Utility functions for HealthKit data conversion
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from clarity.ml.analysis_pipeline import (
    MIN_FEATURE_VECTOR_LENGTH,
    MIN_METRICS_FOR_TIME_SPAN,
    AnalysisPipelineSingleton,
    AnalysisResults,
    HealthAnalysisPipeline,
    _convert_raw_data_to_metrics,
    _create_biometric_data_from_sample,
    _create_sleep_metric_from_sample,
    _get_healthkit_type_mapping,
    _process_category_samples,
    _process_quantity_samples,
    _process_workout_data,
    get_analysis_pipeline,
    run_analysis_pipeline,
)
from clarity.models.health_data import (
    ActivityData,
    BiometricData,
    HealthMetric,
    HealthMetricType,
    SleepData,
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
        assert pipeline.firestore_client is None

        # Check logger is set up
        assert hasattr(pipeline, "logger")

    @pytest.mark.asyncio
    @staticmethod
    async def test_get_firestore_client_creates_client() -> None:
        """Test that _get_firestore_client creates client on first call."""
        pipeline = HealthAnalysisPipeline()

        with patch("clarity.ml.analysis_pipeline.FirestoreClient") as mock_firestore:
            mock_client = MagicMock()
            mock_firestore.return_value = mock_client

            client = await pipeline._get_firestore_client()

            assert client == mock_client
            assert pipeline.firestore_client == mock_client
            mock_firestore.assert_called_once()

    @pytest.mark.asyncio
    @staticmethod
    async def test_get_firestore_client_reuses_existing() -> None:
        """Test that _get_firestore_client reuses existing client."""
        pipeline = HealthAnalysisPipeline()
        existing_client = MagicMock()
        pipeline.firestore_client = existing_client

        client = await pipeline._get_firestore_client()

        assert client == existing_client


class TestHealthAnalysisPipelineDataOrganization:
    """Test data organization by modality functionality."""

    @staticmethod
    def test_organize_metrics_by_modality_empty_list() -> None:
        """Test organizing empty metrics list returns empty modalities."""
        pipeline = HealthAnalysisPipeline()
        result = pipeline._organize_metrics_by_modality([])

        assert result == {"cardio": [], "respiratory": [], "activity": [], "sleep": [], "other": []}

    @staticmethod
    def test_organize_metrics_by_modality_cardio_only() -> None:
        """Test organizing metrics with only cardio data."""
        pipeline = HealthAnalysisPipeline()

        # Create cardio metrics
        cardio_metric = HealthMetric(
            metric_type=HealthMetricType.HEART_RATE,
            biometric_data=BiometricData(heart_rate=75.0)
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
            id="1",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=datetime.now(UTC),
            data=BiometricData(value=75.0, unit="bpm")
        )

        respiratory_metric = HealthMetric(
            id="2",
            user_id="user1",
            metric_type=HealthMetricType.RESPIRATORY_RATE,
            timestamp=datetime.now(UTC),
            data=BiometricData(value=16.0, unit="breaths/min")
        )

        activity_metric = HealthMetric(
            id="3",
            user_id="user1",
            metric_type=HealthMetricType.STEPS,
            timestamp=datetime.now(UTC),
            data=ActivityData(value=1000, unit="count")
        )

        metrics = [cardio_metric, respiratory_metric, activity_metric]
        result = pipeline._organize_metrics_by_modality(metrics)

        assert len(result["cardio"]) == 1
        assert result["cardio"][0] == cardio_metric
        assert len(result["respiratory"]) == 1
        assert result["respiratory"][0] == respiratory_metric
        assert len(result["activity"]) == 1
        assert result["activity"][0] == activity_metric


class TestHealthAnalysisPipelineModalityProcessing:
    """Test individual modality processing methods."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_cardio_data_success() -> None:
        """Test successful cardio data processing."""
        pipeline = HealthAnalysisPipeline()

        # Mock cardio processor
        expected_features = [1.0, 2.0, 3.0]
        pipeline.cardio_processor.process = MagicMock(return_value=expected_features)

        cardio_metric = HealthMetric(
            id="1",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=datetime.now(UTC),
            data=BiometricData(value=75.0, unit="bpm")
        )

        result = await pipeline._process_cardio_data([cardio_metric])

        assert result == expected_features
        pipeline.cardio_processor.process.assert_called_once_with([cardio_metric])

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_respiratory_data_success() -> None:
        """Test successful respiratory data processing."""
        pipeline = HealthAnalysisPipeline()

        # Mock respiratory processor
        expected_features = [4.0, 5.0, 6.0]
        pipeline.respiratory_processor.process = MagicMock(return_value=expected_features)

        respiratory_metric = HealthMetric(
            id="1",
            user_id="user1",
            metric_type=HealthMetricType.RESPIRATORY_RATE,
            timestamp=datetime.now(UTC),
            data=BiometricData(value=16.0, unit="breaths/min")
        )

        result = await pipeline._process_respiratory_data([respiratory_metric])

        assert result == expected_features
        pipeline.respiratory_processor.process.assert_called_once_with([respiratory_metric])

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

        with patch("clarity.ml.analysis_pipeline.get_pat_service", return_value=mock_pat_service):
            activity_metric = HealthMetric(
                id="1",
                user_id="user1",
                metric_type=HealthMetricType.STEPS,
                timestamp=datetime.now(UTC),
                data=ActivityData(value=1000, unit="count")
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
        mock_pat_service.analyze_actigraphy.side_effect = Exception("PAT service error")

        with patch("clarity.ml.analysis_pipeline.get_pat_service", return_value=mock_pat_service):
            activity_metric = HealthMetric(
                id="1",
                user_id="user1",
                metric_type=HealthMetricType.STEPS,
                timestamp=datetime.now(UTC),
                data=ActivityData(value=1000, unit="count")
            )

            result = await pipeline._process_activity_data("user1", [activity_metric])

            # Should return empty list on error
            assert result == []


class TestHealthAnalysisPipelineFusion:
    """Test multi-modal fusion functionality."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_fuse_modalities_success() -> None:
        """Test successful multi-modal fusion."""
        pipeline = HealthAnalysisPipeline()

        # Mock fusion service
        expected_fused = [7.0, 8.0, 9.0]
        pipeline.fusion_service.fuse_embeddings = AsyncMock(return_value=expected_fused)

        modality_features = {
            "cardio": [1.0, 2.0, 3.0],
            "respiratory": [4.0, 5.0, 6.0]
        }

        result = await pipeline._fuse_modalities(modality_features)

        assert result == expected_fused
        pipeline.fusion_service.fuse_embeddings.assert_called_once_with(modality_features)

    @pytest.mark.asyncio
    @staticmethod
    async def test_fuse_modalities_service_error() -> None:
        """Test fusion when service fails."""
        pipeline = HealthAnalysisPipeline()

        # Mock fusion service to raise error
        pipeline.fusion_service.fuse_embeddings = AsyncMock(side_effect=Exception("Fusion error"))

        modality_features = {
            "cardio": [1.0, 2.0, 3.0],
            "respiratory": [4.0, 5.0, 6.0]
        }

        result = await pipeline._fuse_modalities(modality_features)

        # Should return empty list on error
        assert result == []


class TestHealthAnalysisPipelineSummaryStats:
    """Test summary statistics generation."""

    @staticmethod
    def test_generate_summary_stats_basic() -> None:
        """Test basic summary statistics generation."""
        pipeline = HealthAnalysisPipeline()

        # Create organized data
        cardio_metric = HealthMetric(
            id="1",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=datetime.now(UTC),
            data=BiometricData(value=75.0, unit="bpm")
        )

        organized_data = {
            "cardio": [cardio_metric],
            "respiratory": [],
            "activity": []
        }

        modality_features = {
            "cardio": [1.0, 2.0, 3.0]
        }

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
            id="1",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=start_time,
            data=BiometricData(value=75.0, unit="bpm")
        )

        cardio_metric2 = HealthMetric(
            id="2",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=end_time,
            data=BiometricData(value=80.0, unit="bpm")
        )

        organized_data = {
            "cardio": [cardio_metric1, cardio_metric2],
            "respiratory": [],
            "activity": []
        }

        result = HealthAnalysisPipeline._generate_data_coverage(organized_data)

        assert result["total_metrics"] == 2
        assert result["modalities"] == ["cardio"]
        assert result["time_span_hours"] == 24.0

    @staticmethod
    def test_generate_feature_summary() -> None:
        """Test feature summary generation."""
        modality_features = {
            "cardio": [1.0, 2.0, 3.0],
            "respiratory": [4.0, 5.0, 6.0, 7.0]
        }

        result = HealthAnalysisPipeline._generate_feature_summary(modality_features)

        assert result["total_features"] == 7
        assert result["modality_dimensions"]["cardio"] == 3
        assert result["modality_dimensions"]["respiratory"] == 4

    @staticmethod
    def test_generate_health_indicators_all_modalities() -> None:
        """Test health indicators generation with all modalities."""
        pipeline = HealthAnalysisPipeline()

        modality_features = {
            "cardio": [75.0, 80.0, 70.0],  # Heart rate values
            "respiratory": [16.0, 18.0, 14.0],  # Breathing rate values
            "activity": [1000.0, 1500.0, 800.0]  # Activity values
        }

        activity_features = [
            {"avg_heart_rate": 75.0, "total_steps": 1000},
            {"avg_heart_rate": 80.0, "total_steps": 1500}
        ]

        result = pipeline._generate_health_indicators(modality_features, activity_features)

        assert "cardio" in result
        assert "respiratory" in result
        assert "activity" in result

    @staticmethod
    def test_extract_cardio_health_indicators() -> None:
        """Test cardio health indicators extraction."""
        modality_features = {
            "cardio": [75.0, 80.0, 70.0, 85.0]
        }

        result = HealthAnalysisPipeline._extract_cardio_health_indicators(modality_features)

        assert result is not None
        assert "avg_heart_rate" in result
        assert "heart_rate_variability" in result
        assert result["avg_heart_rate"] == 77.5  # (75+80+70+85)/4

    @staticmethod
    def test_extract_respiratory_health_indicators() -> None:
        """Test respiratory health indicators extraction."""
        modality_features = {
            "respiratory": [16.0, 18.0, 14.0, 20.0]
        }

        result = HealthAnalysisPipeline._extract_respiratory_health_indicators(modality_features)

        assert result is not None
        assert "avg_respiratory_rate" in result
        assert "respiratory_variability" in result
        assert result["avg_respiratory_rate"] == 17.0  # (16+18+14+20)/4

    @staticmethod
    def test_extract_activity_health_indicators() -> None:
        """Test activity health indicators extraction."""
        activity_features = [
            {"avg_heart_rate": 75.0, "total_steps": 1000, "active_minutes": 30},
            {"avg_heart_rate": 80.0, "total_steps": 1500, "active_minutes": 45}
        ]

        result = HealthAnalysisPipeline._extract_activity_health_indicators(activity_features)

        assert result is not None
        assert "avg_daily_steps" in result
        assert "avg_active_minutes" in result
        assert result["avg_daily_steps"] == 1250.0  # (1000+1500)/2
        assert result["avg_active_minutes"] == 37.5  # (30+45)/2

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
            id="1",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=start_time,
            data=BiometricData(value=75.0, unit="bpm")
        )

        metric2 = HealthMetric(
            id="2",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=end_time,
            data=BiometricData(value=80.0, unit="bpm")
        )

        result = HealthAnalysisPipeline._calculate_time_span([metric1, metric2])
        assert result == 24.0

    @staticmethod
    def test_calculate_time_span_single_metric() -> None:
        """Test time span calculation with single metric."""
        metric = HealthMetric(
            id="1",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=datetime.now(UTC),
            data=BiometricData(value=75.0, unit="bpm")
        )

        result = HealthAnalysisPipeline._calculate_time_span([metric])
        assert result == 0.0


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
        pipeline.cardio_processor.process = MagicMock(return_value=expected_cardio_features)

        cardio_metric = HealthMetric(
            id="1",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=datetime.now(UTC),
            data=BiometricData(value=75.0, unit="bpm")
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

        pipeline.cardio_processor.process = MagicMock(return_value=expected_cardio)
        pipeline.respiratory_processor.process = MagicMock(return_value=expected_respiratory)
        pipeline.fusion_service.fuse_embeddings = AsyncMock(return_value=expected_fused)

        cardio_metric = HealthMetric(
            id="1",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=datetime.now(UTC),
            data=BiometricData(value=75.0, unit="bpm")
        )

        respiratory_metric = HealthMetric(
            id="2",
            user_id="user1",
            metric_type=HealthMetricType.RESPIRATORY_RATE,
            timestamp=datetime.now(UTC),
            data=BiometricData(value=16.0, unit="breaths/min")
        )

        result = await pipeline.process_health_data("user1", [cardio_metric, respiratory_metric])

        assert result.cardio_features == expected_cardio
        assert result.respiratory_features == expected_respiratory
        assert result.fused_vector == expected_fused
        assert set(result.processing_metadata["modalities_processed"]) == {"cardio", "respiratory"}

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_health_data_with_firestore_save() -> None:
        """Test processing with Firestore saving when processing_id provided."""
        pipeline = HealthAnalysisPipeline()

        # Mock processors
        expected_cardio = [1.0, 2.0, 3.0]
        pipeline.cardio_processor.process = MagicMock(return_value=expected_cardio)

        # Mock Firestore client
        mock_firestore = AsyncMock()
        pipeline.firestore_client = mock_firestore

        cardio_metric = HealthMetric(
            id="1",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=datetime.now(UTC),
            data=BiometricData(value=75.0, unit="bpm")
        )

        processing_id = "test_processing_123"
        result = await pipeline.process_health_data("user1", [cardio_metric], processing_id)

        # Verify Firestore save was called
        mock_firestore.save_analysis_result.assert_called_once()
        call_args = mock_firestore.save_analysis_result.call_args
        assert call_args[1]["user_id"] == "user1"
        assert call_args[1]["processing_id"] == processing_id

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
                    "end_date": datetime.now(UTC).isoformat()
                }
            ],
            "category_samples": [],
            "workout_data": []
        }

        with patch("clarity.ml.analysis_pipeline.get_analysis_pipeline") as mock_get_pipeline:
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
        """Test run_analysis_pipeline with data conversion error."""
        # Invalid health data that should cause conversion to fail
        invalid_health_data = {"invalid_key": "invalid_value"}

        with pytest.raises(Exception):
            await run_analysis_pipeline("user1", invalid_health_data)


class TestHealthKitDataConversion:
    """Test HealthKit data conversion utility functions."""

    @staticmethod
    def test_get_healthkit_type_mapping() -> None:
        """Test HealthKit type mapping returns expected types."""
        mapping = _get_healthkit_type_mapping()

        assert isinstance(mapping, dict)
        assert "heart_rate" in mapping
        assert mapping["heart_rate"] == HealthMetricType.HEART_RATE
        assert "respiratory_rate" in mapping
        assert mapping["respiratory_rate"] == HealthMetricType.RESPIRATORY_RATE
        assert "steps" in mapping
        assert mapping["steps"] == HealthMetricType.STEPS

    @staticmethod
    def test_convert_raw_data_to_metrics_empty() -> None:
        """Test converting empty raw data."""
        health_data = {
            "quantity_samples": [],
            "category_samples": [],
            "workout_data": []
        }

        result = _convert_raw_data_to_metrics(health_data)
        assert result == []

    @staticmethod
    def test_process_quantity_samples() -> None:
        """Test processing quantity samples."""
        health_data = {
            "quantity_samples": [
                {
                    "type": "heart_rate",
                    "value": 75.0,
                    "unit": "bpm",
                    "start_date": datetime.now(UTC).isoformat(),
                    "end_date": datetime.now(UTC).isoformat()
                }
            ]
        }

        result = _process_quantity_samples(health_data)

        assert len(result) == 1
        assert result[0].metric_type == HealthMetricType.HEART_RATE
        assert isinstance(result[0].data, BiometricData)
        assert result[0].data.value == 75.0

    @staticmethod
    def test_create_biometric_data_from_sample() -> None:
        """Test creating biometric data from sample."""
        sample = {
            "value": 75.0,
            "unit": "bpm",
            "start_date": datetime.now(UTC).isoformat(),
            "end_date": datetime.now(UTC).isoformat()
        }

        result = _create_biometric_data_from_sample(sample, "heart_rate")

        assert isinstance(result, BiometricData)
        assert result.value == 75.0
        assert result.unit == "bpm"

    @staticmethod
    def test_process_category_samples() -> None:
        """Test processing category samples."""
        health_data = {
            "category_samples": [
                {
                    "type": "sleep_analysis",
                    "value": "in_bed",
                    "start_date": datetime.now(UTC).isoformat(),
                    "end_date": (datetime.now(UTC) + timedelta(hours=8)).isoformat()
                }
            ]
        }

        result = _process_category_samples(health_data)

        assert len(result) == 1
        assert result[0].metric_type == HealthMetricType.SLEEP

    @staticmethod
    def test_create_sleep_metric_from_sample() -> None:
        """Test creating sleep metric from sample."""
        start_time = datetime.now(UTC)
        end_time = start_time + timedelta(hours=8)

        sample = {
            "type": "sleep_analysis",
            "value": "in_bed",
            "start_date": start_time.isoformat(),
            "end_date": end_time.isoformat()
        }

        health_data = {"category_samples": [sample]}

        result = _create_sleep_metric_from_sample(sample, health_data)

        assert result.metric_type == HealthMetricType.SLEEP
        assert isinstance(result.data, SleepData)
        assert result.data.duration_minutes == 480  # 8 hours = 480 minutes

    @staticmethod
    def test_process_workout_data() -> None:
        """Test processing workout data."""
        health_data = {
            "workout_data": [
                {
                    "type": "running",
                    "start_date": datetime.now(UTC).isoformat(),
                    "end_date": (datetime.now(UTC) + timedelta(minutes=30)).isoformat(),
                    "duration": 1800,  # 30 minutes in seconds
                    "total_energy_burned": 250
                }
            ]
        }

        result = _process_workout_data(health_data)

        assert len(result) == 1
        assert result[0].metric_type == HealthMetricType.WORKOUT
        assert isinstance(result[0].data, ActivityData)


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
        pipeline.cardio_processor.process = MagicMock(side_effect=Exception("Processor error"))

        cardio_metric = HealthMetric(
            id="1",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=datetime.now(UTC),
            data=BiometricData(value=75.0, unit="bpm")
        )

        # Should not raise exception, but handle gracefully
        result = await pipeline.process_health_data("user1", [cardio_metric])

        # Processing should continue despite processor error
        assert isinstance(result, AnalysisResults)
        assert result.processing_metadata["total_metrics"] == 1

    @pytest.mark.asyncio
    @staticmethod
    async def test_process_health_data_firestore_error() -> None:
        """Test processing when Firestore save fails."""
        pipeline = HealthAnalysisPipeline()

        # Mock processors
        pipeline.cardio_processor.process = MagicMock(return_value=[1.0, 2.0, 3.0])

        # Mock Firestore client to raise error
        mock_firestore = AsyncMock()
        mock_firestore.save_analysis_result.side_effect = Exception("Firestore error")
        pipeline.firestore_client = mock_firestore

        cardio_metric = HealthMetric(
            id="1",
            user_id="user1",
            metric_type=HealthMetricType.HEART_RATE,
            timestamp=datetime.now(UTC),
            data=BiometricData(value=75.0, unit="bpm")
        )

        # Should continue processing despite Firestore error
        result = await pipeline.process_health_data("user1", [cardio_metric], "processing_123")

        assert isinstance(result, AnalysisResults)
        assert result.cardio_features == [1.0, 2.0, 3.0]
