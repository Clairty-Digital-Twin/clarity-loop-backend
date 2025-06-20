"""Comprehensive production tests for Health Analysis Pipeline.

Tests the complete analysis workflow, including modality processing,
data organization, PAT integration, fusion, and DynamoDB storage.
Focuses on real functionality without over-mocking.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

from clarity.ml.analysis_pipeline import (
    MIN_FEATURE_VECTOR_LENGTH,
    MIN_METRICS_FOR_TIME_SPAN,
    AnalysisResults,
    HealthAnalysisPipeline,
    _get_healthkit_type_mapping,  # noqa: PLC2701
    get_analysis_pipeline,
)
from clarity.ml.processors.sleep_processor import SleepFeatures
from clarity.models.health_data import (
    ActivityData,
    BiometricData,
    HealthMetric,
    HealthMetricType,
    SleepData,
)


class TestAnalysisResults:
    """Test AnalysisResults container class."""

    def test_analysis_results_initialization(self):
        """Test AnalysisResults initialization with default values."""
        results = AnalysisResults()

        assert results.cardio_features == []
        assert results.respiratory_features == []
        assert results.activity_features == []
        assert results.activity_embedding == []
        assert results.sleep_features == {}
        assert results.fused_vector == []
        assert results.summary_stats == {}
        assert results.processing_metadata == {}

    def test_analysis_results_assignment(self):
        """Test AnalysisResults field assignment."""
        results = AnalysisResults()

        # Test all field assignments
        results.cardio_features = [1.0, 2.0, 3.0]
        results.respiratory_features = [4.0, 5.0]
        results.activity_features = [{"steps": 1000}]
        results.activity_embedding = [0.1] * 128
        results.sleep_features = {"efficiency": 0.85}
        results.fused_vector = [0.5] * 64
        results.summary_stats = {"total_metrics": 10}
        results.processing_metadata = {"user_id": "test_user"}

        assert len(results.cardio_features) == 3
        assert len(results.respiratory_features) == 2
        assert len(results.activity_features) == 1
        assert len(results.activity_embedding) == 128
        assert results.sleep_features["efficiency"] == 0.85
        assert len(results.fused_vector) == 64
        assert results.summary_stats["total_metrics"] == 10
        assert results.processing_metadata["user_id"] == "test_user"


class TestHealthAnalysisPipelineInitialization:
    """Test pipeline initialization and setup."""

    def test_pipeline_initialization(self):
        """Test basic pipeline initialization."""
        pipeline = HealthAnalysisPipeline()

        # Check processor initialization
        assert pipeline.cardio_processor is not None
        assert pipeline.respiratory_processor is not None
        assert pipeline.activity_processor is not None
        assert pipeline.sleep_processor is not None
        assert pipeline.preprocessor is not None

        # Check ML services (initially None)
        assert pipeline.pat_service is None
        assert pipeline.fusion_service is not None

        # Check storage client (initially None)
        assert pipeline.dynamodb_client is None

    @patch("os.getenv")
    @patch("clarity.ml.analysis_pipeline.DynamoDBHealthDataRepository")
    async def test_get_dynamodb_client(
        self, mock_repository: MagicMock, mock_getenv: MagicMock
    ) -> None:
        """Test DynamoDB client initialization."""
        mock_getenv.side_effect = {
            "DYNAMODB_TABLE_NAME": "test-table",
            "AWS_REGION": "us-west-2",
        }.get

        mock_client = Mock()
        mock_repository.return_value = mock_client

        pipeline = HealthAnalysisPipeline()
        client = await pipeline._get_dynamodb_client()

        assert client == mock_client
        assert pipeline.dynamodb_client == mock_client
        mock_repository.assert_called_once_with(
            table_name="test-table", region="us-west-2"
        )

        # Test cached client
        client2 = await pipeline._get_dynamodb_client()
        assert client2 == mock_client
        assert mock_repository.call_count == 1


class TestDataOrganization:
    """Test data organization by modality."""

    def test_organize_metrics_by_modality_empty(self):
        """Test organization with empty metrics."""
        pipeline = HealthAnalysisPipeline()
        result = pipeline._organize_metrics_by_modality([])

        expected = {
            "cardio": [],
            "respiratory": [],
            "activity": [],
            "sleep": [],
            "other": [],
        }
        assert result == expected

    def test_organize_metrics_by_modality_comprehensive(self):
        """Test organization with comprehensive metrics."""
        pipeline = HealthAnalysisPipeline()

        # Create test metrics for each modality
        metrics = [
            # Cardio metrics
            HealthMetric(
                id=str(uuid4()),
                user_id="test_user",
                metric_type=HealthMetricType.HEART_RATE,
                created_at=datetime.now(UTC),
                biometric_data=BiometricData(heart_rate=75),
            ),
            HealthMetric(
                id=str(uuid4()),
                user_id="test_user",
                metric_type=HealthMetricType.HEART_RATE_VARIABILITY,
                created_at=datetime.now(UTC),
                biometric_data=BiometricData(heart_rate_variability=45),
            ),
            # Respiratory metrics
            HealthMetric(
                id=str(uuid4()),
                user_id="test_user",
                metric_type=HealthMetricType.RESPIRATORY_RATE,
                created_at=datetime.now(UTC),
                biometric_data=BiometricData(respiratory_rate=16),
            ),
            HealthMetric(
                id=str(uuid4()),
                user_id="test_user",
                metric_type=HealthMetricType.BLOOD_OXYGEN,
                created_at=datetime.now(UTC),
                biometric_data=BiometricData(oxygen_saturation=98),
            ),
            # Activity metrics
            HealthMetric(
                id=str(uuid4()),
                user_id="test_user",
                metric_type=HealthMetricType.ACTIVITY_LEVEL,
                created_at=datetime.now(UTC),
                activity_data=ActivityData(steps=1000),
            ),
            HealthMetric(
                id=str(uuid4()),
                user_id="test_user",
                metric_type=HealthMetricType.ACTIVITY_LEVEL,
                created_at=datetime.now(UTC),
                activity_data=ActivityData(active_energy=250.0),
            ),
            # Sleep metrics
            HealthMetric(
                id=str(uuid4()),
                user_id="test_user",
                metric_type=HealthMetricType.SLEEP_ANALYSIS,
                created_at=datetime.now(UTC),
                sleep_data=SleepData(
                    total_sleep_minutes=480,
                    sleep_efficiency=0.85,
                    sleep_start=datetime.now(UTC) - timedelta(hours=8),
                    sleep_end=datetime.now(UTC),
                ),
            ),
        ]

        result = pipeline._organize_metrics_by_modality(metrics)

        assert len(result["cardio"]) == 2
        assert len(result["respiratory"]) == 2
        assert len(result["activity"]) == 2
        assert len(result["sleep"]) == 1
        assert len(result["other"]) == 0


class TestModalityProcessing:
    """Test individual modality processing methods."""

    async def test_process_cardio_data_comprehensive(self):
        """Test cardiovascular data processing."""
        pipeline = HealthAnalysisPipeline()

        # Create test cardio metrics
        base_time = datetime.now(UTC)
        cardio_metrics = [
            HealthMetric(
                id=str(uuid4()),
                user_id="test_user",
                metric_type=HealthMetricType.HEART_RATE,
                created_at=base_time,
                biometric_data=BiometricData(heart_rate=75),
            ),
            HealthMetric(
                id=str(uuid4()),
                user_id="test_user",
                metric_type=HealthMetricType.HEART_RATE,
                created_at=base_time + timedelta(minutes=1),
                biometric_data=BiometricData(heart_rate=78),
            ),
        ]

        with patch.object(pipeline.cardio_processor, "process") as mock_process:
            mock_process.return_value = [1.0, 2.0, 3.0]

            result = await pipeline._process_cardio_data(cardio_metrics)

            assert result == [1.0, 2.0, 3.0]
            mock_process.assert_called_once()

    async def test_process_respiratory_data_comprehensive(self):
        """Test respiratory data processing."""
        pipeline = HealthAnalysisPipeline()

        # Create test respiratory metrics
        base_time = datetime.now(UTC)
        respiratory_metrics = [
            HealthMetric(
                id=str(uuid4()),
                user_id="test_user",
                metric_type=HealthMetricType.RESPIRATORY_RATE,
                created_at=base_time,
                biometric_data=BiometricData(respiratory_rate=16),
            ),
        ]

        with patch.object(pipeline.respiratory_processor, "process") as mock_process:
            mock_process.return_value = [4.0, 5.0]

            result = await pipeline._process_respiratory_data(respiratory_metrics)

            assert result == [4.0, 5.0]
            mock_process.assert_called_once()

    def test_convert_sleep_features_to_vector(self):
        """Test sleep features to vector conversion."""
        # Create mock sleep features
        sleep_features = SleepFeatures(
            total_sleep_minutes=480,  # 8 hours
            sleep_efficiency=0.85,
            sleep_latency=15,
            waso_minutes=30,
            awakenings_count=3,
            rem_percentage=0.20,
            deep_percentage=0.15,
            consistency_score=0.75,
        )

        result = HealthAnalysisPipeline._convert_sleep_features_to_vector(
            sleep_features
        )

        assert len(result) == 8
        assert result[0] == 1.0  # 480/480 normalized
        assert result[1] == 0.85  # efficiency
        assert result[2] == 0.25  # 15/60 normalized
        assert result[3] == 0.25  # 30/120 normalized
        assert result[4] == 0.3  # 3/10 normalized
        assert result[5] == 0.20  # REM percentage
        assert result[6] == 0.15  # Deep percentage
        assert result[7] == 0.75  # Consistency score


class TestMainWorkflow:
    """Test the main process_health_data workflow."""

    @patch("clarity.ml.analysis_pipeline.get_pat_service")
    async def test_process_health_data_single_modality(
        self, mock_get_pat: MagicMock
    ) -> None:
        """Test processing with single modality (cardio only)."""
        pipeline = HealthAnalysisPipeline()

        # Create test cardio metric
        metrics = [
            HealthMetric(
                id=str(uuid4()),
                user_id="test_user",
                metric_type=HealthMetricType.HEART_RATE,
                created_at=datetime.now(UTC),
                biometric_data=BiometricData(heart_rate=75),
            ),
        ]

        # Mock cardio processor
        with patch.object(pipeline.cardio_processor, "process") as mock_cardio:
            mock_cardio.return_value = [1.0, 2.0, 3.0]

            result = await pipeline.process_health_data("test_user", metrics)

            assert isinstance(result, AnalysisResults)
            assert result.cardio_features == [1.0, 2.0, 3.0]
            assert result.fused_vector == [
                1.0,
                2.0,
                3.0,
            ]  # Single modality becomes fused vector
            assert result.processing_metadata["user_id"] == "test_user"
            assert result.processing_metadata["total_metrics"] == 1
            assert result.processing_metadata["modalities_processed"] == ["cardio"]

    async def test_process_health_data_no_metrics(self):
        """Test processing with no metrics."""
        pipeline = HealthAnalysisPipeline()

        result = await pipeline.process_health_data("test_user", [])

        assert isinstance(result, AnalysisResults)
        assert result.cardio_features == []
        assert result.respiratory_features == []
        assert result.activity_features == []
        assert result.fused_vector == []
        assert result.processing_metadata["total_metrics"] == 0
        assert result.processing_metadata["modalities_processed"] == []


class TestSummaryStatistics:
    """Test summary statistics generation."""

    def test_generate_data_coverage(self):
        """Test data coverage statistics generation."""
        base_time = datetime.now(UTC)
        organized_data = {
            "cardio": [Mock(), Mock()],  # 2 cardio metrics
            "respiratory": [Mock()],  # 1 respiratory metric
            "activity": [],  # 0 activity metrics
            "sleep": [Mock()],  # 1 sleep metric
            "other": [],  # 0 other metrics
        }

        # Mock time span calculation
        for metrics in organized_data.values():
            for metric in metrics:
                metric.created_at = base_time

        result = HealthAnalysisPipeline._generate_data_coverage(organized_data)

        # The method returns nested dictionaries for each modality with data
        assert "cardio" in result
        assert "respiratory" in result
        assert "sleep" in result
        assert "activity" not in result  # No activity metrics
        assert "other" not in result  # No other metrics

        # Check cardio data structure
        assert result["cardio"]["metric_count"] == 2
        assert (
            result["cardio"]["time_span_hours"] == 1.0
        )  # Single timestamp defaults to 1.0
        assert result["cardio"]["data_density"] == 2.0  # 2 metrics / 1 hour

    def test_generate_feature_summary(self):
        """Test feature summary generation."""
        modality_features = {
            "cardio": [1.0, 2.0, 3.0],  # 3 features
            "respiratory": [4.0, 5.0],  # 2 features
            "activity": [6.0] * 128,  # 128 features
        }

        result = HealthAnalysisPipeline._generate_feature_summary(modality_features)

        # The method returns nested dictionaries for each modality
        assert "cardio" in result
        assert "respiratory" in result
        assert "activity" in result

        # Check cardio feature summary
        assert result["cardio"]["feature_count"] == 3
        assert result["cardio"]["mean_value"] == 2.0  # (1+2+3)/3

        # Check respiratory feature summary
        assert result["respiratory"]["feature_count"] == 2
        assert result["respiratory"]["mean_value"] == 4.5  # (4+5)/2

        # Check activity feature summary
        assert result["activity"]["feature_count"] == 128
        assert result["activity"]["mean_value"] == 6.0  # All values are 6.0

    def test_calculate_time_span(self):
        """Test time span calculation."""
        base_time = datetime.now(UTC)
        metrics = [
            Mock(created_at=base_time),
            Mock(created_at=base_time + timedelta(hours=1)),
            Mock(created_at=base_time + timedelta(hours=2)),
        ]

        time_span = HealthAnalysisPipeline._calculate_time_span(metrics)

        assert time_span == 2.0  # 2 hours difference

    def test_calculate_time_span_single_metric(self):
        """Test time span calculation with single metric."""
        metrics = [Mock(created_at=datetime.now(UTC))]

        time_span = HealthAnalysisPipeline._calculate_time_span(metrics)

        assert time_span == 1.0  # Fixed: Single metric defaults to 1.0 hour, not 0.0


class TestUtilityFunctions:
    """Test utility and helper functions."""

    def test_get_analysis_pipeline_singleton(self):
        """Test analysis pipeline singleton functionality."""
        pipeline1 = get_analysis_pipeline()
        pipeline2 = get_analysis_pipeline()

        assert pipeline1 is pipeline2
        assert isinstance(pipeline1, HealthAnalysisPipeline)

    def test_get_healthkit_type_mapping(self):
        """Test HealthKit type mapping."""
        mapping = _get_healthkit_type_mapping()

        assert isinstance(mapping, dict)
        assert len(mapping) > 0

        # Test some known mappings - the actual keys are lowercase
        assert "heartrate" in mapping
        assert mapping["heartrate"] == HealthMetricType.HEART_RATE
        assert "heart_rate" in mapping
        assert mapping["heart_rate"] == HealthMetricType.HEART_RATE


class TestConstants:
    """Test module constants."""

    def test_constants_defined(self):
        """Test that required constants are defined."""
        assert MIN_FEATURE_VECTOR_LENGTH == 8
        assert MIN_METRICS_FOR_TIME_SPAN == 2
        assert isinstance(MIN_FEATURE_VECTOR_LENGTH, int)
        assert isinstance(MIN_METRICS_FOR_TIME_SPAN, int)


class TestCreateActivityEmbedding:
    """Test activity embedding creation from PAT analysis."""

    def test_create_activity_embedding_with_actual_embedding(self):
        """Test embedding creation when PAT analysis has embedding."""
        pipeline = HealthAnalysisPipeline()

        # Mock analysis result with embedding
        analysis_result = Mock()
        analysis_result.embedding = [0.1] * 128

        result = pipeline._create_activity_embedding_from_analysis(analysis_result)

        assert result == [0.1] * 128

    def test_create_activity_embedding_without_embedding(self):
        """Test embedding creation when PAT analysis has no embedding."""
        pipeline = HealthAnalysisPipeline()

        # Mock analysis result without embedding
        analysis_result = Mock()
        analysis_result.embedding = None
        analysis_result.sleep_efficiency = 85.0
        analysis_result.total_sleep_time = 7.5
        analysis_result.circadian_rhythm_score = 0.8
        analysis_result.activity_fragmentation = 0.3
        analysis_result.wake_after_sleep_onset = 25.0
        analysis_result.sleep_onset_latency = 12.0
        analysis_result.depression_risk_score = 0.2

        result = pipeline._create_activity_embedding_from_analysis(analysis_result)

        assert len(result) == 128

        # Test specific normalized values
        assert result[0] == (85.0 - 75) / 25  # sleep_efficiency
        assert result[1] == (7.5 - 7.5) / 2.5  # total_sleep_time
        assert result[2] == (0.8 - 0.5) / 0.5  # circadian_rhythm_score

        # Check that remaining dimensions are filled
        assert all(isinstance(x, float) for x in result)
