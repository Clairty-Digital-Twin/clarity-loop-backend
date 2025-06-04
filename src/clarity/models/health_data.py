"""Clinical-Grade Health Data Models.

Revolutionary Pydantic models for the psychiatry digital twin platform.
These models establish entirely new standards for clinical data validation,
HIPAA compliance, and mental health analytics.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import PositiveInt

# Constants for validation
MAX_METRICS_PER_UPLOAD = 100
MAX_SLEEP_DURATION_MINUTES = 30
MAX_NOTES_LENGTH = 1000


class ProcessingStatus(StrEnum):
    """Processing status for health data uploads."""

    RECEIVED = "received"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


class HealthMetricType(StrEnum):
    """Types of health metrics supported by the digital twin."""

    HEART_RATE = "heart_rate"
    HEART_RATE_VARIABILITY = "heart_rate_variability"
    BLOOD_PRESSURE = "blood_pressure"
    BLOOD_OXYGEN = "blood_oxygen"
    RESPIRATORY_RATE = (
        "respiratory_rate"  # 🚀 FIXED: New dedicated respiratory rate type
    )
    SLEEP_ANALYSIS = "sleep_analysis"
    ACTIVITY_LEVEL = "activity_level"
    STRESS_INDICATORS = "stress_indicators"
    MOOD_ASSESSMENT = "mood_assessment"
    COGNITIVE_METRICS = "cognitive_metrics"
    ENVIRONMENTAL = "environmental"
    BODY_TEMPERATURE = "body_temperature"
    BLOOD_GLUCOSE = "blood_glucose"


class SleepStage(StrEnum):
    """Sleep stages for detailed sleep analysis."""

    AWAKE = "awake"
    LIGHT = "light"
    DEEP = "deep"
    REM = "rem"


class MoodScale(StrEnum):
    """Standardized mood assessment scale."""

    VERY_LOW = "very_low"
    LOW = "low"
    NEUTRAL = "neutral"
    GOOD = "good"
    EXCELLENT = "excellent"


class ValidationError(BaseModel):
    """Standardized validation error structure."""

    field: str
    message: str
    code: str
    value: Any


class BiometricData(BaseModel):
    """Biometric measurements data."""

    heart_rate: float | None = Field(
        None, ge=30, le=300, description="Heart rate in BPM"
    )
    blood_pressure_systolic: int | None = Field(
        None, ge=70, le=250, description="Systolic BP in mmHg"
    )
    blood_pressure_diastolic: int | None = Field(
        None, ge=40, le=150, description="Diastolic BP in mmHg"
    )
    oxygen_saturation: float | None = Field(
        None, ge=70.0, le=100.0, description="Blood oxygen saturation in %"
    )
    heart_rate_variability: float | None = Field(
        None, ge=0, le=200, description="HRV in milliseconds"
    )
    respiratory_rate: float | None = Field(
        None, ge=5, le=60, description="Respiratory rate in breaths per minute"
    )
    body_temperature: float | None = Field(
        None, ge=35.0, le=42.0, description="Body temperature in Celsius"
    )
    blood_glucose: float | None = Field(
        None, ge=50, le=500, description="Blood glucose in mg/dL"
    )


class SleepData(BaseModel):
    """Comprehensive sleep analysis data."""

    total_sleep_minutes: PositiveInt = Field(
        description="Total sleep duration in minutes", examples=[480]
    )

    sleep_efficiency: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        description="Sleep efficiency as a ratio (0-1)", examples=[0.85]
    )

    time_to_sleep_minutes: Annotated[int, Field(ge=0, le=180)] | None = Field(
        None, description="Time to fall asleep in minutes", examples=[12]
    )

    wake_count: Annotated[int, Field(ge=0, le=50)] | None = Field(
        None, description="Number of times awakened during sleep", examples=[2]
    )

    sleep_stages: dict[SleepStage, int] | None = Field(
        None,
        description="Minutes spent in each sleep stage",
        examples=[{"light": 240, "deep": 120, "rem": 90, "awake": 30}],
    )

    sleep_start: datetime = Field(description="Sleep start time (UTC)")

    sleep_end: datetime = Field(description="Sleep end time (UTC)")

    @model_validator(mode="before")
    @classmethod
    def validate_sleep_times(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate sleep timing consistency."""
        start = values.get("sleep_start")
        end = values.get("sleep_end")
        total_minutes = values.get("total_sleep_minutes")

        if start and end:
            if end <= start:
                msg = "Sleep end time must be after start time"
                raise ValueError(msg)

            calculated_minutes = int((end - start).total_seconds() / 60)
            if (
                total_minutes
                and abs(calculated_minutes - total_minutes) > MAX_SLEEP_DURATION_MINUTES
            ):
                msg = "Total sleep minutes inconsistent with start/end times"
                raise ValueError(msg)

        return values


class ActivityData(BaseModel):
    """Activity and exercise data."""

    steps: int | None = Field(None, ge=0, le=100000, description="Step count")
    distance: float | None = Field(
        None, ge=0, le=100.0, description="Distance in kilometers"
    )
    active_energy: float | None = Field(
        None, ge=0, le=5000.0, description="Active energy burned in kcal"
    )
    exercise_minutes: int | None = Field(
        None, ge=0, le=1440, description="Exercise time in minutes"
    )
    flights_climbed: int | None = Field(
        None, ge=0, le=500, description="Flights of stairs climbed"
    )
    vo2_max: float | None = Field(
        None, ge=10.0, le=80.0, description="VO₂ max in mL/kg/min"
    )
    active_minutes: int | None = Field(
        None, ge=0, le=1440, description="Total active minutes"
    )
    resting_heart_rate: float | None = Field(
        None, ge=30, le=120, description="Resting heart rate in BPM"
    )


class MentalHealthIndicator(BaseModel):
    """Mental health and mood indicators."""

    mood_score: MoodScale | None = Field(
        None, description="Standardized mood assessment"
    )

    stress_level: Annotated[float, Field(ge=1.0, le=10.0)] | None = Field(
        None, description="Stress level on 1-10 scale", examples=[3.5]
    )

    anxiety_level: Annotated[float, Field(ge=1.0, le=10.0)] | None = Field(
        None, description="Anxiety level on 1-10 scale", examples=[2.8]
    )

    energy_level: Annotated[float, Field(ge=1.0, le=10.0)] | None = Field(
        None, description="Energy level on 1-10 scale", examples=[7.2]
    )

    focus_rating: Annotated[float, Field(ge=1.0, le=10.0)] | None = Field(
        None, description="Focus/concentration rating on 1-10 scale", examples=[6.5]
    )

    social_interaction_minutes: Annotated[int, Field(ge=0, le=1440)] | None = Field(
        None, description="Minutes of social interaction", examples=[120]
    )

    meditation_minutes: Annotated[int, Field(ge=0, le=480)] | None = Field(
        None, description="Minutes of meditation/mindfulness", examples=[15]
    )

    notes: str | None = Field(
        None,
        max_length=MAX_NOTES_LENGTH,
        description="Free-form notes about mental state",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Assessment timestamp (UTC)",
    )


class HealthMetric(BaseModel):
    """Unified health metric container."""

    metric_id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for this metric"
    )

    metric_type: HealthMetricType = Field(description="Type of health metric")

    biometric_data: BiometricData | None = None
    sleep_data: SleepData | None = None
    activity_data: ActivityData | None = None
    mental_health_data: MentalHealthIndicator | None = None

    device_id: str | None = Field(None, description="Identifier of the source device")

    raw_data: dict[str, Any] | None = Field(
        None, description="Raw data from the source device"
    )

    metadata: dict[str, Any] | None = Field(
        None, description="Additional metadata about the measurement"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Record creation timestamp (UTC)",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_metric_data(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Ensure the metric has appropriate data for its type."""
        metric_type = values.get("metric_type")

        # Type guard: ensure metric_type is valid before using as dict key
        if not metric_type or not isinstance(metric_type, HealthMetricType):
            return values  # Skip validation if metric_type is invalid

        type_data_map = {
            HealthMetricType.HEART_RATE: "biometric_data",
            HealthMetricType.HEART_RATE_VARIABILITY: "biometric_data",
            HealthMetricType.BLOOD_PRESSURE: "biometric_data",
            HealthMetricType.SLEEP_ANALYSIS: "sleep_data",
            HealthMetricType.ACTIVITY_LEVEL: "activity_data",
            HealthMetricType.STRESS_INDICATORS: "mental_health_data",
            HealthMetricType.MOOD_ASSESSMENT: "mental_health_data",
            HealthMetricType.COGNITIVE_METRICS: "mental_health_data",
        }

        if metric_type in type_data_map:
            required_field = type_data_map[metric_type]
            if not values.get(required_field):
                msg = f"Metric type {metric_type} requires {required_field}"
                raise ValueError(msg)

        return values


class HealthDataUpload(BaseModel):
    """Clinical-grade health data upload request."""

    user_id: UUID = Field(description="Authenticated user identifier")

    metrics: Annotated[
        list[HealthMetric],
        Field(description="Health metrics to upload", min_length=1, max_length=100),
    ]

    upload_source: str = Field(
        description="Source of the upload (e.g., 'apple_health', 'fitbit', 'manual')",
        examples=["apple_health"],
    )

    client_timestamp: datetime = Field(description="Client-side upload timestamp (UTC)")

    sync_token: str | None = Field(
        None, description="Synchronization token for incremental uploads"
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics_consistency(cls, v: list[HealthMetric]) -> list[HealthMetric]:
        """Validate metrics are consistent and reasonable."""
        if len(v) > MAX_METRICS_PER_UPLOAD:
            msg = f"Maximum {MAX_METRICS_PER_UPLOAD} metrics per upload"
            raise ValueError(msg)

        # Check for duplicate metric IDs
        metric_ids = [m.metric_id for m in v]
        if len(metric_ids) != len(set(metric_ids)):
            msg = "Duplicate metric IDs not allowed"
            raise ValueError(msg)

        return v


class HealthDataResponse(BaseModel):
    """Response for health data upload requests."""

    processing_id: UUID = Field(
        default_factory=uuid4, description="Unique processing identifier"
    )

    status: ProcessingStatus = Field(description="Current processing status")

    accepted_metrics: int = Field(
        description="Number of metrics accepted for processing"
    )

    rejected_metrics: int = Field(default=0, description="Number of metrics rejected")

    validation_errors: list[ValidationError] = Field(
        default_factory=list, description="Validation errors encountered"
    )

    estimated_processing_time: int | None = Field(
        None, description="Estimated processing time in seconds"
    )

    sync_token: str | None = Field(
        None, description="New synchronization token for next upload"
    )

    message: str = Field(
        description="Human-readable status message",
        examples=["Health data uploaded successfully and is being processed"],
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response timestamp (UTC)",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "processing_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "processing",
                "accepted_metrics": 5,
                "rejected_metrics": 0,
                "validation_errors": [],
                "estimated_processing_time": 30,
                "message": "Health data uploaded successfully and is being processed",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
