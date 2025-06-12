"""Tests for Apple Watch data processor."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from clarity.integrations.apple_watch import (
    AppleWatchDataProcessor,
    ProcessedHealthData,
)
from clarity.integrations.healthkit import HealthDataPoint


@pytest.mark.asyncio
async def test_process_heart_rate():
    processor = AppleWatchDataProcessor()
    result = ProcessedHealthData()
    start_time = datetime.now(UTC) - timedelta(days=1)
    end_time = datetime.now(UTC)

    samples = [
        HealthDataPoint(
            timestamp=start_time + timedelta(minutes=i),
            value=float(70 + (i % 20)),
            unit="count/min",
        )
        for i in range(1440)
    ]

    await processor._process_heart_rate(samples, result, start_time, end_time)

    assert result.heart_rate_series is not None
    assert result.avg_hr is not None
    assert result.max_hr is not None
    assert result.resting_hr is not None


@pytest.mark.asyncio
async def test_process_hrv():
    processor = AppleWatchDataProcessor()
    result = ProcessedHealthData()
    start_time = datetime.now(UTC) - timedelta(days=1)
    end_time = datetime.now(UTC)

    samples = [
        HealthDataPoint(
            timestamp=start_time + timedelta(minutes=i),
            value=float(50 + (i % 10)),
            unit="ms",
        )
        for i in range(1440)
    ]

    await processor._process_hrv(samples, result, start_time, end_time)

    assert result.hrv_series is not None
    assert result.hrv_median is not None


@pytest.mark.asyncio
async def test_process_respiratory_rate():
    processor = AppleWatchDataProcessor()
    result = ProcessedHealthData()
    start_time = datetime.now(UTC) - timedelta(days=1)
    end_time = datetime.now(UTC)

    samples = [
        HealthDataPoint(
            timestamp=start_time + timedelta(minutes=i * 5),
            value=float(16 + (i % 4)),
            unit="count/min",
        )
        for i in range(288)
    ]

    await processor._process_respiratory_rate(samples, result, start_time, end_time)

    assert result.respiratory_rate_series is not None
    assert result.avg_respiratory_rate is not None


@pytest.mark.asyncio
async def test_process_steps():
    processor = AppleWatchDataProcessor()
    result = ProcessedHealthData()
    start_time = datetime.now(UTC) - timedelta(days=7)
    end_time = datetime.now(UTC)

    samples = [
        HealthDataPoint(
            timestamp=start_time + timedelta(minutes=i),
            value=float(i % 100),
            unit="count",
        )
        for i in range(10080)
    ]

    await processor._process_steps(samples, result, start_time, end_time)

    assert result.movement_proxy_vector is not None
    assert len(result.movement_proxy_vector) == 10080


@pytest.mark.asyncio
async def test_process_spo2():
    processor = AppleWatchDataProcessor()
    result = ProcessedHealthData()
    samples = [
        HealthDataPoint(timestamp=datetime.now(UTC), value=98.0, unit="%"),
        HealthDataPoint(timestamp=datetime.now(UTC), value=99.0, unit="%"),
    ]

    await processor._process_spo2(samples, result)

    assert result.avg_spo2 == 98.5
    assert result.min_spo2 == 98.0


@pytest.mark.asyncio
async def test_process_blood_pressure():
    processor = AppleWatchDataProcessor()
    result = ProcessedHealthData()
    sample = HealthDataPoint(
        timestamp=datetime.now(UTC), value=0, unit=""
    )  # value and unit are not used
    sample.systolic = 120.0
    sample.diastolic = 80.0
    samples = [sample]

    await processor._process_blood_pressure(samples, result)

    assert result.systolic_bp == 120.0
    assert result.diastolic_bp == 80.0


@pytest.mark.asyncio
async def test_process_temperature():
    processor = AppleWatchDataProcessor()
    result = ProcessedHealthData()
    samples = [
        HealthDataPoint(timestamp=datetime.now(UTC), value=0.5, unit="degC"),
        HealthDataPoint(timestamp=datetime.now(UTC), value=-0.5, unit="degC"),
    ]

    await processor._process_temperature(samples, result)

    assert result.avg_temp_deviation == 0.0


@pytest.mark.asyncio
async def test_process_vo2_max():
    processor = AppleWatchDataProcessor()
    result = ProcessedHealthData()
    samples = [
        HealthDataPoint(
            timestamp=datetime.now(UTC) - timedelta(days=1),
            value=42.0,
            unit="mL/min·kg",
        ),
        HealthDataPoint(timestamp=datetime.now(UTC), value=45.0, unit="mL/min·kg"),
    ]

    await processor._process_vo2_max(samples, result)

    assert result.vo2_max == 45.0
    assert result.vo2_max_trend == 3.0


@pytest.mark.asyncio
async def test_process_workouts():
    processor = AppleWatchDataProcessor()
    result = ProcessedHealthData()
    result.resting_hr = 60.0  # Set a resting HR for intensity calculation
    workout1 = MagicMock()
    workout1.duration_minutes = 30
    workout1.avg_heart_rate = 150
    workout1.age = 30
    workout2 = MagicMock()
    workout2.duration_minutes = 60
    workout2.avg_heart_rate = 140
    workout2.age = 30
    samples = [workout1, workout2]

    await processor._process_workouts(samples, result)

    assert result.workout_count == 2
    assert result.total_active_minutes == 90
    assert result.avg_workout_intensity is not None


@pytest.mark.asyncio
async def test_process_ecg():
    processor = AppleWatchDataProcessor()
    result = ProcessedHealthData()
    ecg_sample = MagicMock()
    ecg_sample.classification = "Atrial Fibrillation"
    samples = [ecg_sample]

    await processor._process_ecg(samples, result)

    assert result.afib_detected is True
    assert result.ecg_classification == "Atrial Fibrillation"


def test_calculate_completeness():
    processor = AppleWatchDataProcessor()
    result = ProcessedHealthData()

    result.heart_rate_series = np.array([70, 72, np.nan, 75])
    result.hrv_series = np.array([50, 55, 52, 58])
    result.respiratory_rate_series = np.array([16, np.nan, np.nan, 18])

    completeness = processor._calculate_completeness(result)
    # Expected: (0.75 + 1.0 + 0.5 + 0.0) / 4 = 0.5625 * 100 = 56.25
    assert completeness == pytest.approx(56.25)
