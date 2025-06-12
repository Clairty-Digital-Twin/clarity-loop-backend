"""Health Data Preprocessing Service.

This service handles the preprocessing of health data for ML model input,
following the Strategy pattern for different preprocessing approaches.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import TYPE_CHECKING, Protocol

import numpy as np
from pydantic import BaseModel, Field
import torch

if TYPE_CHECKING:
    from clarity.models.health_data import HealthMetric

logger = logging.getLogger(__name__)


class ActigraphyDataPoint(BaseModel):
    """Individual actigraphy data point for ML processing."""

    timestamp: datetime
    value: float = Field(description="Activity count or acceleration value")


class PreprocessingStrategy(Protocol):
    """Protocol for preprocessing strategies."""

    def preprocess(
        self, data_points: list[ActigraphyDataPoint], target_length: int
    ) -> torch.Tensor:
        """Preprocess data points into model-ready tensor."""
        ...


class StandardActigraphyPreprocessor:
    """Standard actigraphy preprocessing implementation."""

    @staticmethod
    def preprocess(
        data_points: list[ActigraphyDataPoint], target_length: int = 1440
    ) -> torch.Tensor:
        """Preprocess actigraphy data for PAT model input.

        Args:
            data_points: Raw actigraphy data points
            target_length: Target sequence length (default: 1440 for 24h at 1min resolution)

        Returns:
            Preprocessed tensor ready for model input
        """
        # Extract activity values
        values = [point.value for point in data_points]

        # Convert to numpy array
        activity_data = np.array(values, dtype=np.float32)

        # Normalize data (z-score normalization)
        if len(activity_data) > 1:
            mean_val = np.mean(activity_data)
            std_val = np.std(activity_data)
            if std_val > 0:
                activity_data = (activity_data - mean_val) / std_val

        # Resize to target length
        if len(activity_data) != target_length:
            # Simple interpolation/padding
            if len(activity_data) > target_length:
                # Down-sample
                indices = np.linspace(
                    0, len(activity_data) - 1, target_length, dtype=int
                )
                activity_data = activity_data[indices]
            else:
                # Pad with zeros
                padded = np.zeros(target_length, dtype=np.float32)
                padded[: len(activity_data)] = activity_data
                activity_data = padded

        # Convert to tensor (PAT expects 1D sequence, service adds batch dim)
        return torch.FloatTensor(activity_data)


class HealthDataPreprocessor:
    """Main preprocessing service using Strategy pattern."""

    def __init__(self, strategy: PreprocessingStrategy | None = None) -> None:
        self.strategy = strategy or StandardActigraphyPreprocessor()

    def set_strategy(self, strategy: PreprocessingStrategy) -> None:
        """Set the preprocessing strategy."""
        self.strategy = strategy

    @staticmethod
    def convert_health_metrics_to_actigraphy(
        metrics: list[HealthMetric],
    ) -> list[ActigraphyDataPoint]:
        """Convert HealthMetric objects to ActigraphyDataPoint objects."""
        actigraphy_points: list[ActigraphyDataPoint] = []

        for metric in metrics:
            # Extract activity data from different metric types
            if metric.activity_data and metric.activity_data.steps:
                actigraphy_points.append(
                    ActigraphyDataPoint(
                        timestamp=metric.created_at,
                        value=float(metric.activity_data.steps),
                    )
                )
            elif metric.biometric_data and metric.biometric_data.heart_rate:
                # Use heart rate as activity proxy
                actigraphy_points.append(
                    ActigraphyDataPoint(
                        timestamp=metric.created_at,
                        value=float(metric.biometric_data.heart_rate),
                    )
                )

        return actigraphy_points

    def preprocess_for_pat_model(
        self, data_points: list[ActigraphyDataPoint], target_length: int = 1440
    ) -> torch.Tensor:
        """Preprocess data using the current strategy."""
        return self.strategy.preprocess(data_points, target_length)
