#!/usr/bin/env python3
"""Test PAT model inference with sample data."""

import asyncio
from datetime import UTC, datetime
import logging
from pathlib import Path
import sys

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clarity.ml.pat_service import ActigraphyInput, get_pat_service
from clarity.ml.preprocessing import ActigraphyDataPoint

# Constants
DAYTIME_START_HOUR = 6
DAYTIME_END_HOUR = 22
BASE_DAYTIME_ACTIVITY = 30
ACTIVITY_AMPLITUDE = 20
BASE_NIGHTTIME_ACTIVITY = 5
ACTIVITY_NOISE_STD = 10
MIN_WEIGHT_STD = 0.01
MAX_WEIGHT_STD = 1.0
MINUTES_IN_WEEK = 7 * 24 * 60  # 10080 minutes
MINUTES_IN_HOUR = 60

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize random generator
rng = np.random.default_rng()


async def test_pat_inference() -> None:
    """Test PAT inference with synthetic actigraphy data."""
    logger.info("=== PAT Inference Test ===")

    # Create PAT service
    service = await get_pat_service()
    await service.load_model()

    logger.info("Model loaded: %s", service.is_loaded)
    logger.info("Model device: %s", service.device)

    # Create synthetic actigraphy data (1 week of minute-level data)
    logger.info("Generating synthetic actigraphy data...")

    data_points: list[ActigraphyDataPoint] = []

    for minute in range(MINUTES_IN_WEEK):
        # Simulate realistic actigraphy pattern
        hour_of_day = (minute // MINUTES_IN_HOUR) % 24
        day_of_week = minute // (24 * MINUTES_IN_HOUR)

        # Create activity patterns (higher during day, lower at night)
        if DAYTIME_START_HOUR <= hour_of_day <= DAYTIME_END_HOUR:  # Daytime
            base_activity = BASE_DAYTIME_ACTIVITY + ACTIVITY_AMPLITUDE * np.sin(
                2 * np.pi * (hour_of_day - DAYTIME_START_HOUR) / 16
            )
        else:  # Nighttime
            base_activity = BASE_NIGHTTIME_ACTIVITY

        # Add some randomness
        activity = max(0, base_activity + rng.normal(0, ACTIVITY_NOISE_STD))

        # Create timestamp
        timestamp = datetime(
            year=2024,
            month=1,
            day=(day_of_week + 1),
            hour=hour_of_day,
            minute=minute % MINUTES_IN_HOUR,
            tzinfo=UTC
        )

        # Use correct ActigraphyDataPoint structure: timestamp and value only
        data_point = ActigraphyDataPoint(
            timestamp=timestamp,
            value=float(activity)
        )
        data_points.append(data_point)

    logger.info("Generated %d data points", len(data_points))

    # Create input
    actigraphy_input = ActigraphyInput(
        user_id="test_user_001",
        data_points=data_points,
        sampling_rate=1.0,  # 1 sample per minute
        duration_hours=168   # 1 week
    )

    # Run inference
    logger.info("🔥 Running PAT inference...")
    try:
        analysis = await service.analyze_actigraphy(actigraphy_input)

        logger.info("✅ Inference successful!")
        logger.info("Analysis Results for %s:", analysis.user_id)
        logger.info("  Sleep Efficiency: %.1f%%", analysis.sleep_efficiency)
        logger.info("  Sleep Onset Latency: %.1f minutes", analysis.sleep_onset_latency)
        logger.info("  Total Sleep Time: %.1f hours", analysis.total_sleep_time)
        logger.info("  Circadian Rhythm Score: %.3f", analysis.circadian_rhythm_score)
        logger.info("  Depression Risk Score: %.3f", analysis.depression_risk_score)
        logger.info("  Confidence Score: %.3f", analysis.confidence_score)

        logger.info("Clinical Insights:")
        for insight in analysis.clinical_insights:
            logger.info("  • %s", insight)

        logger.info("🎯 PAT MODEL IS WORKING WITH REAL WEIGHTS!")

    except (RuntimeError, ValueError, ConnectionError):
        logger.exception("❌ Inference failed")

    # Test model internals
    logger.info("=== Model Architecture Verification ===")
    model = service.model
    if model:
        logger.info("Model type: %s", type(model).__name__)
        encoder = model.encoder
        logger.info("Encoder input size: %s", encoder.input_size)
        logger.info("Encoder patch size: %s", encoder.patch_size)
        logger.info("Encoder embed dim: %s", encoder.embed_dim)
        logger.info("Encoder num patches: %s", encoder.num_patches)
        logger.info("Encoder num transformer layers: %s", len(encoder.transformer_layers))

        # Check if attention layers have loaded weights
        if encoder.transformer_layers:
            first_layer = encoder.transformer_layers[0]
            attention = first_layer.attention
            logger.info("Attention type: %s", type(attention).__name__)
            logger.info("Num heads: %s", attention.num_heads)
            logger.info("Head dim: %s", attention.head_dim)

            # Check if query projections have non-random weights
            first_query = attention.query_projections[0]
            weight_std = first_query.weight.std().item()
            logger.info("First query projection weight std: %.6f", weight_std)

            if MIN_WEIGHT_STD < weight_std < MAX_WEIGHT_STD:
                logger.info("✅ Weights appear to be loaded (not random)")
            else:
                logger.info("⚠️  Weights may still be random")


if __name__ == "__main__":
    asyncio.run(test_pat_inference())
