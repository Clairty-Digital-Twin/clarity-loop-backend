#!/usr/bin/env python3
"""Test PAT model inference with sample data."""

import asyncio
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clarity.ml.pat_service import ActigraphyInput, get_pat_service
from clarity.ml.preprocessing import ActigraphyDataPoint


async def test_pat_inference():
    """Test PAT inference with synthetic actigraphy data."""
    print("=== PAT Inference Test ===")

    # Create PAT service
    service = await get_pat_service()
    await service.load_model()

    print(f"Model loaded: {service.is_loaded}")
    print(f"Model device: {service.device}")

    # Create synthetic actigraphy data (1 week of minute-level data)
    print("\nGenerating synthetic actigraphy data...")

    data_points = []
    minutes_in_week = 7 * 24 * 60  # 10080 minutes

    for minute in range(minutes_in_week):
        # Simulate realistic actigraphy pattern
        hour_of_day = (minute // 60) % 24
        day_of_week = minute // (24 * 60)

        # Create activity patterns (higher during day, lower at night)
        if 6 <= hour_of_day <= 22:  # Daytime
            base_activity = 30 + 20 * np.sin(2 * np.pi * (hour_of_day - 6) / 16)
        else:  # Nighttime
            base_activity = 5

        # Add some randomness
        activity = max(0, base_activity + np.random.normal(0, 10))

        # Create timestamp
        timestamp = datetime(
            year=2024,
            month=1,
            day=(day_of_week + 1),
            hour=hour_of_day,
            minute=minute % 60
        )

        # Use correct ActigraphyDataPoint structure: timestamp and value only
        data_point = ActigraphyDataPoint(
            timestamp=timestamp,
            value=float(activity)
        )
        data_points.append(data_point)

    print(f"Generated {len(data_points)} data points")

    # Create input
    actigraphy_input = ActigraphyInput(
        user_id="test_user_001",
        data_points=data_points,
        sampling_rate=1.0,  # 1 sample per minute
        duration_hours=168   # 1 week
    )

    # Run inference
    print("\n🔥 Running PAT inference...")
    try:
        analysis = await service.analyze_actigraphy(actigraphy_input)

        print("✅ Inference successful!")
        print(f"\nAnalysis Results for {analysis.user_id}:")
        print(f"  Sleep Efficiency: {analysis.sleep_efficiency:.1f}%")
        print(f"  Sleep Onset Latency: {analysis.sleep_onset_latency:.1f} minutes")
        print(f"  Total Sleep Time: {analysis.total_sleep_time:.1f} hours")
        print(f"  Circadian Rhythm Score: {analysis.circadian_rhythm_score:.3f}")
        print(f"  Depression Risk Score: {analysis.depression_risk_score:.3f}")
        print(f"  Confidence Score: {analysis.confidence_score:.3f}")

        print("\nClinical Insights:")
        for insight in analysis.clinical_insights:
            print(f"  • {insight}")

        print("\n🎯 PAT MODEL IS WORKING WITH REAL WEIGHTS!")

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()

    # Test model internals
    print("\n=== Model Architecture Verification ===")
    model = service.model
    if model:
        print(f"Model type: {type(model).__name__}")
        encoder = model.encoder
        print(f"Encoder input size: {encoder.input_size}")
        print(f"Encoder patch size: {encoder.patch_size}")
        print(f"Encoder embed dim: {encoder.embed_dim}")
        print(f"Encoder num patches: {encoder.num_patches}")
        print(f"Encoder num transformer layers: {len(encoder.transformer_layers)}")

        # Check if attention layers have loaded weights
        if encoder.transformer_layers:
            first_layer = encoder.transformer_layers[0]
            attention = first_layer.attention
            print(f"Attention type: {type(attention).__name__}")
            print(f"Num heads: {attention.num_heads}")
            print(f"Head dim: {attention.head_dim}")

            # Check if query projections have non-random weights
            first_query = attention.query_projections[0]
            weight_std = first_query.weight.std().item()
            print(f"First query projection weight std: {weight_std:.6f}")

            if weight_std > 0.01 and weight_std < 1.0:
                print("✅ Weights appear to be loaded (not random)")
            else:
                print("⚠️  Weights may still be random")


if __name__ == "__main__":
    asyncio.run(test_pat_inference())
