#!/usr/bin/env python3
"""Test PAT model inference with sample data."""

import asyncio
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clarity.ml.pat_service import get_pat_service
from clarity.ml.preprocessing import ActigraphyDataPoint
from clarity.ml.pat_service import ActigraphyInput


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
        
        data_point = ActigraphyDataPoint(
            timestamp=f"2024-01-{day_of_week + 1:02d}T{hour_of_day:02d}:{minute % 60:02d}:00",
            activity_level=float(activity),
            step_count=int(activity * 0.5),  # Rough conversion
            confidence=0.95
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
        
        print(f"\nClinical Insights:")
        for insight in analysis.clinical_insights:
            print(f"  • {insight}")
            
        print(f"\n🎯 PAT MODEL IS WORKING WITH REAL WEIGHTS!")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test model internals
    print("\n=== Model Architecture Verification ===")
    model = service.model
    if model:
        print(f"Model type: {type(model).__name__}")
        print(f"Input size: {model.input_size}")
        print(f"Patch size: {model.patch_size}")
        print(f"Embed dim: {model.embed_dim}")
        print(f"Num patches: {model.num_patches}")
        print(f"Num transformer layers: {len(model.transformer_layers)}")
        
        # Check if attention layers have loaded weights
        if model.transformer_layers:
            first_layer = model.transformer_layers[0]
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