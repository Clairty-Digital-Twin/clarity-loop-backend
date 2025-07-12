#!/usr/bin/env python3
"""Debug PAT service initialization to find the exact failure point."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clarity.ml.pat_service import PATModelService, get_pat_service

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')

async def test_pat_service():
    """Test PAT service initialization step by step."""
    print("🔍 DEBUG: Testing PAT service initialization...")
    
    try:
        # Test 1: Direct service creation
        print("\n1. Testing direct PATModelService creation...")
        service = PATModelService(model_size="medium")
        print(f"   ✅ Service created: {service}")
        print(f"   📍 Model path: {service.model_path}")
        print(f"   📦 Model size: {service.model_size}")
        print(f"   💻 Device: {service.device}")
        print(f"   ⚡ Is loaded: {service.is_loaded}")
        
        # Test 2: Model loading
        print("\n2. Testing model loading...")
        await service.load_model()
        print(f"   ✅ Model loaded: {service.is_loaded}")
        print(f"   🧠 Model object: {service.model}")
        
        # Test 3: Global service
        print("\n3. Testing global service...")
        global_service = await get_pat_service()
        print(f"   ✅ Global service: {global_service}")
        print(f"   ⚡ Is loaded: {global_service.is_loaded}")
        
        # Test 4: Simple inference test
        print("\n4. Testing basic inference capability...")
        from clarity.ml.pat_service import ActigraphyInput
        from clarity.ml.preprocessing import ActigraphyDataPoint
        from datetime import datetime
        
        # Create minimal test data
        test_data = ActigraphyInput(
            user_id="test-user",
            data_points=[
                ActigraphyDataPoint(timestamp=datetime.now(), value=1.0),
                ActigraphyDataPoint(timestamp=datetime.now(), value=2.0),
            ],
            sampling_rate=1.0,
            duration_hours=1.0
        )
        
        # Try analysis
        result = await global_service.analyze_actigraphy(test_data)
        print(f"   ✅ Analysis result: {type(result)}")
        print(f"   📊 User ID: {result.user_id}")
        print(f"   💤 Sleep efficiency: {result.sleep_efficiency}")
        
        print("\n🎉 ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_pat_service())
    sys.exit(0 if success else 1) 