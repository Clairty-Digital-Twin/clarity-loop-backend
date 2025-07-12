#!/usr/bin/env python3
"""Debug production-specific issues with PAT service."""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clarity.ml.pat_service import PATModelService

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')


async def test_production_issues():
    """Test production-specific issues that might cause PAT service to fail."""
    print("🔍 DEBUG: Testing production-specific issues...")

    try:
        # Test 1: Check if model files exist and are readable
        print("\n1. Testing model file accessibility...")
        model_path = Path(__file__).parent / "models" / "pat" / "PAT-M_29k_weights.h5"
        print(f"   📍 Model path: {model_path}")
        print(f"   📁 Exists: {model_path.exists()}")

        if model_path.exists():
            print(f"   📏 Size: {model_path.stat().st_size} bytes")
            print(f"   🔒 Permissions: {oct(model_path.stat().st_mode)}")

            # Test reading the file
            try:
                with open(model_path, 'rb') as f:
                    first_bytes = f.read(100)
                    print(f"   ✅ File readable, first 20 bytes: {first_bytes[:20]}")
            except Exception as e:
                print(f"   ❌ File read error: {e}")

        # Test 2: Check h5py availability and functionality
        print("\n2. Testing h5py functionality...")
        try:
            import h5py
            print(f"   ✅ h5py available: {h5py.__version__}")

            # Test opening the model file
            if model_path.exists():
                try:
                    with h5py.File(model_path, 'r') as f:
                        print("   ✅ h5py can open file")
                        print(f"   📊 Keys in file: {list(f.keys())}")
                except Exception as e:
                    print(f"   ❌ h5py file error: {e}")

        except ImportError as e:
            print(f"   ❌ h5py not available: {e}")

        # Test 3: Test service creation in isolation
        print("\n3. Testing service creation...")
        service = PATModelService(model_size="medium")
        print("   ✅ Service created")
        print(f"   📍 Using model path: {service.model_path}")

        # Test 4: Test model loading with error handling
        print("\n4. Testing model loading...")
        try:
            await service.load_model()
            print(f"   ✅ Model loaded: {service.is_loaded}")

            if service.model:
                print(f"   🧠 Model type: {type(service.model)}")
                print(f"   🔧 Model on device: {next(service.model.parameters()).device}")

        except Exception as e:
            print(f"   ❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 5: Check for missing dependencies
        print("\n5. Testing dependencies...")
        try:
            import torch
            print(f"   ✅ PyTorch: {torch.__version__}")
        except ImportError as e:
            print(f"   ❌ PyTorch missing: {e}")

        try:
            import numpy as np
            print(f"   ✅ NumPy: {np.__version__}")
        except ImportError as e:
            print(f"   ❌ NumPy missing: {e}")

        # Test 6: Test with minimal data
        print("\n6. Testing minimal inference...")
        if service.is_loaded:
            try:
                from datetime import datetime

                from clarity.ml.pat_service import ActigraphyInput
                from clarity.ml.preprocessing import ActigraphyDataPoint

                # Create the absolute minimum data
                test_data = ActigraphyInput(
                    user_id="test-user",
                    data_points=[ActigraphyDataPoint(timestamp=datetime.now(), value=1.0)],
                    sampling_rate=1.0,
                    duration_hours=1.0
                )

                result = await service.analyze_actigraphy(test_data)
                print("   ✅ Minimal inference successful")
                print(f"   💤 Sleep efficiency: {result.sleep_efficiency}")

            except Exception as e:
                print(f"   ❌ Minimal inference failed: {e}")
                import traceback
                traceback.print_exc()

        print("\n🎯 DEBUG COMPLETE")

    except Exception as e:
        print(f"\n❌ OVERALL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_production_issues())
