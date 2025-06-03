#!/usr/bin/env python3
"""Test script for PAT model weight loading."""

import asyncio
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

try:
    from src.clarity.ml.pat_service import PATModelService
    
    async def test_weight_loading():
        """Test PAT model weight loading."""
        print("🧪 Testing PAT model weight loading...")
        
        service = PATModelService(model_size='medium')
        await service.load_model()
        
        print('✅ PAT model loaded successfully!')
        print(f'   Model loaded: {service.is_loaded}')
        print(f'   Model size: {service.model_size}')
        print(f'   Device: {service.device}')
        
        # Test health check
        health = await service.health_check()
        print(f'   Health check: {health}')
        
        return service.is_loaded

    if __name__ == "__main__":
        success = asyncio.run(test_weight_loading())
        sys.exit(0 if success else 1)
        
except Exception as e:
    print(f"❌ Error testing PAT loading: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 