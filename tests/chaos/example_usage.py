#!/usr/bin/env python3
"""Example usage of chaos testing for model corruption scenarios.

This script demonstrates how to manually trigger and test various
corruption scenarios for development and debugging purposes.
"""

import asyncio
import logging
from pathlib import Path
import tempfile

from clarity.ml.model_integrity import ModelChecksumManager
from clarity.ml.pat_model_loader import ModelSize, PATModelLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_checksum_validation():
    """Demonstrate model checksum validation."""
    logger.info("=== Demonstrating Checksum Validation ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models"
        model_dir.mkdir()
        
        # Create checksum manager
        manager = ModelChecksumManager(model_dir)
        
        # Create a mock model file
        model_file = model_dir / "test_model.pth"
        model_file.write_bytes(b"This is a test model content")
        
        # Register the model
        logger.info("Registering model with checksum...")
        manager.register_model("test_model", ["test_model.pth"])
        
        # Verify integrity - should pass
        logger.info("Verifying model integrity (should pass)...")
        is_valid = manager.verify_model_integrity("test_model")
        logger.info(f"Model valid: {is_valid}")
        
        # Corrupt the model
        logger.info("Corrupting model file...")
        with open(model_file, "ab") as f:
            f.write(b"CORRUPTION")
        
        # Verify again - should fail
        logger.info("Verifying model integrity (should fail)...")
        is_valid = manager.verify_model_integrity("test_model")
        logger.info(f"Model valid: {is_valid}")


async def demonstrate_circuit_breaker():
    """Demonstrate circuit breaker behavior."""
    logger.info("\n=== Demonstrating Circuit Breaker ===")
    
    from clarity.utils.decorators import resilient_prediction
    
    failure_count = 0
    
    @resilient_prediction(
        failure_threshold=3,
        recovery_timeout=2,
        model_name="demo_model"
    )
    async def flaky_model_prediction(should_fail: bool = True):
        nonlocal failure_count
        if should_fail:
            failure_count += 1
            raise RuntimeError(f"Model failure #{failure_count}")
        return {"prediction": "success"}
    
    # Trigger failures to open circuit
    logger.info("Triggering failures to open circuit breaker...")
    for i in range(4):
        try:
            await flaky_model_prediction(should_fail=True)
        except Exception as e:
            logger.warning(f"Attempt {i+1}: {type(e).__name__}: {e}")
    
    logger.info(f"Total actual model calls: {failure_count}")
    
    # Wait for recovery
    logger.info("Waiting for circuit breaker recovery timeout...")
    await asyncio.sleep(2.5)
    
    # Try again with success
    logger.info("Attempting prediction after recovery...")
    try:
        result = await flaky_model_prediction(should_fail=False)
        logger.info(f"Success: {result}")
    except Exception as e:
        logger.error(f"Still failing: {e}")


async def demonstrate_model_fallback():
    """Demonstrate model version fallback."""
    logger.info("\n=== Demonstrating Model Fallback ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models"
        model_dir.mkdir()
        
        # Create model loader
        loader = PATModelLoader(model_dir)
        
        # This would normally load actual models
        logger.info("Model fallback demonstration requires actual model files")
        logger.info("In production, this would:")
        logger.info("1. Try to load latest model version")
        logger.info("2. Detect corruption via checksum or load failure")
        logger.info("3. Automatically fallback to previous version")
        logger.info("4. Log the fallback event for monitoring")


async def demonstrate_concurrent_corruption():
    """Demonstrate handling corruption under load."""
    logger.info("\n=== Demonstrating Concurrent Corruption Handling ===")
    
    async def simulate_model_request(request_id: int, delay: float):
        """Simulate a model request with potential corruption."""
        await asyncio.sleep(delay)
        
        # Simulate 30% corruption rate after 1 second
        if delay > 1.0 and request_id % 3 == 0:
            logger.error(f"Request {request_id}: Model corrupted!")
            raise RuntimeError("Model corruption detected")
        else:
            logger.info(f"Request {request_id}: Success")
            return f"Result for request {request_id}"
    
    # Launch concurrent requests
    logger.info("Launching 10 concurrent requests...")
    tasks = [
        simulate_model_request(i, i * 0.3)
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    successes = sum(1 for r in results if not isinstance(r, Exception))
    failures = sum(1 for r in results if isinstance(r, Exception))
    
    logger.info(f"Results: {successes} successes, {failures} failures")


async def main():
    """Run all demonstrations."""
    logger.info("Starting Chaos Testing Demonstrations\n")
    
    await demonstrate_checksum_validation()
    await demonstrate_circuit_breaker()
    await demonstrate_model_fallback()
    await demonstrate_concurrent_corruption()
    
    logger.info("\nDemonstrations complete!")


if __name__ == "__main__":
    asyncio.run(main())