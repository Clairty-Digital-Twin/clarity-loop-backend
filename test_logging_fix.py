#!/usr/bin/env python3
"""Test script to verify logging duplication fix."""

import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Test 1: Import and use logging config multiple times
print("Test 1: Testing multiple imports of logging config")
from clarity.core.logging_config import configure_basic_logging, setup_logging

# First configuration
configure_basic_logging(level=logging.INFO)
logger1 = logging.getLogger("test1")
logger1.info("First logger message - should appear once")

# Second configuration (should not duplicate)
configure_basic_logging(level=logging.INFO)
logger2 = logging.getLogger("test2")
logger2.info("Second logger message - should appear once")

# Test 2: Test setup_logging
print("\nTest 2: Testing setup_logging")
setup_logging()
logger3 = logging.getLogger("test3")
logger3.info("Third logger message - should appear once")

# Test 3: Force reconfiguration
print("\nTest 3: Testing force reconfiguration")
setup_logging(force=True)
logger4 = logging.getLogger("test4")
logger4.info("Fourth logger message - after force reconfiguration")

# Test 4: Check handler count
print("\nTest 4: Checking handler count")
root_logger = logging.getLogger()
print(f"Root logger handlers: {len(root_logger.handlers)}")
for i, handler in enumerate(root_logger.handlers):
    print(f"  Handler {i}: {handler.__class__.__name__}")

# Test 5: Import multiple modules that configure logging
print("\nTest 5: Testing imports from multiple modules")
try:
    # These imports should not cause duplicate handlers
    from clarity.main import logger as main_logger
    from clarity.startup.orchestrator import logger as orch_logger
    
    print("Successfully imported loggers from multiple modules")
    
    # Test a log message
    test_logger = logging.getLogger("final_test")
    test_logger.info("Final test message - should appear only once")
    
except Exception as e:
    print(f"Error during imports: {e}")

print("\nLogging test completed. Check above for any duplicate messages.")