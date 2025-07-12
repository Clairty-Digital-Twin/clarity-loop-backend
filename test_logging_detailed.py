#!/usr/bin/env python3
"""Detailed test to diagnose logging duplication."""

import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Before any imports, check the initial state
print("=== INITIAL STATE ===")
root_logger = logging.getLogger()
print(f"Root logger handlers: {len(root_logger.handlers)}")
print(f"Root logger level: {root_logger.level}")

# Import logging config
from clarity.core.logging_config import configure_basic_logging

print("\n=== AFTER IMPORT ===")
print(f"Root logger handlers: {len(root_logger.handlers)}")

# Configure logging
configure_basic_logging(level=logging.INFO)

print("\n=== AFTER FIRST CONFIG ===")
print(f"Root logger handlers: {len(root_logger.handlers)}")
for i, handler in enumerate(root_logger.handlers):
    print(f"  Handler {i}: {handler.__class__.__name__} - Level: {handler.level}")

# Test message
test_logger = logging.getLogger("test")
print(f"\nTest logger propagate: {test_logger.propagate}")
print(f"Test logger handlers: {len(test_logger.handlers)}")

# Create a custom handler to count messages
class CountingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.messages = []
    
    def emit(self, record):
        self.count += 1
        self.messages.append(self.format(record))

# Add counting handler
counter = CountingHandler()
counter.setFormatter(logging.Formatter("%(message)s"))
test_logger.addHandler(counter)

# Log a test message
print("\n=== LOGGING TEST MESSAGE ===")
test_logger.info("Test message")

print(f"\nMessages received by counting handler: {counter.count}")
print(f"Messages: {counter.messages}")

# Check all loggers
print("\n=== ALL LOGGERS ===")
for name in logging.Logger.manager.loggerDict:
    logger = logging.getLogger(name)
    print(f"{name}: handlers={len(logger.handlers)}, propagate={logger.propagate}")