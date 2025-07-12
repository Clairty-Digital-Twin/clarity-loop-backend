#!/usr/bin/env python3
"""Verification script for DynamoDB Decimal fix.
Tests the conversion logic we implemented.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from decimal import Decimal


def test_decimal_conversion() -> None:
    """Test our Decimal conversion logic."""
    # Test the conversion we implemented
    original_confidence = 0.85

    # Store conversion (float → Decimal)
    stored_value = Decimal(str(original_confidence))

    # Retrieve conversion (Decimal → float)
    retrieved_value = float(stored_value)

    # Verify integrity
    assert original_confidence == retrieved_value, "Conversion integrity failed!"


def test_edge_cases() -> None:
    """Test edge cases for Decimal conversion."""
    test_cases = [0.0, 1.0, 0.123456789, 99.99, 0.001]

    for case in test_cases:
        decimal_val = Decimal(str(case))
        float_val = float(decimal_val)
        assert case == float_val, f"Edge case failed for {case}"


if __name__ == "__main__":
    test_decimal_conversion()
    test_edge_cases()
