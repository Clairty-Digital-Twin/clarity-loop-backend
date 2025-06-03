"""Clarity Digital Twin - Integrations Module.

External service integrations for the Clarity platform:
- Apple HealthKit integration for Apple Watch data
- Wearable device APIs
- Third-party health platforms
"""

from clarity.integrations.apple_watch import AppleWatchDataProcessor
from clarity.integrations.healthkit import HealthKitClient

__all__ = ["AppleWatchDataProcessor", "HealthKitClient"]
