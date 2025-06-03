"""
Clarity Digital Twin - Integrations Module

External service integrations for the Clarity platform:
- Apple HealthKit integration for Apple Watch data
- Wearable device APIs
- Third-party health platforms
"""

from .healthkit import HealthKitClient
from .apple_watch import AppleWatchDataProcessor

__all__ = ["HealthKitClient", "AppleWatchDataProcessor"]