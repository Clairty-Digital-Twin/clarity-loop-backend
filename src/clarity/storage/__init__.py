"""CLARITY Digital Twin Platform - Storage Layer.

Provides high-performance, HIPAA-compliant data storage services
for the health data processing pipeline.
"""

from clarity.storage.firestore_client import FirestoreClient

__all__ = ["FirestoreClient"]
