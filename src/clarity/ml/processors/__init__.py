"""Health Data Processors Package.

Modular signal processing components for different health data modalities.
Each processor extracts features or embeddings from specific health domains.
"""

from .cardio_processor import CardioProcessor
from .respiration_processor import RespirationProcessor

__all__ = [
    "CardioProcessor", 
    "RespirationProcessor"
]