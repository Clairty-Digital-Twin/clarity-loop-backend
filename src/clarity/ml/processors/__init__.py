"""Health Data Processors Package.

Modular signal processing components for different health data modalities.
Each processor extracts features or embeddings from specific health domains.
"""

# removed - breaks FastAPI

from clarity.ml.processors.cardio_processor import CardioProcessor
from clarity.ml.processors.respiration_processor import RespirationProcessor

__all__ = ["CardioProcessor", "RespirationProcessor"]
