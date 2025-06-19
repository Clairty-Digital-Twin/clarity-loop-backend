"""Machine Learning port interfaces.

Defines the contract for ML model services following Clean Architecture.
Business logic layer depends on this abstraction, not concrete implementations.
"""

# removed - breaks FastAPI

from abc import ABC, abstractmethod


class IMLModelService(ABC):
    """Abstract interface for machine learning model services.

    Defines the contract for ML model operations following Clean Architecture.
    Business logic layer depends on this abstraction, not concrete implementations.
    """

    @abstractmethod
    async def load_model(self) -> None:
        """Load the ML model asynchronously.

        Performs model initialization, weight loading, and any other
        setup operations required before the model can be used for inference.
        """

    @abstractmethod
    async def health_check(self) -> dict[str, str | bool]:
        """Check the health status of the ML model service.

        Returns:
            Dictionary containing health status information including
            model readiness, version, last update time, etc.
        """
