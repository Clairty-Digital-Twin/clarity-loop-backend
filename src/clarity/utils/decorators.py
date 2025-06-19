"""Resilience and utility decorators for the CLARITY platform."""

# removed - breaks FastAPI

from collections.abc import Awaitable, Callable
import functools
import logging
from typing import ParamSpec, TypeVar

from circuitbreaker import (
    CircuitBreaker,
    CircuitBreakerError,
)
from prometheus_client import Counter

from clarity.core.exceptions import DataValidationError, ServiceUnavailableProblem

# Import health data service error at module level to avoid circular imports
try:
    from clarity.services.health_data_service import HealthDataServiceError

    _HEALTH_DATA_SERVICE_ERROR: type[Exception] | None = HealthDataServiceError
except ImportError:
    _HEALTH_DATA_SERVICE_ERROR = None

logger = logging.getLogger(__name__)

# Prometheus metrics for predictions
PREDICTION_SUCCESS = Counter(
    "prediction_success_total", "Total number of successful predictions", ["model_name"]
)
PREDICTION_FAILURE = Counter(
    "prediction_failure_total", "Total number of failed predictions", ["model_name"]
)

T = TypeVar("T")
P = ParamSpec("P")


def resilient_prediction(
    failure_threshold: int = 5, recovery_timeout: int = 60, model_name: str = "ML model"
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """A decorator that makes an ML prediction function resilient.

    It adds a circuit breaker and standardized error handling.

    Args:
        failure_threshold: Number of failures to open the circuit.
        recovery_timeout: Seconds to wait before moving to half-open state.
        model_name: Name of the model for logging and error messages.

    Returns:
        A decorated function.
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        """Decorator that adds a circuit breaker to an async function."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold, recovery_timeout=recovery_timeout
        )

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            """Wrapper that adds error handling and logging around the circuit breaker."""
            try:
                # Use the circuit breaker decorator approach
                result: T = await circuit_breaker(func)(*args, **kwargs)
            except CircuitBreakerError as e:
                msg = f"{model_name} is currently unavailable. Please try again later."
                logger.exception("Circuit breaker is open for %s", model_name)
                PREDICTION_FAILURE.labels(model_name=model_name).inc()
                raise ServiceUnavailableProblem(msg) from e
            except Exception as e:
                # Allow validation errors to pass through - these are user input issues
                if isinstance(e, DataValidationError):
                    PREDICTION_FAILURE.labels(model_name=model_name).inc()
                    raise

                # Allow domain-specific exceptions to pass through unchanged
                if _HEALTH_DATA_SERVICE_ERROR and isinstance(
                    e, _HEALTH_DATA_SERVICE_ERROR
                ):
                    PREDICTION_FAILURE.labels(model_name=model_name).inc()
                    raise

                # For all other exceptions, wrap in ServiceUnavailableProblem
                msg = f"An unexpected error occurred in {model_name}."
                logger.exception("An unexpected error occurred during prediction")
                PREDICTION_FAILURE.labels(model_name=model_name).inc()
                raise ServiceUnavailableProblem(msg) from e
            else:
                PREDICTION_SUCCESS.labels(model_name=model_name).inc()
                return result

        return wrapper

    return decorator
