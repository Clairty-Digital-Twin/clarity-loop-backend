"""Resilience and utility decorators for the CLARITY platform."""
from collections.abc import Callable
import functools
import logging
from typing import Any, TypeVar

from circuitbreaker import CircuitBreakerError, circuit

from clarity.services.health_data_service import MLPredictionError

logger = logging.getLogger(__name__)

# Type variable for decorated function's return value
T = TypeVar("T")


def resilient_prediction(
    failure_threshold: int = 5, recovery_timeout: int = 60, model_name: str = "ML model"
) -> Callable[..., Callable[..., T]]:
    """A decorator that makes an ML prediction function resilient.

    It adds a circuit breaker and standardized error handling.

    Args:
        failure_threshold: Number of failures to open the circuit.
        recovery_timeout: Seconds to wait before moving to half-open state.
        model_name: Name of the model for logging and error messages.

    Returns:
        A decorated function.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Apply the circuit breaker decorator to the original function
        circuit_breaker_func = circuit(
            failure_threshold=failure_threshold, recovery_timeout=recovery_timeout
        )(func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:  # noqa: ANN401
            """Wrapper that adds error handling and logging around the circuit breaker."""
            try:
                # The first argument of the decorated method is 'self'
                # The second is typically the input data.
                user_id = "unknown"
                if args and hasattr(args[1], "user_id"):
                    user_id = args[1].user_id

                logger.debug(
                    "Calling resilient prediction for model '%s' for user '%s'",
                    model_name,
                    user_id,
                )
                return await circuit_breaker_func(*args, **kwargs)

            except CircuitBreakerError as e:
                logger.exception(
                    "Circuit open for model '%s'. Prediction failed.", model_name
                )
                raise MLPredictionError(
                    message=f"Circuit is open for {model_name}. Service is temporarily unavailable.",
                    model_name=model_name,
                ) from e
            except Exception as e:
                logger.critical(
                    "Prediction failed for model '%s': %s",
                    model_name,
                    e,
                    exc_info=True,
                )
                raise MLPredictionError(
                    message=f"Error during prediction: {e!s}", model_name=model_name
                ) from e

        return wrapper

    return decorator
