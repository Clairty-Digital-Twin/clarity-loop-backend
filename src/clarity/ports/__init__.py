"""Ports layer - Service interfaces following Clean Architecture.

This layer defines the contracts (interfaces/ports) that the business logic
layer depends on. It follows the Dependency Inversion Principle where
high-level modules (business logic) depend on abstractions (ports),
not on low-level modules (infrastructure implementations).

The ports are implemented by adapters in the infrastructure layer.
"""

# removed - breaks FastAPI

from clarity.ports.auth_ports import IAuthProvider
from clarity.ports.config_ports import IConfigProvider
from clarity.ports.data_ports import IHealthDataRepository
from clarity.ports.middleware_ports import IMiddleware
from clarity.ports.ml_ports import IMLModelService

__all__ = [
    "IAuthProvider",
    "IConfigProvider",
    "IHealthDataRepository",
    "IMLModelService",
    "IMiddleware",
]
