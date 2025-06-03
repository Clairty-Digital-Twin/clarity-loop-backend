"""Ports layer - Service interfaces following Clean Architecture.

This layer defines the contracts (interfaces/ports) that the business logic
layer depends on. It follows the Dependency Inversion Principle where
high-level modules (business logic) depend on abstractions (ports),
not on low-level modules (infrastructure implementations).

The ports are implemented by adapters in the infrastructure layer.
"""

from .auth_ports import IAuthProvider
from .config_ports import IConfigProvider
from .data_ports import IHealthDataRepository
from .middleware_ports import IMiddleware
from .ml_ports import IMLModelService

__all__ = [
    "IAuthProvider",
    "IConfigProvider", 
    "IHealthDataRepository",
    "IMiddleware",
    "IMLModelService",
] 