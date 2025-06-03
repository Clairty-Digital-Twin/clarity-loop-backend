"""Core interfaces for dependency inversion - DEPRECATED.

This module has been refactored into the ports layer following Clean Architecture.
Use clarity.ports.* imports instead of this module.

MIGRATION GUIDE:
- IAuthProvider -> from clarity.ports.auth_ports import IAuthProvider
- IConfigProvider -> from clarity.ports.config_ports import IConfigProvider
- IHealthDataRepository -> from clarity.ports.data_ports import IHealthDataRepository
- IMiddleware -> from clarity.ports.middleware_ports import IMiddleware
- IMLModelService -> from clarity.ports.ml_ports import IMLModelService
"""

# Import from ports layer for backward compatibility
from clarity.ports.auth_ports import IAuthProvider
from clarity.ports.config_ports import IConfigProvider
from clarity.ports.data_ports import IHealthDataRepository
from clarity.ports.middleware_ports import IMiddleware
from clarity.ports.ml_ports import IMLModelService

__all__ = [
    "IAuthProvider",
    "IConfigProvider",
    "IHealthDataRepository", 
    "IMiddleware",
    "IMLModelService",
]
