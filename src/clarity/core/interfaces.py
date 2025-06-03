"""🚨 DEPRECATED: Core interfaces - USE PORTS LAYER INSTEAD

⚠️  This module is DEPRECATED and will be removed in a future version.
    All imports have been migrated to the new ports layer following Clean Architecture.

❌ DO NOT USE:
    from clarity.core.interfaces import IAuthProvider

✅ USE INSTEAD:
    from clarity.ports.auth_ports import IAuthProvider

📋 COMPLETE MIGRATION GUIDE:
- IAuthProvider        → from clarity.ports.auth_ports import IAuthProvider
- IConfigProvider      → from clarity.ports.config_ports import IConfigProvider  
- IHealthDataRepository → from clarity.ports.data_ports import IHealthDataRepository
- IMiddleware          → from clarity.ports.middleware_ports import IMiddleware
- IMLModelService      → from clarity.ports.ml_ports import IMLModelService

🎯 All existing imports have been updated. This file provides backward 
   compatibility only and will be removed once all tests are verified.
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
