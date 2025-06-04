"""WebSocket API module for real-time communication."""

from .chat_handler import router as chat_router

__all__ = ["chat_router"]