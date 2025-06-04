"""WebSocket API module for real-time communication."""

from clarity.api.v1.websocket.chat_handler import router as chat_router

__all__ = ["chat_router"]
