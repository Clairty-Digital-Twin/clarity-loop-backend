"""WebSocket API module for real-time communication."""

# removed – breaks FastAPI

from clarity.api.v1.websocket.chat_handler import router as chat_router

__all__ = ["chat_router"]
