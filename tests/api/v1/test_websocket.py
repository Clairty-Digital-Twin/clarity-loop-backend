"""WebSocket API endpoint tests for the CLARITY chat system."""

import logging
from datetime import datetime, UTC
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
import pytest
from starlette.websockets import WebSocket, WebSocketDisconnect

from clarity.api.v1.websocket import chat_handler
from clarity.api.v1.websocket.connection_manager import ConnectionManager
from clarity.api.v1.websocket.lifespan import get_connection_manager
from clarity.api.v1.websocket.models import (
    ChatMessage,
    ConnectionInfo,
    HeartbeatMessage,
    MessageType,
    TypingMessage,
)
from clarity.auth.firebase_auth import get_current_user_websocket
from clarity.ml.gemini_service import GeminiService
from clarity.ml.pat_service import PATModelService
from clarity.models.user import User

logger = logging.getLogger(__name__)

# Test constants
TEST_USER_ID = "test-user-123"
TEST_TOKEN = "test-token"  # noqa: S105  # This is a test token, not a real secret
MOCK_FIREBASE_TOKEN = "mock-firebase-token"  # noqa: S105  # This is a test token


class _TestConnectionInfo(ConnectionInfo):
    """Extended ConnectionInfo for testing purposes."""

    room_id: str
    last_active: datetime = Field(default_factory=datetime.utcnow)
    last_heartbeat_ack: datetime = Field(default_factory=datetime.utcnow)
    message_timestamps: list[datetime] = Field(default_factory=list)


class _TestConnectionManager:
    """Test implementation of ConnectionManager for WebSocket testing."""

    def __init__(
        self,
        heartbeat_interval: int = 30,
        max_connections_per_user: int = 10,
        connection_timeout: int = 300,
        message_rate_limit_count: int = 100,
        message_rate_limit_period_seconds: int = 1,
        max_message_size: int = 64 * 1024,
    ) -> None:
        """Initialize test connection manager with configurable parameters."""
        # Connection state
        self.active_connections: set[WebSocket] = set()
        self.connection_info: dict[WebSocket, _TestConnectionInfo] = {}
        self.user_connections: dict[str, set[WebSocket]] = {}
        self.room_connections: dict[str, set[WebSocket]] = {}

        # Configuration
        self.heartbeat_interval = heartbeat_interval
        self.max_connections_per_user = max_connections_per_user
        self.connection_timeout = connection_timeout
        self.message_rate_limit_count = message_rate_limit_count
        self.message_rate_limit_period_seconds = message_rate_limit_period_seconds
        self.max_message_size = max_message_size

        # Test helpers
        self.sent_messages: list[dict[str, Any]] = []
        self.connection_log: list[str] = []

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        username: str,
        room_id: str = "general",
    ) -> _TestConnectionInfo:
        """Connect a WebSocket and store connection info."""
        connection_info = _TestConnectionInfo(
            user_id=user_id,
            username=username,
            session_id=f"test-session-{user_id}",
            room_id=room_id,
            connected_at=datetime.now(UTC),
        )

        # Add to our tracking structures
        self.active_connections.add(websocket)
        self.connection_info[websocket] = connection_info

        # Track by user
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(websocket)

        # Track by room
        if room_id not in self.room_connections:
            self.room_connections[room_id] = set()
        self.room_connections[room_id].add(websocket)

        logger.info(
            "Test connection established: user=%s, room=%s", username, room_id
        )
        self.connection_log.append(f"Connected: {username} to {room_id}")

        return connection_info

    async def disconnect(self, websocket: WebSocket, reason: str | None = None) -> None:
        """Disconnect a WebSocket and clean up."""
        if websocket not in self.active_connections:
            return

        connection_info = self.connection_info.get(websocket)
        if connection_info:
            # Remove from user connections
            user_id = connection_info.user_id
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(websocket)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]

            # Remove from room connections
            room_id = connection_info.room_id
            if room_id in self.room_connections:
                self.room_connections[room_id].discard(websocket)
                if not self.room_connections[room_id]:
                    del self.room_connections[room_id]

            logger.info(
                "Test connection disconnected: user=%s, reason=%s",
                connection_info.username,
                reason,
            )
            self.connection_log.append(
                f"Disconnected: {connection_info.username} ({reason or 'unknown'})"
            )

        # Clean up
        self.active_connections.discard(websocket)
        self.connection_info.pop(websocket, None)

    async def send_to_connection(self, websocket: WebSocket, message: object) -> None:
        """Send a message to a specific connection."""
        logger.info("Attempting to send message to connection: %s", message)

        # Convert message to dict if it's a Pydantic model
        if hasattr(message, "model_dump"):
            message_dict = message.model_dump(mode="json")  # type: ignore[attr-defined]
        else:
            message_dict = message

        # Log for test verification
        self.sent_messages.append({
            "target": "connection",
            "websocket": websocket,
            "message": message_dict,
        })

        # In a real implementation, this would send via WebSocket
        # For tests, we just log
        logger.info("Recorded direct message send to %s", websocket)

    async def send_to_user(self, user_id: str, message: object) -> None:
        """Send a message to all active connections for a given user."""
        logger.info("Attempting to send message to user %s: %s", user_id, message)

        # Convert message to dict if it's a Pydantic model
        if hasattr(message, "model_dump"):
            message_dict = message.model_dump(mode="json")  # type: ignore[attr-defined]
        else:
            message_dict = message

        # Log for test verification
        self.sent_messages.append({
            "target": "user",
            "user_id": user_id,
            "message": message_dict,
        })

        user_connections = self.user_connections.get(user_id, set())
        logger.info(
            "Found %d connections for user %s", len(user_connections), user_id
        )

        # In a real implementation, this would send to all user connections
        for websocket in user_connections:
            await self.send_to_connection(websocket, message)

    async def broadcast_to_room(
        self, room_id: str, message: object, exclude_websocket: WebSocket | None = None
    ) -> None:
        """Broadcast a message to all connections in a room."""
        logger.info("Attempting to broadcast message to room %s: %s", room_id, message)

        # Convert message to dict if it's a Pydantic model
        if hasattr(message, "model_dump"):
            message_dict = message.model_dump(mode="json")  # type: ignore[attr-defined]
        else:
            message_dict = message

        # Log for test verification
        self.sent_messages.append({
            "target": "room",
            "room_id": room_id,
            "message": message_dict,
            "exclude_websocket": exclude_websocket,
        })

        room_connections = self.room_connections.get(room_id, set())
        target_connections = {
            ws for ws in room_connections if ws != exclude_websocket
        }

        logger.info(
            "Broadcasting to %d connections in room %s", len(target_connections), room_id
        )

        # In a real implementation, this would send to all room connections
        for websocket in target_connections:
            await self.send_to_connection(websocket, message)

    async def handle_heartbeat(self, websocket: WebSocket) -> None:
        """Handle heartbeat for a connection."""
        connection_info = self.connection_info.get(websocket)
        if connection_info:
            connection_info.last_heartbeat_ack = datetime.now(UTC)
            logger.info("Heartbeat handled for connection: %s", connection_info.username)

    def is_rate_limited(self, websocket: WebSocket) -> bool:
        """Check if a connection is rate limited."""
        # For tests, never rate limit
        return False

    @staticmethod
    def handle_message(_websocket: WebSocket, _raw_message: str) -> bool:
        """Handle an incoming WebSocket message - always allow in tests."""
        # In tests, always return True to allow message processing
        return True

    def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get user information from connections."""
        connections = self.user_connections.get(user_id, set())
        if not connections:
            return None

        # Return info from first connection
        websocket = next(iter(connections))
        connection_info = self.connection_info.get(websocket)
        if connection_info:
            return {
                "user_id": connection_info.user_id,
                "username": connection_info.username,
                "room_id": connection_info.room_id,
                "connected_at": connection_info.connected_at,
            }
        return None

    def get_room_users(self, room_id: str) -> set[str]:
        """Get set of user IDs in a room."""
        connections = self.room_connections.get(room_id, set())
        return {
            self.connection_info[ws].user_id
            for ws in connections
            if ws in self.connection_info
        }

    def get_user_count(self) -> int:
        """Get total number of unique users connected."""
        return len(self.user_connections)

    def get_room_user_count(self, room_id: str) -> int:
        """Get number of users in a specific room."""
        return len(self.get_room_users(room_id))

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.active_connections)

    def get_connection_info_for_websocket(
        self, websocket: WebSocket
    ) -> _TestConnectionInfo | None:
        """Get connection info for a specific websocket."""
        return self.connection_info.get(websocket)

    async def handle_message(self, _websocket: WebSocket, _raw_message: str) -> bool:
        """Handle an incoming WebSocket message - always allow in tests."""
        # In tests, always return True to allow message processing
        return True


@pytest.fixture
def mock_test_connection_manager() -> _TestConnectionManager:
    """Create a test connection manager."""
    return _TestConnectionManager()


def create_mock_connection_manager() -> _TestConnectionManager:
    """Helper to create a _TestConnectionManager instance."""
    return _TestConnectionManager()


def mock_get_current_user_websocket(token: str) -> User:
    """Mock for get_current_user_websocket dependency."""
    if token == TEST_TOKEN:  # noqa: S105  # This is a test comparison
        return User(
            uid=TEST_USER_ID,
            email="test@example.com",
            display_name="Test User",
            firebase_token=MOCK_FIREBASE_TOKEN,
            created_at=datetime.now(UTC),
            last_login=datetime.now(UTC),
            profile={},
        )
    raise HTTPException(status_code=401, detail="Could not validate credentials")


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Create FastAPI app for testing."""
    # This fixture should return a FastAPI app instance configured for testing.
    # It needs to override dependencies to use our mocks.
    app = FastAPI()

    # Include the chat router
    app.include_router(chat_handler.router, prefix="/api/v1")

    # Override dependencies for testing
    app.dependency_overrides[get_current_user_websocket] = (
        mock_get_current_user_websocket
    )
    # Create properly mocked GeminiService
    mock_gemini = AsyncMock(spec=GeminiService)

    # Create a mock response that returns dynamic content
    def mock_generate_insights(request: object) -> object:
        """Create mock insights response."""
        response = MagicMock()
        response.narrative = f"AI Response to: {getattr(request, 'context', 'unknown')}"
        return response

    mock_gemini.generate_health_insights = mock_generate_insights
    app.dependency_overrides[chat_handler.get_gemini_service] = lambda: mock_gemini
    app.dependency_overrides[chat_handler.get_pat_model_service] = lambda: AsyncMock(
        spec=PATModelService
    )
    app.dependency_overrides[get_connection_manager] = create_mock_connection_manager

    # GeminiService is mocked via dependency override above

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_get_connection_manager():
    """Mock connection manager fixture."""
    # This fixture provides a MagicMock for the ConnectionManager.
    # It's used in tests that directly patch get_connection_manager.
    with MagicMock(spec=ConnectionManager) as mock_manager:
        yield mock_manager


@pytest.fixture
def connection_manager():
    """Real connection manager fixture."""
    # This fixture provides an instance of the actual ConnectionManager for tests
    # that need to interact with it directly.
    return ConnectionManager()


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self) -> None:
        self.sent_data: list[str] = []
        self.received_data: list[str] = []
        self.closed: bool = False

    async def accept(self) -> None:
        """Mock accept method."""

    async def send_text(self, data: str) -> None:
        """Mock send_text method."""
        self.sent_data.append(data)

    async def close(self, _code: int = 1000, _reason: str = "") -> None:
        """Mock close method."""
        self.closed = True


@pytest.mark.asyncio
class TestWebSocketEndpoints:
    """WebSocket endpoint tests."""

    @staticmethod
    async def test_websocket_chat_endpoint_authenticated(
        client: TestClient,
    ) -> None:
        """Test authenticated WebSocket chat endpoint."""
        user_id = TEST_USER_ID
        test_token = TEST_TOKEN

        with client.websocket_connect(
            f"/api/v1/chat/{user_id}?token={test_token}"
        ) as websocket:

            # Send a chat message
            chat_message = ChatMessage(
                user_id=user_id,
                timestamp=datetime.now(UTC),
                type=MessageType.MESSAGE,
                content="Hello AI",
            )
            websocket.send_json(chat_message.model_dump(mode="json"))

            # Expecting a response from the AI handler
            response_data = websocket.receive_json()

            assert response_data["type"] == MessageType.MESSAGE.value
            assert "AI Response to: Hello AI" in response_data["content"]
            assert response_data["user_id"] == "AI"

            # Test typing indicator
            typing_indicator = TypingMessage(
                user_id=user_id,
                timestamp=datetime.now(UTC),
                is_typing=True,
                type=MessageType.TYPING,
                username="test-user",
            )
            websocket.send_json(typing_indicator.model_dump(mode="json"))

            response_data = websocket.receive_json()
            assert response_data["type"] == MessageType.TYPING.value
            assert response_data["user_id"] == user_id
            assert response_data["is_typing"] is True

            # Test heartbeat
            heartbeat_message = HeartbeatMessage(
                timestamp=datetime.now(UTC),
                type=MessageType.HEARTBEAT,
            )
            websocket.send_json(heartbeat_message.model_dump(mode="json"))

            response_data = websocket.receive_json()
            assert response_data["type"] == MessageType.HEARTBEAT_ACK.value

    @staticmethod
    async def test_websocket_chat_endpoint_anonymous(client: TestClient) -> None:
        """Test anonymous WebSocket connection (should fail)."""
        user_id = "anonymous-user-123"
        with (
            pytest.raises(WebSocketDisconnect) as excinfo,
            client.websocket_connect(f"/api/v1/chat/{user_id}") as websocket,
        ):
            # This connection should be rejected by the auth middleware
            websocket.send_text("Hello")
        assert excinfo.value.code == 1008  # Policy Violation

    @staticmethod
    async def test_websocket_invalid_message_format(client: TestClient) -> None:
        """Test WebSocket with invalid message format."""
        user_id = TEST_USER_ID
        test_token = TEST_TOKEN

        with client.websocket_connect(
            f"/api/v1/chat/{user_id}?token={test_token}"
        ) as websocket:
            # Send an invalid message format
            websocket.send_text("this is not json")

            response_data = websocket.receive_json()
            assert response_data["type"] == MessageType.ERROR.value
            assert "Invalid JSON format" in response_data["message"]

    @staticmethod
    async def test_websocket_typing_indicator(client: TestClient) -> None:
        """Test WebSocket typing indicator functionality."""
        user_id = TEST_USER_ID
        test_token = TEST_TOKEN

        with client.websocket_connect(
            f"/api/v1/chat/{user_id}?token={test_token}"
        ) as websocket:
            # Send a typing indicator message
            typing_message = TypingMessage(
                user_id=user_id,
                username="test-user",
                timestamp=datetime.now(UTC),
                is_typing=True,
                type=MessageType.TYPING,
            )
            websocket.send_json(typing_message.model_dump(mode="json"))

            # Expect a typing indicator response
            response_data = websocket.receive_json()
            assert response_data["type"] == MessageType.TYPING.value
            assert response_data["user_id"] == user_id
            assert response_data["is_typing"] is True

            # Send another typing indicator, now indicating not typing
            typing_message.is_typing = False
            websocket.send_json(typing_message.model_dump(mode="json"))

            # Expect a typing indicator response
            response_data = websocket.receive_json()
            assert response_data["type"] == MessageType.TYPING.value
            assert response_data["user_id"] == user_id
            assert response_data["is_typing"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
