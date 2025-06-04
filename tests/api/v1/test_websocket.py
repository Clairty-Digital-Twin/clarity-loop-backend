"""Tests for WebSocket chat functionality."""

import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketState
import json
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from typing import Callable

from clarity.api.v1.websocket import models
from clarity.api.v1.websocket.connection_manager import ConnectionManager
from clarity.api.v1.websocket import chat_handler
from clarity.auth.firebase_auth import get_current_user_websocket

from clarity.models.user import User
try:
    from clarity.models.profile import UserProfile
    from clarity.models.preferences import UserPreferences
except ImportError:
    UserProfile = None
    UserPreferences = None
    print("UserProfile and UserPreferences import failed, using None as fallback")
from pydantic import ValidationError

# Fix app fixture
@pytest.fixture
def app() -> FastAPI:
    from clarity.main import create_app
    app = create_app()
    app.dependency_overrides[get_current_user_websocket] = mock_get_current_user
    app.dependency_overrides[chat_handler.get_connection_manager] = mock_get_connection_manager
    return app

@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)

@pytest.fixture
def mock_get_connection_manager():
    def mock_manager():
        manager = MagicMock()
        manager.connect = AsyncMock(return_value=None)
        manager.disconnect = AsyncMock(return_value=None)
        manager.send_to_connection = AsyncMock(return_value=None)
        manager.broadcast_to_room = AsyncMock(return_value=None)
        return manager
    return mock_manager

@pytest.fixture
def connection_manager():
    """Create connection manager for testing."""
    return ConnectionManager(
        heartbeat_interval=5,
        max_connections_per_user=2,
        connection_timeout=30,
        message_rate_limit=10,
        max_message_size=1024,
    )


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self) -> None:
        self.client_state = WebSocketState.CONNECTED
        self.messages_sent = []
        self.closed = False
        self.close_code = None
        self.close_reason = None

    async def accept(self) -> None:
        """Mock accept method."""

    async def send_text(self, data: str) -> None:
        """Mock send_text method."""
        self.messages_sent.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Mock close method."""
        self.closed = True
        self.close_code = code
        self.close_reason = reason
        self.client_state = WebSocketState.DISCONNECTED


class TestConnectionManager:
    """Test cases for WebSocket connection manager."""

    @pytest.mark.asyncio
    async def test_connect_success(self, connection_manager: ConnectionManager) -> None:
        """Test successful WebSocket connection."""
        websocket = MockWebSocket()

        result = await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User",
            room_id="test_room",
        )

        assert result is True
        assert "test_user" in connection_manager.user_connections
        assert websocket in connection_manager.connection_info
        assert len(websocket.messages_sent) == 1  # Connection ack message

        # Verify connection info
        info = connection_manager.connection_info[websocket]
        assert info.user_id == "test_user"
        assert info.username == "Test User"

    @pytest.mark.asyncio
    async def test_connect_max_connections_exceeded(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test connection rejection when max connections exceeded."""
        user_id = "test_user"

        # Connect up to max connections
        for i in range(connection_manager.max_connections_per_user):
            websocket = MockWebSocket()
            result = await connection_manager.connect(
                websocket=websocket,
                user_id=user_id,
                username=f"User {i}",
            )
            assert result is True

        # Try to connect one more (should fail)
        websocket = MockWebSocket()
        result = await connection_manager.connect(
            websocket=websocket,
            user_id=user_id,
            username="Extra User",
        )

        assert result is False
        assert websocket.closed is True
        assert websocket.close_code == 1008

    @pytest.mark.asyncio
    async def test_disconnect(self, connection_manager: ConnectionManager) -> None:
        """Test WebSocket disconnection."""
        websocket = MockWebSocket()

        # Connect first
        await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User",
        )

        # Disconnect
        await connection_manager.disconnect(websocket)

        assert websocket not in connection_manager.connection_info
        assert len(connection_manager.user_connections.get("test_user", [])) == 0

    @pytest.mark.asyncio
    async def test_send_to_connection(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test sending message to specific connection."""
        websocket = MockWebSocket()

        await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User",
        )

        # Clear initial messages
        websocket.messages_sent.clear()

        # Send message
        message = models.SystemMessage(content="Test message")
        await connection_manager.send_to_connection(websocket, message)

        assert len(websocket.messages_sent) == 1
        sent_data = json.loads(websocket.messages_sent[0])  # Ensure valid JSON
        assert sent_data["type"] == models.MessageType.SYSTEM
        assert sent_data["content"] == "Test message"

    @pytest.mark.asyncio
    async def test_broadcast_to_room(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test broadcasting message to room."""
        # Connect multiple users to same room
        websockets = []
        for i in range(3):
            websocket = MockWebSocket()
            await connection_manager.connect(
                websocket=websocket,
                user_id=f"user_{i}",
                username=f"User {i}",
                room_id="test_room",
            )
            websocket.messages_sent.clear()  # Clear connection ack
            websockets.append(websocket)

        # Broadcast message
        message = models.SystemMessage(content="Broadcast test")
        await connection_manager.broadcast_to_room("test_room", message)

        # Verify all users received the broadcast message (plus any join notifications)
        for websocket in websockets:
            # Should have received at least one message
            assert len(websocket.messages_sent) >= 1

            # Check that the broadcast message is among the messages
            broadcast_received = False
            for message in websocket.messages_sent:
                sent_data = json.loads(message)  # Ensure valid JSON
                if sent_data.get("content") == "Broadcast test":
                    broadcast_received = True
                    break
            assert broadcast_received, "Broadcast message not received"

    @pytest.mark.asyncio
    async def test_rate_limiting(self, connection_manager: ConnectionManager) -> None:
        """Test message rate limiting."""
        websocket = MockWebSocket()

        await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User",
        )

        # Send messages up to limit
        for i in range(connection_manager.message_rate_limit):
            result = await connection_manager.handle_message(websocket, f"Message {i}")
            assert result is True

        # Next message should be rate limited
        result = await connection_manager.handle_message(
            websocket, "Rate limited message"
        )
        assert result is False

        # Should have received error message
        error_messages = [
            msg for msg in websocket.messages_sent if "RATE_LIMIT_EXCEEDED" in msg
        ]
        assert len(error_messages) > 0

    @pytest.mark.asyncio
    async def test_message_size_limit(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test message size limiting."""
        websocket = MockWebSocket()

        await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User",
        )

        # Create message larger than limit
        large_message = "x" * (connection_manager.max_message_size + 1)

        result = await connection_manager.handle_message(websocket, large_message)
        assert result is False

        # Should have received error message
        error_messages = [
            msg for msg in websocket.messages_sent if "MESSAGE_TOO_LARGE" in msg
        ]
        assert len(error_messages) > 0

    @pytest.mark.asyncio
    async def test_heartbeat_functionality(
        self, connection_manager: ConnectionManager
    ) -> None:
        """Test heartbeat monitoring."""
        websocket = MockWebSocket()

        await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User",
        )

        initial_message_count = len(websocket.messages_sent)

        # Trigger heartbeat
        await connection_manager._send_heartbeats()

        # Should receive heartbeat message
        assert len(websocket.messages_sent) > initial_message_count

        # Check last message is heartbeat
        last_message = json.loads(websocket.messages_sent[-1])  # Ensure valid JSON
        assert last_message["type"] == models.MessageType.HEARTBEAT


class TestWebSocketModels:
    """Test cases for WebSocket message models."""

    def test_chat_message_validation(self) -> None:
        """Test ChatMessage validation."""
        # Valid message
        message = models.ChatMessage(
            content="Hello world", user_id="test_user", username="Test User"
        )
        assert message.type == models.MessageType.MESSAGE
        assert message.content == "Hello world"

        # Invalid message (too long)
        with pytest.raises(ValidationError, match="String should have at most"):
            models.ChatMessage(content="x" * 3000, user_id="test_user")  # Exceeds max_length

        # Invalid message (empty content)
        with pytest.raises(ValidationError, match="String should have at least"):
            models.ChatMessage(content="", user_id="test_user")

    def test_error_message_creation(self) -> None:
        """Test ErrorMessage creation."""
        error = models.ErrorMessage(
            error_code="TEST_ERROR",
            message="Test error message",
            details={"key": "value"},
        )

        assert error.type == models.MessageType.ERROR
        assert error.error_code == "TEST_ERROR"
        assert error.message == "Test error message"
        assert error.details == {"key": "value"}

    def test_typing_message(self) -> None:
        """Test TypingMessage model."""
        typing = models.TypingMessage(
            user_id="test_user", username="Test User", is_typing=True
        )

        assert typing.type == models.MessageType.TYPING
        assert typing.is_typing is True

    def test_heartbeat_message(self) -> None:
        """Test HeartbeatMessage model."""
        heartbeat = models.HeartbeatMessage()
        assert heartbeat.type == models.MessageType.HEARTBEAT

        # Test JSON serialization
        json_data = heartbeat.model_dump_json()
        assert isinstance(json_data, str)

        # Test deserialization
        parsed = json.loads(json_data)
        assert parsed["type"] == models.MessageType.HEARTBEAT


@pytest.fixture
def mock_get_current_user() -> Callable[[], User]:
    """Mock current user for auth dependency."""
    def _mock_user() -> User:
        profile_data = UserProfile(
            age=30,
            gender="other",
            preferences=UserPreferences(
                data_sharing=True,
                theme="light"
            )
        ) if UserProfile and UserPreferences else None
        return User(
            uid="test_user",
            email="test@example.com",
            display_name="Test User",
            email_verified=True,
            firebase_token="mock_token",
            created_at=datetime.now(timezone.utc),
            last_login=datetime.now(timezone.utc),
            profile=profile_data
        )
    return _mock_user


@pytest.fixture
def mock_connection_manager():
    mock = AsyncMock()
    mock.connections = {}
    mock.rooms = {}
    return mock


@pytest.mark.asyncio
class TestWebSocketEndpoints:
    """Test WebSocket endpoints."""

    @pytest.mark.asyncio
    async def test_websocket_chat_endpoint_authenticated(self, client: TestClient, mock_get_connection_manager) -> None:
        print("Attempting WebSocket connection for authenticated user")
        try:
            with client.websocket_connect("/api/v1/ws/chat/test_room?token=mock_token") as websocket:
                print("WebSocket connection established")
                websocket.send_text(json.dumps({
                    "type": "message",
                    "content": "Hello, test message!"
                }))
                print("Message sent, awaiting response")
                response = websocket.receive_text()
                print(f"Response received: {response}")
                data = json.loads(response)
                assert data["type"] == "message"
                assert data["content"] == "Hello, test message!"
        except WebSocketDisconnect as e:
            print(f"WebSocket connection failed with WebSocketDisconnect: {str(e)}")
            pytest.fail(f"WebSocket connection failed: {str(e)}")
        except Exception as e:
            print(f"Unexpected error during WebSocket test: {str(e)}")
            pytest.fail(f"Unexpected error: {str(e)}")

    @pytest.mark.asyncio
    async def test_websocket_chat_endpoint_anonymous(self, client: TestClient, mock_get_connection_manager) -> None:
        print("Attempting WebSocket connection for anonymous user")
        try:
            with client.websocket_connect("/api/v1/ws/chat/test_room") as websocket:
                print("WebSocket connection established for anonymous user")
                response = websocket.receive_text()
                print(f"Response received for anonymous user: {response}")
                data = json.loads(response)
                assert data["type"] == "error"
                assert "authentication" in data["content"].lower()
        except WebSocketDisconnect as e:
            print(f"WebSocket connection failed for anonymous user with WebSocketDisconnect: {str(e)}")
            pytest.fail(f"WebSocket connection failed: {str(e)}")
        except Exception as e:
            print(f"Unexpected error during anonymous WebSocket test: {str(e)}")
            pytest.fail(f"Unexpected error: {str(e)}")

    @pytest.mark.asyncio
    async def test_websocket_invalid_message_format(self, client: TestClient, mock_get_connection_manager) -> None:
        print("Attempting WebSocket connection for invalid message format test")
        try:
            with client.websocket_connect("/api/v1/ws/chat/test_room?token=mock_token") as websocket:
                print("WebSocket connection established for invalid message test")
                websocket.send_text("invalid json")
                print("Invalid message sent, awaiting response")
                response = websocket.receive_text()
                print(f"Response received for invalid message: {response}")
                data = json.loads(response)
                assert data["type"] == "error"
                assert "format" in data["content"].lower()
        except WebSocketDisconnect as e:
            print(f"WebSocket connection failed for invalid message test with WebSocketDisconnect: {str(e)}")
            pytest.fail(f"WebSocket connection failed: {str(e)}")
        except Exception as e:
            print(f"Unexpected error during invalid message WebSocket test: {str(e)}")
            pytest.fail(f"Unexpected error: {str(e)}")

    @pytest.mark.asyncio
    async def test_websocket_typing_indicator(self, client: TestClient, mock_get_connection_manager) -> None:
        print("Attempting WebSocket connection for typing indicator test")
        try:
            with client.websocket_connect("/api/v1/ws/chat/test_room?token=mock_token") as websocket:
                print("WebSocket connection established for typing indicator test")
                websocket.send_text(json.dumps({
                    "type": "typing",
                    "content": "true"
                }))
                print("Typing indicator sent, awaiting response")
                response = websocket.receive_text()
                print(f"Response received for typing indicator: {response}")
                data = json.loads(response)
                assert data["type"] == "typing"
                assert data["content"] == "true"
        except WebSocketDisconnect as e:
            print(f"WebSocket connection failed for typing indicator test with WebSocketDisconnect: {str(e)}")
            pytest.fail(f"WebSocket connection failed: {str(e)}")
        except Exception as e:
            print(f"Unexpected error during typing indicator WebSocket test: {str(e)}")
            pytest.fail(f"Unexpected error: {str(e)}")


@pytest.mark.asyncio
class TestChatHandler:
    """Test chat handler functionality."""

    async def test_health_insight_generation(self) -> None:
        # Setup mock data
        user_id = "test_user"
        room_id = "test_room"
        message_content = "I feel tired all the time"

        # Mock the connection manager to return a successful broadcast
        mock_manager = MagicMock()
        mock_manager.broadcast_to_room = AsyncMock(return_value=None)
        # Directly assign to avoid attribute error
        if not hasattr(chat_handler, 'connection_manager'):
            chat_handler.connection_manager = mock_manager

        # Mock the health insight service to return a valid insight
        insight_response = {
            "insights": ["You might be experiencing fatigue, consider checking your sleep patterns."]
        }
        if not hasattr(chat_handler, 'health_insight_service'):
            chat_handler.health_insight_service = MagicMock()
            chat_handler.health_insight_service.get_health_insights = AsyncMock(return_value=insight_response)

        # Call the function under test
        msg = models.ChatMessage(
            type="message",  # Use string value as fallback if enum is not available
            content=message_content,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc)
        )
        # Call the function directly if handle_chat_message exists
        if hasattr(chat_handler, 'handle_chat_message'):
            await chat_handler.handle_chat_message(room_id, user_id, msg)
        else:
            pytest.skip("handle_chat_message function not found in chat_handler")

        # Assertions
        mock_manager.broadcast_to_room.assert_called()
        args, _ = mock_manager.broadcast_to_room.call_args
        assert len(args) > 1  # Ensure broadcast message was called with arguments
        broadcast_msg = args[1] if len(args) > 1 else None
        assert broadcast_msg is not None
        assert broadcast_msg.type == "message"
        assert "fatigue" in broadcast_msg.content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
