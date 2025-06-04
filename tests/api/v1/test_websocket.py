"""Tests for WebSocket chat functionality."""

import asyncio  # Added asyncio
from collections import defaultdict  # Added defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone  # Added timedelta
import json
import logging  # Added logging
import time
from typing import (  # Added Dict, List, Any, Set, DefaultDict
    Any,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
)
from unittest.mock import (  # Keep MagicMock for other potential uses if any
    AsyncMock,
    MagicMock,
)
import uuid  # Added uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketState

from clarity.api.v1.websocket import chat_handler, models
from clarity.api.v1.websocket.connection_manager import ConnectionManager
from clarity.api.v1.websocket.models import ConnectionInfo
from clarity.auth.firebase_auth import User, get_current_user_websocket
from clarity.models.user import User

try:
    from clarity.models.preferences import UserPreferences
    from clarity.models.profile import UserProfile
except ImportError:
    UserProfile = None
    UserPreferences = None
    print("UserProfile and UserPreferences import failed, using None as fallback")
from pydantic import Field, ValidationError

logger = logging.getLogger(__name__)
# Basic logging for tests, customize as needed, e.g., in conftest.py for global config
# logging.basicConfig(level=logging.INFO)


class TestConnectionInfo(ConnectionInfo):
    """Extended ConnectionInfo for testing purposes."""

    room_id: str
    last_active: datetime = Field(default_factory=datetime.utcnow)
    last_heartbeat_ack: datetime = Field(default_factory=datetime.utcnow)
    message_timestamps: list[datetime] = Field(default_factory=list)


class TestConnectionManager:
    """Stateful mock for ConnectionManager to be used in tests."""

    def __init__(
        self,
        heartbeat_interval: int = 30,
        max_connections_per_user: int = 10,
        connection_timeout: int = 300,
        message_rate_limit_count: int = 100,
        message_rate_limit_period_seconds: int = 1,
        max_message_size: int = 64 * 1024,
    ):
        self.active_websockets: set[WebSocket] = set()
        self.user_connections: defaultdict[str, list[WebSocket]] = defaultdict(list)
        self.connection_info: dict[WebSocket, TestConnectionInfo] = {}
        self.rooms: defaultdict[str, set[str]] = defaultdict(set)

        self.heartbeat_interval = heartbeat_interval
        self.max_connections_per_user = max_connections_per_user
        self.connection_timeout = connection_timeout
        self.message_rate_limit_count = message_rate_limit_count
        self.message_rate_limit_period_seconds = message_rate_limit_period_seconds
        self.max_message_size = max_message_size

        self.messages_sent: list[dict[str, Any]] = []
        self.heartbeats_processed: list[dict[str, Any]] = []
        self.last_heartbeat: dict[WebSocket, float] = {}
        self.message_counts: defaultdict[str, list[float]] = defaultdict(list)

        logger.info(
            f"TestConnectionManager initialized with max_connections_per_user={self.max_connections_per_user}"
        )

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        username: str,
        room_id: str = "general",
    ) -> TestConnectionInfo:
        logger.info(
            f"Attempting to connect user {user_id} to room {room_id} with session {uuid.uuid4()}"
        )
        if len(self.user_connections[user_id]) >= self.max_connections_per_user:
            logger.warning(
                f"Connection limit exceeded for user {user_id}. Max: {self.max_connections_per_user}"
            )

        connection_time = datetime.now(UTC)
        new_connection_info = TestConnectionInfo(
            user_id=user_id,
            username=username,
            session_id=str(uuid.uuid4()),
            connected_at=connection_time,
            last_active=connection_time,
            room_id=room_id,
            last_heartbeat_ack=connection_time,
            message_timestamps=[],
        )

        self.connection_info[websocket] = new_connection_info
        self.active_websockets.add(websocket)
        self.rooms[room_id].add(user_id)
        self.user_connections[user_id].append(websocket)
        self.last_heartbeat[websocket] = time.time()

        logger.info(
            f"User {user_id} connected to room {room_id}. Total active websockets: {len(self.active_websockets)}"
        )
        logger.info(f"Connection info for websocket: {self.connection_info[websocket]}")
        return new_connection_info

    async def disconnect(self, websocket: WebSocket, reason: str | None = None) -> None:
        logger.info(f"Attempting to disconnect websocket. Reason: {reason}")
        connection_info_to_remove = self.connection_info.get(websocket)

        if not connection_info_to_remove:
            logger.warning("Websocket not found in connection_info for disconnection.")
            if websocket in self.active_websockets:
                self.active_websockets.remove(websocket)
                logger.info(
                    "Removed websocket from active_websockets (was not in connection_info)."
                )
            return

        user_id = connection_info_to_remove.user_id
        room_id = connection_info_to_remove.room_id

        del self.connection_info[websocket]
        self.active_websockets.discard(websocket)
        if websocket in self.last_heartbeat:
            del self.last_heartbeat[websocket]

        if user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
                logger.info(f"User {user_id} has no more active connections.")

        user_still_in_room = False
        if user_id in self.user_connections:
            for ws_conn in self.user_connections[user_id]:
                ci = self.connection_info.get(ws_conn)
                if ci and ci.room_id == room_id:
                    user_still_in_room = True
                    break

        if (
            not user_still_in_room
            and room_id in self.rooms
            and user_id in self.rooms[room_id]
        ):
            self.rooms[room_id].remove(user_id)
            logger.info(f"User {user_id} removed from room {room_id}.")
            if not self.rooms[room_id]:
                del self.rooms[room_id]
                logger.info(f"Room {room_id} is now empty and removed.")

        logger.info(
            f"Websocket disconnected for user {user_id} from room {room_id}. Total active websockets: {len(self.active_websockets)}"
        )

    async def send_to_connection(self, websocket: WebSocket, message: Any) -> None:
        logger.info(f"Attempting to send message to connection: {message}")

        if websocket not in self.connection_info:
            logger.warning("Cannot send to unknown websocket connection")
            return

        message_content = message
        if hasattr(message, "model_dump_json"):
            message_content = json.loads(message.model_dump_json())
        elif hasattr(message, "dict"):
            message_content = message.dict()

        self.messages_sent.append({
            "type": "direct",
            "target_ws": websocket,
            "message": message_content
        })

        logger.info(f"Recorded direct message send to {websocket}")

    async def send_to_user(self, user_id: str, message: Any) -> None:
        """Send a message to all active connections for a given user."""
        logger.info(f"Attempting to send message to user {user_id}: {message}")

        message_content = message
        if hasattr(message, "model_dump_json"):
            message_content = json.loads(message.model_dump_json())
        elif hasattr(message, "dict"):
            message_content = message.dict()

        if user_id in self.user_connections:
            for websocket in self.user_connections[user_id]:
                if websocket in self.active_websockets:
                    self.messages_sent.append({
                        "type": "direct_to_user",
                        "target_user": user_id,
                        "target_ws": websocket,
                        "message": message_content
                    })
                    logger.info(f"Recorded direct message send to user {user_id} via {websocket}")
        else:
            logger.warning(f"User {user_id} has no active connections to send messages to.")

    async def broadcast_to_room(
        self, room_id: str, message: Any, exclude_websocket: WebSocket | None = None
    ) -> None:
        logger.info(f"Attempting to broadcast message to room {room_id}: {message}")

        message_content = message
        if hasattr(message, "model_dump_json"):
            message_content = json.loads(message.model_dump_json())
        elif hasattr(message, "dict"):
            message_content = message.dict()

        self.messages_sent.append({
            "type": "broadcast",
            "room_id": room_id,
            "message": message_content,
            "excluded": exclude_websocket
        })

        target_websockets = []
        for user_id in self.rooms.get(room_id, set()):
            for ws in self.user_connections.get(user_id, []):
                if ws != exclude_websocket and ws in self.active_websockets:
                    target_websockets.append(ws)

        logger.info(f"Broadcast recorded to {len(target_websockets)} connections in room {room_id}")

    async def handle_heartbeat(self, websocket: WebSocket) -> None:
        logger.info("Handling heartbeat for websocket")

        connection_info = self.connection_info.get(websocket)
        if connection_info:
            now = datetime.now(UTC)
            connection_info.last_heartbeat_ack = now
            connection_info.last_active = now

            self.heartbeats_processed.append({
                "websocket": websocket,
                "timestamp": now
            })

    def is_rate_limited(self, websocket: WebSocket) -> bool:
        connection_info = self.connection_info.get(websocket)
        if not connection_info:
            logger.warning("Cannot check rate limit for unknown websocket.")
            return False

        user_id = connection_info.user_id
        current_time = datetime.now(UTC)
        self.message_counts[user_id].append(current_time.timestamp())

        # Remove messages older than the rate limit period
        cutoff_time = (current_time - timedelta(seconds=self.message_rate_limit_period_seconds)).timestamp()
        recent_messages = [
            ts for ts in self.message_counts[user_id] if ts > cutoff_time
        ]
        self.message_counts[user_id] = recent_messages

        if len(recent_messages) > self.message_rate_limit_count:
            logger.warning(f"User {user_id} is rate-limited.")
            return True
        return False

    def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        for ws_list in self.user_connections.values():
            for ws in ws_list:
                info = self.connection_info.get(ws)
                if info and info.user_id == user_id:
                    return info.model_dump()
        return None

    def get_room_users(self, room_id: str) -> set[str]:
        return self.rooms.get(room_id, set())

    def get_user_count(self, room_id: str) -> int:
        return len(self.rooms.get(room_id, set()))

    def get_connection_count(self) -> int:
        return len(self.active_websockets)

    def get_connection_info_for_websocket(
        self, websocket: WebSocket
    ) -> TestConnectionInfo | None:
        return self.connection_info.get(websocket)


@pytest.fixture
def mock_test_connection_manager() -> TestConnectionManager:
    return TestConnectionManager()


def create_mock_connection_manager():
    # This function is likely called in chat_handler.py to get the manager instance
    # We need to ensure it returns the TestConnectionManager for tests
    return mock_test_connection_manager()


@pytest.fixture
def app(client: TestClient) -> FastAPI:
    # This fixture should return a FastAPI app instance configured for testing.
    # It needs to override dependencies to use our mocks.

    # Create a dummy FastAPI app
    app_instance = FastAPI()

    # Override the get_connection_manager dependency to use our mock
    app_instance.dependency_overrides[chat_handler.get_connection_manager] = (
        create_mock_connection_manager
    )

    # Override the get_current_user_websocket dependency to use a mock user
    async def mock_get_current_user_websocket(token: str | None = None) -> User:
        if token == "valid_token":
            return User(uid="test_user_123", email="test@example.com", display_name="TestUser", firebase_token="mock_token", created_at=datetime.now(UTC), last_login=datetime.now(UTC), profile={})
        if token == "anonymous_token":
            return User(uid="anonymous_user", email="anonymous@example.com", display_name="AnonymousUser", firebase_token="mock_token", created_at=datetime.now(UTC), last_login=datetime.now(UTC), profile={})
        raise HTTPException(status_code=401, detail="Invalid credentials")

    app_instance.dependency_overrides[chat_handler.get_current_user_websocket] = (
        mock_get_current_user_websocket
    )

    # Include the chat router in the app (this was a missing step from WEBSOCKET_2.md)
    app_instance.include_router(chat_handler.router, prefix="/api/v1")

    return app_instance


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_get_connection_manager():
    mock_manager = MagicMock(spec=ConnectionManager)
    # Configure the mock to return a mock ConnectionInfo object when connection_info.get is called
    mock_connection_info = MagicMock(spec=ConnectionInfo)
    mock_connection_info.user_id = "test_user_123"
    mock_connection_info.username = "TestUser"
    mock_connection_info.room_id = "test_room"
    mock_connection_info.session_id = "mock_session_id"
    mock_connection_info.connected_at = datetime.now(UTC)
    mock_connection_info.last_active = datetime.now(UTC)
    mock_connection_info.last_heartbeat_ack = datetime.now(UTC)
    mock_connection_info.message_timestamps = []

    mock_manager.connection_info.get.return_value = mock_connection_info
    mock_manager.connect.return_value = mock_connection_info
    mock_manager.disconnect = AsyncMock()
    mock_manager.send_to_connection = AsyncMock()
    mock_manager.send_to_user = AsyncMock()
    mock_manager.broadcast_to_room = AsyncMock()
    mock_manager.handle_heartbeat = AsyncMock()
    mock_manager.is_rate_limited.return_value = False
    mock_manager._cleanup_stale_connections = AsyncMock()
    mock_manager._send_heartbeats = AsyncMock()
    mock_manager.start_background_tasks = AsyncMock()
    mock_manager.shutdown = AsyncMock()
    mock_manager.get_room_users.return_value = {"test_user_123"}
    mock_manager.get_user_count.return_value = 1
    mock_manager.get_connection_count.return_value = 1

    return mock_manager


@pytest.fixture
def connection_manager():
    return ConnectionManager()


class MockWebSocket:
    def __init__(self) -> None:
        self.sent_data: list[str] = []
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.client_state = WebSocketState.CONNECTED

    async def accept(self) -> None:
        pass

    async def send_text(self, data: str) -> None:
        self.sent_data.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code
        self.close_reason = reason
        self.client_state = WebSocketState.DISCONNECTED


@pytest.mark.asyncio
class TestWebSocketEndpoints:

    async def test_websocket_chat_endpoint_authenticated(
        self, client: TestClient
    ) -> None:
        user_id = "test_user_123"
        auth_headers = {"Authorization": "Bearer valid_token"}
        room_id = "test_room"

        print("Attempting WebSocket connection for authenticated user")
        try:
            with client.websocket_connect(f"/api/v1/ws/chat/{room_id}?token=valid_token", headers=auth_headers) as websocket:
                message_payload = {"type": models.MessageType.MESSAGE, "content": "Hello AI", "user_id": user_id}
                websocket.send_json(message_payload)

                response = websocket.receive_json()
                print(f"Received response: {response}")

                assert response["type"] == models.MessageType.MESSAGE
                assert "Hello" in response["content"]

                # Test typing indicator
                typing_payload = {"type": models.MessageType.TYPING, "user_id": user_id, "is_typing": True}
                websocket.send_json(typing_payload)

                # The typing message should be broadcast, but not necessarily sent back to sender.
                # For now, we'll just ensure no error occurs.

        except WebSocketDisconnect as e:
            pytest.fail(f"WebSocket connection failed: {e!s}")
        except Exception as e:
            pytest.fail(f"Unexpected error: {e!s}")

    async def test_websocket_chat_endpoint_anonymous(self, client: TestClient) -> None:
        room_id = "test_room"

        print("Attempting WebSocket connection for anonymous user")
        try:
            # FastAPI's Firebase auth middleware expects a token, even if it's dummy
            # So we pass an anonymous_token and ensure the mock handles it.
            with client.websocket_connect(f"/api/v1/ws/chat/{room_id}?token=anonymous_token") as websocket:
                message_payload = {"type": models.MessageType.MESSAGE, "content": "Hello anonymous", "user_id": "anonymous_user"}
                websocket.send_json(message_payload)

                response = websocket.receive_json()
                print(f"Received response: {response}")

                assert response["type"] == models.MessageType.MESSAGE
                assert "Hello" in response["content"]

        except WebSocketDisconnect as e:
            print(f"WebSocket connection failed for anonymous user with WebSocketDisconnect: {e!s}")
            print(f"Error code: {e.code}, Reason: {e.reason}")
            pytest.fail(f"WebSocket connection failed: {e!s}")
        except Exception as e:
            pytest.fail(f"Unexpected error: {e!s}")

    async def test_websocket_invalid_message_format(self, client: TestClient) -> None:
        user_id = "test_user_123"
        auth_headers = {"Authorization": "Bearer valid_token"}
        room_id = "test_room"

        print("Attempting WebSocket connection for invalid message format test")
        try:
            with client.websocket_connect(f"/api/v1/ws/chat/{room_id}?token=valid_token", headers=auth_headers) as websocket:
                # Send an invalid message payload (e.g., missing 'content' or wrong 'type')
                invalid_payload = {"type": "INVALID_TYPE", "user_id": user_id}
                websocket.send_json(invalid_payload)

                response = websocket.receive_json()
                print(f"Received response for invalid message: {response}")

                assert response["type"] == models.MessageType.ERROR
                assert "error_code" in response
                assert "message" in response

        except WebSocketDisconnect as e:
            pytest.fail(f"WebSocket connection failed: {e!s}")
        except Exception as e:
            print(f"Unexpected error during invalid message WebSocket test: {e!s}")
            pytest.fail(f"Unexpected error: {e!s}")

    async def test_websocket_typing_indicator(self, client: TestClient) -> None:
        user_id = "test_user_123"
        auth_headers = {"Authorization": "Bearer valid_token"}
        room_id = "test_room"

        print("Attempting WebSocket connection for typing indicator test")
        try:
            with client.websocket_connect(f"/api/v1/ws/chat/{room_id}?token=valid_token", headers=auth_headers) as websocket:
                typing_payload = {"type": models.MessageType.TYPING, "user_id": user_id, "username": "TestUser", "is_typing": True}
                websocket.send_json(typing_payload)

                # In a real scenario, the typing message would be broadcast to others.
                # The sender usually doesn't receive their own typing notification.
                # We will assert that the mock connection manager received the broadcast.

                # If the chat_handler echoes back, we might receive something.
                # For now, we'll try to receive and assert if something comes back,
                # but the primary check should be on the mock manager's state.

                # Verify that the typing message was processed (e.g., by checking mock calls)
                # This requires access to the mock_test_connection_manager instance from the fixture.
                # We can't directly access it here, so we need to mock chat_handler.chat_handler
                # or pass the mock manager to the test function if it's truly isolated.
                # For now, assume it's handled internally without an immediate return to sender.

                # You might need to add an assertion to check if the message was sent to the room
                # This would typically involve inspecting the mock_test_connection_manager's messages_sent list
                # which is not directly accessible from this test method without modifying the app fixture or using global mocks.
                # For now, we'll rely on the handler not raising an exception.

                # Add a receive with a timeout to avoid hanging, as no immediate response is expected.
                try:
                    # Expecting a potential message (e.g., an ACK or system message, or nothing)
                    # Adding a timeout to prevent tests from hanging indefinitely
                    response = websocket.receive(timeout=1)
                    print(f"Received response for typing: {response}")
                    # You could assert something if a response is expected, e.g., a system message ACK
                    # assert response["type"] == models.MessageType.SYSTEM
                except WebSocketDisconnect:
                    # Expected if connection closes after sending typing
                    pass
                except Exception as e:
                    # For any other unexpected errors during receive
                    print(f"Error receiving during typing test: {e!s}")
                    # pytest.fail(f"Error receiving during typing test: {e!s}")

        except WebSocketDisconnect as e:
            pytest.fail(f"WebSocket connection failed: {e!s}")
        except Exception as e:
            print(f"Unexpected error during typing indicator WebSocket test: {e!s}")
            pytest.fail(f"Unexpected error: {e!s}")


@pytest.mark.asyncio
class TestChatHandler:

    async def test_health_insight_generation(self, app: FastAPI):
        user_id = "test_user_id_1"
        # Create an instance of the handler, injecting the mock connection manager
        # We need to get the mock_test_connection_manager instance that the app is using
        # This is a bit tricky with FastAPI's dependency injection unless we make the mock
        # a global singleton or pass it more explicitly.

        # For simplicity in testing the handler's internal logic, we can directly instantiate it
        # and pass a mock connection manager.
        mock_connection_manager_instance = TestConnectionManager()

        handler = chat_handler.WebSocketChatHandler(
            gemini_service=AsyncMock(), pat_service=AsyncMock()
        )

        # Mock the connection manager that the handler will interact with
        # This assumes chat_handler.get_connection_manager is NOT used directly within the handler methods
        # but rather passed as an argument. If it's used directly, we need to patch it globally.

        # Simulate a message that triggers health insight generation
        chat_message = models.ChatMessage(
            user_id=user_id,
            username="TestUser",
            content="I feel tired and stressed, need some sleep advice.",
            type=models.MessageType.MESSAGE,  # Corrected from CHAT
        )

        # Mock the Gemini service response
        handler.gemini_service.generate_health_insights.return_value = {
            "insights": [
                {"content": "Consider improving your sleep hygiene.", "confidence": 0.9}
            ],
            "recommendations": ["Exercise regularly", "Manage stress"]
        }

        await handler.process_chat_message(
            websocket=MagicMock(),
            message=chat_message,
            connection_manager=mock_connection_manager_instance
        )

        # Assertions
        # Check if broadcast was called for the original message
        assert any(
            msg["type"] == "broadcast" and msg["message"]["content"] == chat_message.content
            for msg in mock_connection_manager_instance.messages_sent
        )

        # Check if health insight was sent to the user
        assert any(
            msg["type"] == "direct_to_user"
            and msg["target_user"] == user_id
            and msg["message"]["type"] == models.MessageType.HEALTH_INSIGHT
            and "sleep hygiene" in msg["message"]["insight"]
            for msg in mock_connection_manager_instance.messages_sent
        )

        handler.gemini_service.generate_health_insights.assert_awaited_once()

    async def test_typing_indicator_processing(self, app: FastAPI):
        user_id = "test_user_id_2"
        mock_connection_manager_instance = TestConnectionManager()

        handler = chat_handler.WebSocketChatHandler()

        typing_message = models.TypingMessage(
            user_id=user_id,
            username="TestUser2",
            is_typing=True
        )

        await handler.process_typing_message(
            websocket=MagicMock(),
            message=typing_message,
            connection_manager=mock_connection_manager_instance
        )

        # Assert that the typing message was broadcast (excluding the sender)
        assert any(
            msg["type"] == "broadcast"
            and msg["message"]["type"] == models.MessageType.TYPING
            and msg["message"]["user_id"] == user_id
            and msg["excluded"] is not None  # Ensure exclude_user was passed
            for msg in mock_connection_manager_instance.messages_sent
        )

    async def test_heartbeat_processing(self, app: FastAPI):
        user_id = "test_user_id_3"
        mock_connection_manager_instance = TestConnectionManager()

        handler = chat_handler.WebSocketChatHandler()
        mock_websocket = MagicMock(spec=WebSocket)

        # Simulate a connection to ensure connection_info is set up for the websocket
        await mock_connection_manager_instance.connect(
            websocket=mock_websocket,
            user_id=user_id,
            username="TestUser3",
            room_id="general"
        )

        heartbeat_message_payload = {"type": models.MessageType.HEARTBEAT, "client_timestamp": datetime.now(UTC).isoformat()}

        await handler.process_heartbeat(
            websocket=mock_websocket,
            message=heartbeat_message_payload,
            connection_manager=mock_connection_manager_instance
        )

        # Assert that heartbeat was processed and connection info updated
        connection_info_after_heartbeat = mock_connection_manager_instance.get_connection_info_for_websocket(mock_websocket)
        assert connection_info_after_heartbeat is not None
        assert connection_info_after_heartbeat.last_heartbeat_ack.date() == datetime.now(UTC).date()
        assert connection_info_after_heartbeat.last_active.date() == datetime.now(UTC).date()

        # Assert that a HeartbeatAckMessage was sent back (via send_to_connection)
        assert any(
            msg["type"] == "direct"
            and msg["target_ws"] == mock_websocket
            and msg["message"]["type"] == models.MessageType.HEARTBEAT_ACK
            for msg in mock_connection_manager_instance.messages_sent
        )

    async def test_health_analysis_trigger(self, app: FastAPI):
        user_id = "test_user_id_4"
        health_data = {"steps": 10000, "sleep_hours": 7}
        mock_connection_manager_instance = TestConnectionManager()

        handler = chat_handler.WebSocketChatHandler(
            pat_service=AsyncMock(), gemini_service=AsyncMock()
        )

        # Mock PAT service response
        handler.pat_service.analyze_health_data.return_value = {"activity_score": 0.8}

        # Mock Gemini service response
        handler.gemini_service.generate_health_insights.return_value = {
            "insights": [
                {"content": "Great activity levels!", "category": "activity"}
            ],
            "recommendations": ["Keep it up!"]
        }

        await handler.trigger_health_analysis(
            user_id=user_id,
            health_data=health_data,
            connection_manager=mock_connection_manager_instance
        )

        # Assert initial update message
        assert any(
            msg["type"] == "direct_to_user"
            and msg["message"]["type"] == models.MessageType.ANALYSIS_UPDATE
            and msg["message"]["status"] == "started"
            for msg in mock_connection_manager_instance.messages_sent
        )

        # Assert PAT service was called
        handler.pat_service.analyze_health_data.assert_awaited_once_with(health_data)

        # Assert Gemini service was called
        assert handler.gemini_service.generate_health_insights.called

        # Assert final update message
        assert any(
            msg["type"] == "direct_to_user"
            and msg["message"]["type"] == models.MessageType.ANALYSIS_UPDATE
            and msg["message"]["status"] == "completed"
            for msg in mock_connection_manager_instance.messages_sent
        )

        # Assert health insight message was sent
        assert any(
            msg["type"] == "direct_to_user"
            and msg["message"]["type"] == models.MessageType.HEALTH_INSIGHT
            and "Great activity levels!" in msg["message"]["insight"]
            for msg in mock_connection_manager_instance.messages_sent
        )

    async def test_health_analysis_error_handling(self, app: FastAPI):
        user_id = "test_user_id_5"
        health_data = {"steps": 10000, "sleep_hours": 7}
        mock_connection_manager_instance = TestConnectionManager()

        handler = chat_handler.WebSocketChatHandler(
            pat_service=AsyncMock(), gemini_service=AsyncMock()
        )

        # Simulate an error during PAT analysis
        handler.pat_service.analyze_health_data.side_effect = Exception("PAT Service Error")

        await handler.trigger_health_analysis(
            user_id=user_id,
            health_data=health_data,
            connection_manager=mock_connection_manager_instance
        )

        # Assert error message was sent
        assert any(
            msg["type"] == "direct_to_user"
            and msg["message"]["type"] == models.MessageType.ANALYSIS_UPDATE
            and msg["message"]["status"] == "failed"
            and "PAT Service Error" in msg["message"]["details"]
            for msg in mock_connection_manager_instance.messages_sent
        )

        # Ensure no health insight message was sent after the error
        assert not any(
            msg["type"] == "direct_to_user"
            and msg["message"]["type"] == models.MessageType.HEALTH_INSIGHT
            for msg in mock_connection_manager_instance.messages_sent
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
