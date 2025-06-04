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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
from pydantic import Field, ValidationError
import pytest
from starlette.websockets import WebSocketState

from clarity.api.v1.websocket import chat_handler, models
from clarity.api.v1.websocket.connection_manager import ConnectionManager
from clarity.api.v1.websocket.models import (
    ChatMessage,
    ConnectionInfo,
    HeartbeatMessage,
    MessageType,
    TypingMessage,
)
from clarity.auth.firebase_auth import User, get_current_user_websocket
from clarity.models.user import User, UserProfile

logger = logging.getLogger(__name__)
# Basic logging for tests, customize as needed, e.g., in conftest.py for global config
# logging.basicConfig(level=logging.INFO)


class _TestConnectionInfo(ConnectionInfo):
    """Extended ConnectionInfo for testing purposes."""

    room_id: str
    last_active: datetime = Field(default_factory=datetime.utcnow)
    last_heartbeat_ack: datetime = Field(default_factory=datetime.utcnow)
    message_timestamps: list[datetime] = Field(default_factory=list)


class _TestConnectionManager:
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
        self.connection_info: dict[WebSocket, _TestConnectionInfo] = {}
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
    ) -> _TestConnectionInfo:
        logger.info(
            f"Attempting to connect user {user_id} to room {room_id} with session {uuid.uuid4()}"
        )
        if len(self.user_connections[user_id]) >= self.max_connections_per_user:
            logger.warning(
                f"Connection limit exceeded for user {user_id}. Max: {self.max_connections_per_user}"
            )

        connection_time = datetime.now(UTC)
        new_connection_info = _TestConnectionInfo(
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
    ) -> _TestConnectionInfo | None:
        return self.connection_info.get(websocket)


@pytest.fixture
def mock_test_connection_manager() -> _TestConnectionManager:
    return _TestConnectionManager()


def create_mock_connection_manager():
    # This function is likely called in chat_handler.py to get the manager instance
    # We need to ensure it returns the _TestConnectionManager for tests
    return _TestConnectionManager()


@pytest.fixture
def app() -> FastAPI:
    # This fixture should return a FastAPI app instance configured for testing.
    # It needs to override dependencies to use our mocks.

    app = FastAPI()

    app.include_router(chat_handler.router, prefix="/api/v1/chat")
    # This mock should return a User instance for successful authentication

    async def mock_get_current_user_websocket(token: str | None = None) -> User:
        if token == "test-token":
            # Correctly instantiate the User model based on its __init__ or create method
            # Assuming User has user_id, email, and display_name for simplicity in tests
            # Adjust these fields to match the actual User model's __init__
            return User(
                uid="test-user-123",
                email="test@example.com",
                display_name="Test User",
                firebase_token="mock-firebase-token",
                created_at=datetime.now(UTC),
                last_login=datetime.now(UTC),
                profile={},
            )
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    app.dependency_overrides[get_current_user_websocket] = mock_get_current_user_websocket

    # Override the get_connection_manager dependency in chat_handler
    # This ensures our test mock is used when the WebSocket endpoint is called
    app.dependency_overrides[chat_handler.get_connection_manager] = (
        create_mock_connection_manager
    )

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_get_connection_manager():
    # This fixture provides a MagicMock for the ConnectionManager.
    # It's used in tests that directly patch get_connection_manager.
    with MagicMock(spec=ConnectionManager) as mock_manager:
        yield mock_manager


@pytest.fixture
def connection_manager():
    # This fixture provides an instance of the actual ConnectionManager for tests
    # that need to interact with it directly.
    return ConnectionManager()


class MockWebSocket:
    def __init__(self) -> None:
        self.sent_data: list[str] = []
        self.received_data: list[str] = []
        self.closed: bool = False

    async def accept(self) -> None:
        pass

    async def send_text(self, data: str) -> None:
        self.sent_data.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True


@pytest.mark.asyncio
class TestWebSocketEndpoints:
    async def test_websocket_chat_endpoint_authenticated(
        self, client: TestClient
    ) -> None:
        user_id = "test-user-123"
        test_token = "test-token"
        auth_headers = {"Authorization": f"Bearer {test_token}"}

        with client.websocket_connect(f"/api/v1/chat/{user_id}", headers=auth_headers) as websocket:
            # Send a chat message
            chat_message = ChatMessage(
                user_id=user_id,
                timestamp=datetime.now(UTC),
                type=MessageType.MESSAGE,
                content="Hello AI",
            )
            websocket.send_json(chat_message.model_dump())

            # Expecting a response from the AI handler
            response_data = websocket.receive_json()
            assert response_data["message_type"] == MessageType.MESSAGE
            assert "AI Response to: Hello AI" in response_data["content"]
            assert response_data["user_id"] == user_id

            # Test typing indicator
            typing_indicator = TypingMessage(
                user_id=user_id,
                timestamp=datetime.now(UTC),
                is_typing=True,
                type=MessageType.TYPING,
                username="test-user",
            )
            websocket.send_json(typing_indicator.model_dump())

            response_data = websocket.receive_json()
            assert response_data["message_type"] == MessageType.TYPING
            assert response_data["user_id"] == user_id
            assert response_data["is_typing"] is True

            # Test heartbeat
            heartbeat_message = HeartbeatMessage(
                timestamp=datetime.now(UTC),
                type=MessageType.HEARTBEAT,
            )
            websocket.send_json(heartbeat_message.model_dump())

            response_data = websocket.receive_json()
            assert response_data["message_type"] == MessageType.HEARTBEAT_ACK
            assert response_data["user_id"] == user_id

    async def test_websocket_chat_endpoint_anonymous(self, client: TestClient) -> None:
        user_id = "anonymous-user-123"
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect(f"/api/v1/chat/{user_id}") as websocket:
                # This connection should be rejected by the auth middleware
                websocket.send_text("Hello")
        assert excinfo.value.code == 1008  # Policy Violation

    async def test_websocket_invalid_message_format(self, client: TestClient) -> None:
        user_id = "test-user-123"
        test_token = "test-token"
        auth_headers = {"Authorization": f"Bearer {test_token}"}

        with client.websocket_connect(f"/api/v1/chat/{user_id}", headers=auth_headers) as websocket:
            # Send an invalid message format
            websocket.send_text("this is not json")

            response_data = websocket.receive_json()
            assert response_data["message_type"] == MessageType.ERROR
            assert "Invalid message format" in response_data["content"]

    async def test_websocket_typing_indicator(self, client: TestClient) -> None:
        user_id = "test-user-123"
        test_token = "test-token"
        auth_headers = {"Authorization": f"Bearer {test_token}"}

        with client.websocket_connect(f"/api/v1/chat/{user_id}", headers=auth_headers) as websocket:
            # Send a typing indicator message
            typing_message = TypingMessage(
                user_id=user_id,
                username="test-user",
                timestamp=datetime.now(UTC),
                is_typing=True,
                type=MessageType.TYPING,
            )
            websocket.send_json(typing_message.model_dump())

            # Expect a typing indicator response
            response_data = websocket.receive_json()
            assert response_data["message_type"] == MessageType.TYPING
            assert response_data["user_id"] == user_id
            assert response_data["is_typing"] is True

            # Send another typing indicator, now indicating not typing
            typing_message.is_typing = False
            websocket.send_json(typing_message.model_dump())

            # Expect a typing indicator response
            response_data = websocket.receive_json()
            assert response_data["message_type"] == MessageType.TYPING
            assert response_data["user_id"] == user_id
            assert response_data["is_typing"] is False


@pytest.mark.asyncio
class TestChatHandler:
    async def test_health_insight_generation(self, app: FastAPI):
        user_id = "test_user_id_1"
        # Mock the external services
        mock_gemini_service = AsyncMock()
        mock_gemini_service.stream_chat_response.return_value = [
            "AI ", "Response ", "Chunk"
        ]
        mock_pat_model_service = AsyncMock()
        mock_pat_model_service.analyze_health_data.return_value = {
            "insight": "Test Insight"
        }

        # Create a mock WebSocket for the handler
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.receive_json.side_effect = [
            {
                "message_type": MessageType.MESSAGE,
                "user_id": user_id,
                "session_id": "test-session-123",
                "timestamp": datetime.now(UTC).isoformat(),
                "content": "Generate insight",
            },
            WebSocketDisconnect,
        ]
        # Mock the connection manager
        mock_manager = AsyncMock(spec=_TestConnectionManager)
        mock_manager.send_to_user = AsyncMock()
        mock_manager.send_to_connection = AsyncMock()
        mock_manager.broadcast_to_room = AsyncMock()

        mock_manager.connect.return_value = _TestConnectionInfo(
            user_id=user_id,
            username="testuser",
            session_id="test-session-123",
            connected_at=datetime.now(UTC),
            room_id="test-room",
        )
        mock_manager.connection_info = {mock_websocket: mock_manager.connect.return_value}

        # Instantiate the handler with the mocked services
        handler = chat_handler.WebSocketChatHandler(
            gemini_service=mock_gemini_service,
            pat_service=mock_pat_model_service
        )

        await handler.chat_ws(mock_websocket, user_id, connection_manager=mock_manager)

        mock_websocket.accept.assert_awaited_once()
        mock_websocket.receive_json.assert_awaited_once()
        mock_pat_model_service.analyze_health_data.assert_awaited_once()
        mock_gemini_service.stream_chat_response.assert_awaited_once()
        mock_websocket.send_json.assert_awaited_with(Any)
        assert mock_manager.send_to_user.call_count == 2
        # The first call is for typing indicator, second is for AI response
        # Check the content of the second call
        second_call_args = mock_manager.send_to_user.call_args_list[1].args[1]
        assert "Test Insight" in second_call_args["content"]

    async def test_typing_indicator_processing(self, app: FastAPI):
        user_id = "test_user_id_2"
        mock_connection_manager_instance = _TestConnectionManager()

        # Mock the external services for this test
        mock_gemini_service = AsyncMock()
        mock_pat_model_service = AsyncMock()

        handler = chat_handler.WebSocketChatHandler(
            gemini_service=mock_gemini_service,
            pat_service=mock_pat_model_service
        )

        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.receive_json.side_effect = [
            {
                "message_type": MessageType.TYPING,
                "user_id": user_id,
                "session_id": "test-session-456",
                "timestamp": datetime.now(UTC).isoformat(),
                "is_typing": True,
            },
            WebSocketDisconnect,
        ]

        mock_connection_manager_instance.send_to_user = AsyncMock()
        mock_connection_manager_instance.send_to_connection = AsyncMock()
        mock_connection_manager_instance.broadcast_to_room = AsyncMock()

        await handler.chat_ws(mock_websocket, user_id, connection_manager=mock_connection_manager_instance)

        mock_websocket.accept.assert_awaited_once()
        mock_websocket.receive_json.assert_awaited_once()
        assert mock_connection_manager_instance.send_to_user.call_count == 1
        sent_message = mock_connection_manager_instance.send_to_user.call_args[0][1]
        assert sent_message["message_type"] == MessageType.TYPING
        assert sent_message["is_typing"] is True

    async def test_heartbeat_processing(self, app: FastAPI):
        user_id = "test_user_id_3"
        mock_connection_manager_instance = _TestConnectionManager()

        # Mock the external services for this test
        mock_gemini_service = AsyncMock()
        mock_pat_model_service = AsyncMock()

        handler = chat_handler.WebSocketChatHandler(
            gemini_service=mock_gemini_service,
            pat_service=mock_pat_model_service
        )

        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.receive_json.side_effect = [
            {
                "message_type": MessageType.HEARTBEAT,
                "user_id": user_id,
                "session_id": "test-session-789",
                "timestamp": datetime.now(UTC).isoformat(),
            },
            WebSocketDisconnect,
        ]

        mock_connection_manager_instance.send_to_user = AsyncMock()
        mock_connection_manager_instance.send_to_connection = AsyncMock()
        mock_connection_manager_instance.broadcast_to_room = AsyncMock()

        await handler.chat_ws(mock_websocket, user_id, connection_manager=mock_connection_manager_instance)

        mock_websocket.accept.assert_awaited_once()
        mock_websocket.receive_json.assert_awaited_once()
        assert mock_connection_manager_instance.send_to_user.call_count == 1
        sent_message = mock_connection_manager_instance.send_to_user.call_args[0][1]
        assert sent_message["message_type"] == MessageType.HEARTBEAT_ACK

    async def test_health_analysis_trigger(self, app: FastAPI):
        user_id = "test_user_id_4"
        health_data = {"steps": 10000, "sleep_hours": 7}
        mock_connection_manager_instance = _TestConnectionManager()

        mock_pat_model_service = AsyncMock()
        mock_pat_model_service.analyze_health_data.return_value = {"insight": "Great job!"}

        handler = chat_handler.WebSocketChatHandler()

        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.receive_json.side_effect = [
            {
                "message_type": models.MessageType.HEALTH_DATA,
                "user_id": user_id,
                "session_id": "test-session-001",
                "timestamp": datetime.now(UTC).isoformat(),
                "content": json.dumps(health_data),
            },
            WebSocketDisconnect,
        ]

        with (
            MagicMock(spec=chat_handler.get_pat_model_service) as mock_get_pat_model_service,
            MagicMock(spec=chat_handler.get_connection_manager) as mock_get_connection_manager,
        ):
            mock_get_pat_model_service.return_value = mock_pat_model_service
            mock_get_connection_manager.return_value = mock_connection_manager_instance

            chat_handler.get_pat_model_service = mock_get_pat_model_service
            chat_handler.get_connection_manager = mock_get_connection_manager

            await handler.chat_ws(mock_websocket, user_id)

            mock_websocket.accept.assert_awaited_once()
            mock_websocket.receive_json.assert_awaited_once()
            mock_pat_model_service.analyze_health_data.assert_awaited_once_with(
                user_id=user_id, health_data=health_data
            )
            assert mock_connection_manager_instance.send_to_user.call_count == 1
            sent_message = mock_connection_manager_instance.send_to_user.call_args[0][1]
            assert sent_message["message_type"] == models.MessageType.HEALTH_INSIGHT
            assert "Great job!" in sent_message["content"]

    async def test_health_analysis_error_handling(self, app: FastAPI):
        user_id = "test_user_id_5"
        health_data = {"steps": 10000, "sleep_hours": 7}
        mock_connection_manager_instance = _TestConnectionManager()

        mock_pat_model_service = AsyncMock()
        mock_pat_model_service.analyze_health_data.side_effect = Exception("AI Error")

        handler = chat_handler.WebSocketChatHandler()

        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.receive_json.side_effect = [
            {
                "message_type": models.MessageType.HEALTH_DATA,
                "user_id": user_id,
                "session_id": "test-session-002",
                "timestamp": datetime.now(UTC).isoformat(),
                "content": json.dumps(health_data),
            },
            WebSocketDisconnect,
        ]

        with (
            MagicMock(spec=chat_handler.get_pat_model_service) as mock_get_pat_model_service,
            MagicMock(spec=chat_handler.get_connection_manager) as mock_get_connection_manager,
        ):
            mock_get_pat_model_service.return_value = mock_pat_model_service
            mock_get_connection_manager.return_value = mock_connection_manager_instance

            chat_handler.get_pat_model_service = mock_get_pat_model_service
            chat_handler.get_connection_manager = mock_get_connection_manager

            await handler.chat_ws(mock_websocket, user_id)

            mock_websocket.accept.assert_awaited_once()
            mock_websocket.receive_json.assert_awaited_once()
            mock_pat_model_service.analyze_health_data.assert_awaited_once_with(
                user_id=user_id, health_data=health_data
            )
            assert mock_connection_manager_instance.send_to_user.call_count == 1
            sent_message = mock_connection_manager_instance.send_to_user.call_args[0][1]
            assert sent_message["message_type"] == models.MessageType.ERROR
            assert "Error processing health data" in sent_message["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
