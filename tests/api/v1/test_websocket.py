"""Tests for WebSocket chat functionality."""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
import functools
import json
import logging
import time
from typing import Any
from unittest.mock import (
    AsyncMock,
    MagicMock,
    Mock,
)
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
from pydantic import Field
import pytest

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
    ) -> None:
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
            "TestConnectionManager initialized with max_connections_per_user=%s",
            self.max_connections_per_user,
        )

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        username: str,
        room_id: str = "general",
    ) -> _TestConnectionInfo:
        logger.info(
            "Attempting to connect user %s to room %s with session %s",
            user_id,
            room_id,
            uuid.uuid4(),
        )
        if len(self.user_connections[user_id]) >= self.max_connections_per_user:
            logger.warning(
                "Connection limit exceeded for user %s. Max: %s",
                user_id,
                self.max_connections_per_user,
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
            "User %s connected to room %s. Total active websockets: %s",
            user_id,
            room_id,
            len(self.active_websockets),
        )
        logger.info(
            "Connection info for websocket: %s", self.connection_info[websocket]
        )
        return new_connection_info

    async def disconnect(self, websocket: WebSocket, reason: str | None = None) -> None:
        logger.info("Attempting to disconnect websocket. Reason: %s", reason)
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
                logger.info("User %s has no more active connections.", user_id)

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
            logger.info("User %s removed from room %s.", user_id, room_id)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
                logger.info("Room %s is now empty and removed.", room_id)

        logger.info(
            "Websocket disconnected for user %s from room %s. Total active websockets: %s",
            user_id,
            room_id,
            len(self.active_websockets),
        )

    async def send_to_connection(self, websocket: WebSocket, message: Any) -> None:
        logger.info("Attempting to send message to connection: %s", message)

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
            "message": message_content,
        })

        # Actually send the message through the WebSocket for TestClient
        try:
            await websocket.send_json(message_content)
            logger.info("Sent message through websocket: %s", message_content)
        except (RuntimeError, ConnectionError, OSError) as e:
            logger.warning("Failed to send message through websocket: %s", e)

        logger.info("Recorded direct message send to %s", websocket)

    async def send_to_user(self, user_id: str, message: Any) -> None:
        """Send a message to all active connections for a given user."""
        logger.info("Attempting to send message to user %s: %s", user_id, message)

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
                        "message": message_content,
                    })
                    # Actually send the message through the WebSocket for TestClient
                    try:
                        await websocket.send_json(message_content)
                        logger.info(
                            "Sent message to user %s through websocket: %s",
                            user_id,
                            message_content,
                        )
                    except (RuntimeError, ConnectionError, OSError) as e:
                        logger.warning(
                            "Failed to send message to user %s through websocket: %s",
                            user_id,
                            e,
                        )

                    logger.info(
                        "Recorded direct message send to user %s via %s",
                        user_id,
                        websocket,
                    )
        else:
            logger.warning(
                "User %s has no active connections to send messages to.", user_id
            )

    async def broadcast_to_room(
        self,
        room_id: str,
        message: Any,
        exclude_websocket: WebSocket | None = None,
    ) -> None:
        logger.info("Attempting to broadcast message to room %s: %s", room_id, message)

        message_content = message
        if hasattr(message, "model_dump_json"):
            message_content = json.loads(message.model_dump_json())
        elif hasattr(message, "dict"):
            message_content = message.dict()

        self.messages_sent.append({
            "type": "broadcast",
            "room_id": room_id,
            "message": message_content,
            "excluded": exclude_websocket,
        })

        target_websockets = []
        for user_id in self.rooms.get(room_id, set()):
            target_websockets.extend(
                ws
                for ws in self.user_connections.get(user_id, [])
                if ws != exclude_websocket and ws in self.active_websockets
            )

        # In tests, send to all connections in room for testing purposes
        for user_id in self.rooms.get(room_id, set()):
            for ws in self.user_connections.get(user_id, []):
                if ws in self.active_websockets:
                    try:
                        message_str = (
                            message.model_dump_json()
                            if hasattr(message, "model_dump_json")
                            else json.dumps(message_content)
                        )
                        await ws.send_text(message_str)
                    except (RuntimeError, ConnectionError, OSError):
                        pass

        logger.info(
            "Broadcast sent to %s connections in room %s",
            len(target_websockets),
            room_id,
        )

    async def handle_heartbeat(self, websocket: WebSocket) -> None:
        logger.info("Handling heartbeat for websocket")

        connection_info = self.connection_info.get(websocket)
        if connection_info:
            now = datetime.now(UTC)
            connection_info.last_heartbeat_ack = now
            connection_info.last_active = now

            self.heartbeats_processed.append({"websocket": websocket, "timestamp": now})

    def is_rate_limited(self, websocket: WebSocket) -> bool:
        connection_info = self.connection_info.get(websocket)
        if not connection_info:
            logger.warning("Cannot check rate limit for unknown websocket.")
            return False

        user_id = connection_info.user_id
        current_time = datetime.now(UTC)
        self.message_counts[user_id].append(current_time.timestamp())

        # Remove messages older than the rate limit period
        cutoff_time = (
            current_time - timedelta(seconds=self.message_rate_limit_period_seconds)
        ).timestamp()
        recent_messages = [
            ts for ts in self.message_counts[user_id] if ts > cutoff_time
        ]
        self.message_counts[user_id] = recent_messages

        if len(recent_messages) > self.message_rate_limit_count:
            logger.warning("User %s is rate-limited.", user_id)
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

    def get_user_count(self) -> int:
        return len(self.user_connections)

    def get_room_user_count(self, room_id: str) -> int:
        return len(self.rooms.get(room_id, set()))

    def get_connection_count(self) -> int:
        return len(self.active_websockets)

    def get_connection_info_for_websocket(
        self, websocket: WebSocket
    ) -> _TestConnectionInfo | None:
        return self.connection_info.get(websocket)

    @staticmethod
    async def handle_message(_websocket: WebSocket, _raw_message: str) -> bool:
        """Handle an incoming WebSocket message - always allow in tests."""
        # In tests, always return True to allow message processing
        return True


@pytest.fixture
def mock_test_connection_manager() -> _TestConnectionManager:
    return _TestConnectionManager()


def create_mock_connection_manager() -> _TestConnectionManager:
    """Helper to create a _TestConnectionManager instance."""
    return _TestConnectionManager()


def mock_get_current_user_websocket(
    token: str, test_env_credentials: dict[str, str]
) -> User:
    """Mock for get_current_user_websocket dependency."""
    if token == test_env_credentials["mock_access_token"]:
        return User(
            uid="test-user-123",
            email="test@example.com",
            display_name="Test User",
            firebase_token=test_env_credentials["mock_firebase_token"],
            created_at=datetime.now(UTC),
            last_login=datetime.now(UTC),
            firebase_token_exp=(datetime.now(UTC) + timedelta(hours=1)).timestamp(),
            profile={},
        )
    raise HTTPException(status_code=401, detail="Could not validate credentials")


@pytest.fixture
def app(
    monkeypatch: pytest.MonkeyPatch, test_env_credentials: dict[str, str]
) -> FastAPI:
    # This fixture should return a FastAPI app instance configured for testing.
    # It needs to override dependencies to use our mocks.
    app = FastAPI()

    # Include the chat router
    app.include_router(chat_handler.router, prefix="/api/v1")

    # Override dependencies for testing
    # Create a partial function with test_env_credentials pre-filled
    mock_auth_dependency = functools.partial(
        mock_get_current_user_websocket, test_env_credentials=test_env_credentials
    )
    app.dependency_overrides[get_current_user_websocket] = mock_auth_dependency

    # Create properly mocked GeminiService
    mock_gemini = Mock(spec=GeminiService)

    # Create a proper mock response
    async def mock_generate_insights(  # noqa: RUF029
        request: Any,
    ) -> object:
        response = MagicMock()
        response.narrative = f"AI Response to: {request.context}"
        return response

    mock_gemini.generate_health_insights = AsyncMock(side_effect=mock_generate_insights)
    app.dependency_overrides[chat_handler.get_gemini_service] = lambda: mock_gemini
    app.dependency_overrides[chat_handler.get_pat_model_service] = lambda: Mock(
        spec=PATModelService
    )
    app.dependency_overrides[get_connection_manager] = create_mock_connection_manager

    # GeminiService is mocked via dependency override above

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_get_connection_manager():
    # This fixture provides a MagicMock for the ConnectionManager.
    # It's used in tests that directly patch get_connection_manager.
    with Mock(spec=ConnectionManager) as mock_manager:
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

    async def close(self, _code: int = 1000, _reason: str = "") -> None:
        self.closed = True


@pytest.mark.asyncio
class TestWebSocketEndpoints:
    async def test_websocket_chat_endpoint_authenticated(  # noqa: PLR6301
        self, client: TestClient, test_env_credentials: dict[str, str]
    ) -> None:
        user_id = "test-user-123"
        test_token = test_env_credentials["mock_access_token"]

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
                username=test_env_credentials["default_username"],
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
        user_id = "anonymous-user-123"
        with (
            pytest.raises(WebSocketDisconnect) as excinfo,
            client.websocket_connect(f"/api/v1/chat/{user_id}") as websocket,
        ):
            # This connection should be rejected by the auth middleware
            websocket.send_text("Hello")
        assert excinfo.value.code == 1008  # Policy Violation

    @staticmethod
    async def test_websocket_invalid_message_format(
        client: TestClient, test_env_credentials: dict[str, str]
    ) -> None:
        user_id = "test-user-123"
        test_token = test_env_credentials["mock_access_token"]

        with client.websocket_connect(
            f"/api/v1/chat/{user_id}?token={test_token}"
        ) as websocket:
            # Send an invalid message format
            websocket.send_text("this is not json")

            response_data = websocket.receive_json()
            assert response_data["type"] == MessageType.ERROR.value
            assert "Invalid JSON format" in response_data["message"]

    @staticmethod
    async def test_websocket_typing_indicator(
        client: TestClient, test_env_credentials: dict[str, str]
    ) -> None:
        user_id = "test-user-123"
        test_token = test_env_credentials["mock_access_token"]

        with client.websocket_connect(
            f"/api/v1/chat/{user_id}?token={test_token}"
        ) as websocket:
            # Send a typing indicator message
            typing_message = TypingMessage(
                user_id=user_id,
                username=test_env_credentials["default_username"],
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

    @staticmethod
    async def test_health_insight_generation(
        client: TestClient, app: FastAPI, test_env_credentials: dict[str, str]
    ) -> None:
        user_id = "test-user-123"  # Must match the user ID from mock_get_current_user_websocket
        test_token = test_env_credentials["mock_access_token"]

        # Mock the external services for this specific test
        mock_gemini_service = Mock(spec=GeminiService)

        # Create a proper mock response
        async def mock_generate_insights_test(  # noqa: RUF029 # Renamed to avoid conflict
            request: Any,  # MODIFIED from object
        ) -> object:
            response = MagicMock()
            response.narrative = f"AI Response to: {request.context}"
            return response

        mock_gemini_service.generate_health_insights = AsyncMock(
            side_effect=mock_generate_insights_test  # MODIFIED
        )
        mock_pat_model_service = Mock(spec=PATModelService)
        # Mock the actual method used by the chat handler
        mock_pat_model_service.analyze_actigraphy = Mock(
            return_value=Mock(
                sleep_efficiency=0.85,
                total_sleep_time=7.5,
                model_dump=lambda: {"sleep_efficiency": 0.85, "total_sleep_time": 7.5},
            )
        )

        # Temporarily override dependencies for this test to use our explicit mocks
        original_gemini_service = app.dependency_overrides.get(
            chat_handler.get_gemini_service
        )
        original_pat_model_service = app.dependency_overrides.get(
            chat_handler.get_pat_model_service
        )
        original_connection_manager = app.dependency_overrides.get(
            get_connection_manager
        )

        app.dependency_overrides[chat_handler.get_gemini_service] = (
            lambda: mock_gemini_service
        )
        app.dependency_overrides[chat_handler.get_pat_model_service] = (
            lambda: mock_pat_model_service
        )

        # Create a specific mock_manager for this test and apply Mock to its methods
        mock_manager = _TestConnectionManager()
        mock_manager.send_to_user = Mock(side_effect=mock_manager.send_to_user)  # type: ignore[method-assign]
        mock_manager.send_to_connection = Mock(  # type: ignore[method-assign]
            side_effect=mock_manager.send_to_connection
        )
        mock_manager.broadcast_to_room = Mock(  # type: ignore[method-assign]
            side_effect=mock_manager.broadcast_to_room
        )

        app.dependency_overrides[get_connection_manager] = lambda: mock_manager

        # Use TestClient.websocket_connect to interact with the WebSocket endpoint
        with client.websocket_connect(
            f"/api/v1/chat/{user_id}?token={test_token}"
        ) as websocket:
            # Send a chat message to trigger insight generation
            chat_message = ChatMessage(
                user_id=user_id,
                timestamp=datetime.now(UTC),
                type=MessageType.MESSAGE,
                content="Generate insight",
            )
            websocket.send_json(chat_message.model_dump(mode="json"))

            # Expect the AI response
            # The connection manager will send these back via send_to_user
            # The test client will receive them
            ai_response = websocket.receive_json()
            assert ai_response["type"] == MessageType.MESSAGE.value
            assert "AI Response to: Generate insight" in ai_response["content"]

            # Assert that the mocked services were called
            # Note: The actual WebSocket communication is working correctly
            # The service method names may vary but the WebSocket functionality is tested
            mock_gemini_service.generate_health_insights.assert_awaited_once()

            # Check that the connection manager methods were called correctly
            # Assert on the explicit mock_manager created for this test
            assert (
                mock_manager.send_to_user.call_count == 1
            )  # Only the AI response is sent via send_to_user
            assert (
                mock_manager.broadcast_to_room.call_count == 0
            )  # No broadcast for regular chat messages

        # Clean up dependency overrides after the test
        if original_gemini_service is not None:
            app.dependency_overrides[chat_handler.get_gemini_service] = (
                original_gemini_service
            )
        else:
            del app.dependency_overrides[chat_handler.get_gemini_service]

        if original_pat_model_service is not None:
            app.dependency_overrides[chat_handler.get_pat_model_service] = (
                original_pat_model_service
            )
        else:
            del app.dependency_overrides[chat_handler.get_pat_model_service]

        if original_connection_manager is not None:
            app.dependency_overrides[get_connection_manager] = (
                original_connection_manager
            )
        else:
            del app.dependency_overrides[get_connection_manager]

    @staticmethod
    async def test_typing_indicator_processing(
        client: TestClient, app: FastAPI, test_env_credentials: dict[str, str]
    ) -> None:
        user_id = "test-user-123"  # Must match the user ID from mock_get_current_user_websocket
        test_token = test_env_credentials["mock_access_token"]

        # Mock the external services for this specific test
        mock_gemini_service = Mock(spec=GeminiService)
        mock_pat_model_service = Mock(spec=PATModelService)

        # Temporarily override dependencies for this test to use our explicit mocks
        original_gemini_service = app.dependency_overrides.get(
            chat_handler.get_gemini_service
        )
        original_pat_model_service = app.dependency_overrides.get(
            chat_handler.get_pat_model_service
        )
        original_connection_manager = app.dependency_overrides.get(
            get_connection_manager
        )

        app.dependency_overrides[chat_handler.get_gemini_service] = (
            lambda: mock_gemini_service
        )
        app.dependency_overrides[chat_handler.get_pat_model_service] = (
            lambda: mock_pat_model_service
        )

        # Create a specific mock_manager for this test and apply Mock to its methods
        mock_manager = _TestConnectionManager()
        mock_manager.send_to_user = Mock(side_effect=mock_manager.send_to_user)  # type: ignore[method-assign]
        mock_manager.send_to_connection = Mock(  # type: ignore[method-assign]
            side_effect=mock_manager.send_to_connection
        )
        mock_manager.broadcast_to_room = Mock(  # type: ignore[method-assign]
            side_effect=mock_manager.broadcast_to_room
        )

        app.dependency_overrides[get_connection_manager] = lambda: mock_manager

        with client.websocket_connect(
            f"/api/v1/chat/{user_id}?token={test_token}"
        ) as websocket:
            # Send a typing indicator message
            typing_message = TypingMessage(
                user_id=user_id,
                username=test_env_credentials["default_username"],
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

            # Check that the connection manager methods were called correctly
            assert (
                mock_manager.broadcast_to_room.call_count == 2
            )  # One for True, one for False

        # Clean up dependency overrides after the test
        if original_gemini_service is not None:
            app.dependency_overrides[chat_handler.get_gemini_service] = (
                original_gemini_service
            )
        else:
            del app.dependency_overrides[chat_handler.get_gemini_service]

        if original_pat_model_service is not None:
            app.dependency_overrides[chat_handler.get_pat_model_service] = (
                original_pat_model_service
            )
        else:
            del app.dependency_overrides[chat_handler.get_pat_model_service]

        if original_connection_manager is not None:
            app.dependency_overrides[get_connection_manager] = (
                original_connection_manager
            )
        else:
            del app.dependency_overrides[get_connection_manager]

    @staticmethod
    async def test_heartbeat_processing(
        client: TestClient, app: FastAPI, test_env_credentials: dict[str, str]
    ) -> None:
        user_id = "test-user-123"  # Must match the user ID from mock_get_current_user_websocket
        test_token = test_env_credentials["mock_access_token"]

        # Mock the external services for this specific test
        mock_gemini_service = Mock(spec=GeminiService)
        mock_pat_model_service = Mock(spec=PATModelService)

        # Temporarily override dependencies for this test to use our explicit mocks
        original_gemini_service = app.dependency_overrides.get(
            chat_handler.get_gemini_service
        )
        original_pat_model_service = app.dependency_overrides.get(
            chat_handler.get_pat_model_service
        )
        original_connection_manager = app.dependency_overrides.get(
            get_connection_manager
        )

        app.dependency_overrides[chat_handler.get_gemini_service] = (
            lambda: mock_gemini_service
        )
        app.dependency_overrides[chat_handler.get_pat_model_service] = (
            lambda: mock_pat_model_service
        )

        # Create a specific mock_manager for this test and apply Mock to its methods
        mock_manager = _TestConnectionManager()
        mock_manager.send_to_user = Mock(side_effect=mock_manager.send_to_user)  # type: ignore[method-assign]
        mock_manager.send_to_connection = Mock(  # type: ignore[method-assign]
            side_effect=mock_manager.send_to_connection
        )
        mock_manager.broadcast_to_room = Mock(  # type: ignore[method-assign]
            side_effect=mock_manager.broadcast_to_room
        )

        app.dependency_overrides[get_connection_manager] = lambda: mock_manager

        with client.websocket_connect(
            f"/api/v1/chat/{user_id}?token={test_token}"
        ) as websocket:
            # Send a heartbeat message
            heartbeat_message = HeartbeatMessage(
                timestamp=datetime.now(UTC),
                type=MessageType.HEARTBEAT,
                client_timestamp=datetime.now(UTC),
            )
            websocket.send_json(heartbeat_message.model_dump(mode="json"))

            # Expect a heartbeat acknowledgment response
            response_data = websocket.receive_json()
            assert response_data["type"] == MessageType.HEARTBEAT_ACK.value
            # HeartbeatAckMessage doesn't have user_id field, just timestamp fields
            assert "timestamp" in response_data

            # Check that the connection manager methods were called correctly
            assert mock_manager.send_to_connection.call_count == 1

        # Clean up dependency overrides after the test
        if original_gemini_service is not None:
            app.dependency_overrides[chat_handler.get_gemini_service] = (
                original_gemini_service
            )
        else:
            del app.dependency_overrides[chat_handler.get_gemini_service]

        if original_pat_model_service is not None:
            app.dependency_overrides[chat_handler.get_pat_model_service] = (
                original_pat_model_service
            )
        else:
            del app.dependency_overrides[chat_handler.get_pat_model_service]

        if original_connection_manager is not None:
            app.dependency_overrides[get_connection_manager] = (
                original_connection_manager
            )
        else:
            del app.dependency_overrides[get_connection_manager]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
