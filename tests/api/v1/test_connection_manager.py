import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import WebSocket
import pytest
from starlette.websockets import WebSocketState

from clarity.api.v1.websocket.connection_manager import ConnectionManager
from clarity.api.v1.websocket.models import (
    ChatMessage,
    MessageType,
)


@pytest.mark.asyncio
async def test_connect_and_disconnect():
    manager = ConnectionManager()
    websocket = AsyncMock(spec=WebSocket)
    websocket.client_state = WebSocketState.CONNECTED
    user_id = "test_user"
    username = "Test User"

    # Test connect
    await manager.connect(websocket, user_id, username)
    assert manager.get_connection_count() == 1
    assert manager.get_user_count() == 1
    assert user_id in manager.user_connections
    assert websocket in manager.connection_info

    # Test disconnect
    await manager.disconnect(websocket)
    assert manager.get_connection_count() == 0
    assert manager.get_user_count() == 0
    assert user_id not in manager.user_connections


def test_cleanup_rate_limiting_data():
    manager = ConnectionManager(message_rate_limit=2)
    user_id = "test_user"

    # Add some old and new message timestamps
    manager.message_counts[user_id] = [
        time.time() - 70,
        time.time() - 80,
        time.time() - 10,
        time.time() - 5,
    ]

    manager.cleanup_rate_limiting_data()

    assert len(manager.message_counts[user_id]) == 2


@pytest.mark.asyncio
async def test_send_heartbeats():
    manager = ConnectionManager()
    ws1 = AsyncMock(spec=WebSocket)
    ws1.client_state = WebSocketState.CONNECTED
    ws2 = AsyncMock(spec=WebSocket)
    ws2.client_state = WebSocketState.CONNECTED

    await manager.connect(ws1, "user1", "userone")
    await manager.connect(ws2, "user2", "usertwo")

    ws1.send_text.reset_mock()
    ws2.send_text.reset_mock()

    await manager._send_heartbeats()

    ws1.send_text.assert_called_once()
    ws2.send_text.assert_called_once()


@pytest.mark.asyncio
async def test_get_room_users():
    manager = ConnectionManager()
    ws1 = AsyncMock(spec=WebSocket)
    ws1.client_state = WebSocketState.CONNECTED
    ws2 = AsyncMock(spec=WebSocket)
    ws2.client_state = WebSocketState.CONNECTED

    await manager.connect(ws1, "user1", "userone", room_id="room1")
    await manager.connect(ws2, "user2", "usertwo", room_id="room1")

    assert set(manager.get_room_users("room1")) == {"user1", "user2"}


@pytest.mark.asyncio
async def test_get_user_info():
    manager = ConnectionManager()
    ws = AsyncMock(spec=WebSocket)
    ws.client_state = WebSocketState.CONNECTED
    user_id = "test_user"
    username = "Test User"

    await manager.connect(ws, user_id, username)

    info = manager.get_user_info(user_id)
    assert info is not None
    assert info["user_id"] == user_id
    assert info["username"] == username


@pytest.mark.asyncio
async def test_rate_limiting():
    manager = ConnectionManager(message_rate_limit=2)
    websocket = AsyncMock(spec=WebSocket)
    websocket.client_state = WebSocketState.CONNECTED
    user_id = "test_user"

    await manager.connect(websocket, user_id, "testuser")

    raw_message = (
        '{"type": "message", "content": "test message", "user_id": "test_user"}'
    )
    assert await manager.handle_message(websocket, raw_message) is True
    assert await manager.handle_message(websocket, raw_message) is True
    assert await manager.handle_message(websocket, raw_message) is False


@pytest.mark.asyncio
@patch("time.time", return_value=1622995200.0)
async def test_background_tasks(mock_time: MagicMock):
    manager = ConnectionManager(heartbeat_interval=0.1, connection_timeout=0.1)
    await manager.start_background_tasks()

    # Test heartbeat
    ws = AsyncMock(spec=WebSocket)
    ws.client_state = WebSocketState.CONNECTED
    await manager.connect(ws, "user1", "userone")

    await asyncio.sleep(0.2)
    # The heartbeat message is created inside the loop, so we can't easily mock it.
    # Instead, we check that send_text was called.
    assert ws.send_text.call_count > 0

    # Test cleanup
    manager.last_heartbeat[ws] = mock_time.return_value - 0.2
    await manager._cleanup_stale_connections()
    assert manager.get_connection_count() == 0

    await manager.stop_background_tasks()


@pytest.mark.asyncio
async def test_broadcast_to_all():
    manager = ConnectionManager()
    ws1 = AsyncMock(spec=WebSocket)
    ws1.client_state = WebSocketState.CONNECTED
    ws2 = AsyncMock(spec=WebSocket)
    ws2.client_state = WebSocketState.CONNECTED

    await manager.connect(ws1, "user1", "userone")
    await manager.connect(ws2, "user2", "usertwo")

    ws1.send_text.reset_mock()
    ws2.send_text.reset_mock()

    message = ChatMessage(type=MessageType.MESSAGE, content="hello all", user_id="test")
    await manager.broadcast_to_all(message)

    ws1.send_text.assert_called_with(message.model_dump_json())
    ws2.send_text.assert_called_with(message.model_dump_json())


@pytest.mark.asyncio
async def test_broadcast_to_room():
    manager = ConnectionManager()
    ws1 = AsyncMock(spec=WebSocket)
    ws1.client_state = WebSocketState.CONNECTED
    ws2 = AsyncMock(spec=WebSocket)
    ws2.client_state = WebSocketState.CONNECTED
    ws3 = AsyncMock(spec=WebSocket)
    ws3.client_state = WebSocketState.CONNECTED

    await manager.connect(ws1, "user1", "userone", room_id="room1")
    await manager.connect(ws2, "user2", "usertwo", room_id="room1")
    await manager.connect(ws3, "user3", "userthree", room_id="room2")

    ws1.send_text.reset_mock()
    ws2.send_text.reset_mock()
    ws3.send_text.reset_mock()

    message = ChatMessage(
        type=MessageType.MESSAGE, content="hello room1", user_id="test"
    )
    await manager.broadcast_to_room("room1", message)

    ws1.send_text.assert_called_with(message.model_dump_json())
    ws2.send_text.assert_called_with(message.model_dump_json())
    ws3.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_send_to_user():
    manager = ConnectionManager()
    ws1 = AsyncMock(spec=WebSocket)
    ws1.client_state = WebSocketState.CONNECTED
    ws2 = AsyncMock(spec=WebSocket)
    ws2.client_state = WebSocketState.CONNECTED
    user_id = "test_user"

    await manager.connect(ws1, user_id, "testuser")
    await manager.connect(ws2, user_id, "testuser")

    ws1.send_text.reset_mock()
    ws2.send_text.reset_mock()

    message = ChatMessage(
        type=MessageType.MESSAGE, content="hello user", user_id="test"
    )
    await manager.send_to_user(user_id, message)

    ws1.send_text.assert_called_with(message.model_dump_json())
    ws2.send_text.assert_called_with(message.model_dump_json())


@pytest.mark.asyncio
async def test_handle_message():
    manager = ConnectionManager()
    websocket = AsyncMock(spec=WebSocket)
    websocket.client_state = WebSocketState.CONNECTED
    user_id = "test_user"

    await manager.connect(websocket, user_id, "testuser")

    raw_message = (
        '{"type": "message", "content": "test message", "user_id": "test_user"}'
    )
    processed = await manager.handle_message(websocket, raw_message)

    assert processed is True
    assert manager.message_counts[user_id]


@pytest.mark.asyncio
async def test_multiple_connections_per_user():
    manager = ConnectionManager()
    user_id = "test_user_multi"
    username = "Multi Connection User"

    ws1 = AsyncMock(spec=WebSocket)
    ws1.client_state = WebSocketState.CONNECTED
    ws2 = AsyncMock(spec=WebSocket)
    ws2.client_state = WebSocketState.CONNECTED

    # Connect twice
    await manager.connect(ws1, user_id, username)
    await manager.connect(ws2, user_id, username)

    assert manager.get_connection_count() == 2
    assert manager.get_user_count() == 1
    assert len(manager.user_connections[user_id]) == 2

    # Disconnect one
    await manager.disconnect(ws1)
    assert manager.get_connection_count() == 1
    assert manager.get_user_count() == 1
    assert len(manager.user_connections[user_id]) == 1

    # Disconnect the other
    await manager.disconnect(ws2)
    assert manager.get_connection_count() == 0
    assert manager.get_user_count() == 0
    assert user_id not in manager.user_connections
