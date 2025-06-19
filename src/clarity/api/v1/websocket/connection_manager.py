"""WebSocket connection management with advanced features."""

# removed - breaks FastAPI

import asyncio
from collections import defaultdict
import contextlib
from datetime import UTC, datetime
import logging
import time
from typing import Any
import uuid
from weakref import WeakSet

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from clarity.api.v1.websocket.models import (
    ConnectionInfo,
    ConnectionMessage,
    ErrorMessage,
    HeartbeatMessage,
    SystemMessage,
    WebSocketMessage,
)

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Advanced WebSocket connection manager with features for production use."""

    def __init__(
        self,
        heartbeat_interval: int = 30,  # seconds
        max_connections_per_user: int = 3,
        connection_timeout: int = 300,  # 5 minutes
        message_rate_limit: int = 60,  # messages per minute
        max_message_size: int = 64 * 1024,  # 64KB
    ) -> None:
        # Core connection storage
        self.connections: dict[str, WebSocket] = {}  # user_id -> websocket
        self.user_connections: dict[str, list[WebSocket]] = defaultdict(
            list
        )  # user_id -> [websockets]
        self.connection_info: dict[WebSocket, ConnectionInfo] = (
            {}
        )  # websocket -> connection info
        self.rooms: dict[str, set[str]] = defaultdict(set)  # room_id -> {user_ids}

        # Performance and security settings
        self.heartbeat_interval = heartbeat_interval
        self.max_connections_per_user = max_connections_per_user
        self.connection_timeout = connection_timeout
        self.message_rate_limit = message_rate_limit
        self.max_message_size = max_message_size

        # Rate limiting and monitoring
        self.message_counts: dict[str, list[float]] = defaultdict(
            list
        )  # user_id -> [timestamps]
        self.last_heartbeat: dict[WebSocket, float] = {}
        self.failed_connections: WeakSet[WebSocket] = WeakSet()

        # Background tasks (started explicitly during app startup)
        self._cleanup_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._started = False

    async def start_background_tasks(self) -> None:
        """Start background maintenance tasks (called during app startup)."""
        if self._started:
            return

        logger.info("Starting WebSocket background tasks...")

        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self._started = True
        logger.info("WebSocket background tasks started")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up stale connections and rate limiting data."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_stale_connections()
                self._cleanup_rate_limiting_data()
            except Exception:
                logger.exception("Error in cleanup loop")

    async def _heartbeat_loop(self) -> None:
        """Background task to send heartbeat messages."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self._send_heartbeats()
            except Exception:
                logger.exception("Error in heartbeat loop")

    async def _cleanup_stale_connections(self) -> None:
        """Remove connections that haven't responded to heartbeats."""
        current_time = time.time()
        stale_connections = []

        for websocket, last_heartbeat in self.last_heartbeat.items():
            if current_time - last_heartbeat > self.connection_timeout:
                stale_connections.append(websocket)

        for websocket in stale_connections:
            await self._force_disconnect(websocket, "Connection timeout")

    def _cleanup_rate_limiting_data(self) -> None:
        """Clean up old rate limiting data."""
        current_time = time.time()
        cutoff_time = current_time - 60  # Keep only last minute

        for user_id in list(self.message_counts.keys()):
            self.message_counts[user_id] = [
                timestamp
                for timestamp in self.message_counts[user_id]
                if timestamp > cutoff_time
            ]

            # Remove empty entries
            if not self.message_counts[user_id]:
                del self.message_counts[user_id]

    async def _send_heartbeats(self) -> None:
        """Send heartbeat messages to all connected clients."""
        if not self.connections:
            return

        heartbeat_message = HeartbeatMessage()
        message_str = heartbeat_message.model_dump_json()

        disconnected = []
        for websocket in list(self.connection_info.keys()):
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(message_str)
                else:
                    disconnected.append(websocket)
            except (RuntimeError, ConnectionError, OSError) as e:
                logger.warning("Failed to send heartbeat: %s", e)
                disconnected.append(websocket)

        # Clean up disconnected websockets
        for websocket in disconnected:
            await self._force_disconnect(websocket, "Heartbeat failed")

    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded message rate limit."""
        current_time = time.time()
        user_messages = self.message_counts[user_id]

        # Remove messages older than 1 minute
        message_expiry_seconds = 60
        recent_messages = [
            ts for ts in user_messages if current_time - ts < message_expiry_seconds
        ]
        self.message_counts[user_id] = recent_messages

        return len(recent_messages) < self.message_rate_limit

    def _record_message(self, user_id: str) -> None:
        """Record a message for rate limiting."""
        self.message_counts[user_id].append(time.time())

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        username: str,
        room_id: str = "general",
    ) -> bool:
        """Accept a new WebSocket connection with validation and limits.

        Returns:
            bool: True if connection was accepted, False if rejected
        """
        # Check connection limits
        if len(self.user_connections[user_id]) >= self.max_connections_per_user:
            await websocket.close(code=1008, reason="Too many connections")
            return False

        try:
            # Connection should already be accepted by the endpoint handler
            # Do not call websocket.accept() here as it can only be called once

            # Create connection info
            session_id = str(uuid.uuid4())
            connection_info = ConnectionInfo(
                user_id=user_id,
                username=username,
                session_id=session_id,
            )

            # Store connection data
            self.connections[session_id] = websocket
            self.user_connections[user_id].append(websocket)
            self.connection_info[websocket] = connection_info
            self.last_heartbeat[websocket] = time.time()

            # Add to room
            self.rooms[room_id].add(user_id)

            logger.info(
                "User %s (%s) connected with session %s", username, user_id, session_id
            )

            # Send connection acknowledgment
            ack_message = ConnectionMessage(
                user_id=user_id,
                session_id=session_id,
                server_info={
                    "server_time": datetime.now(UTC).isoformat(),
                    "heartbeat_interval": self.heartbeat_interval,
                },
            )
            await self.send_to_connection(websocket, ack_message)

            # Notify room about new user
            system_message = SystemMessage(
                content=f"{username} joined the chat", level="info"
            )
            await self.broadcast_to_room(room_id, system_message, exclude_user=user_id)

        except Exception:
            logger.exception("Error connecting user %s", user_id)
            with contextlib.suppress(Exception):
                await websocket.close(code=1011, reason="Server error")
            return False
        else:
            return True

    async def disconnect(
        self, websocket: WebSocket, reason: str = "Normal closure"
    ) -> None:
        """Handle WebSocket disconnection."""
        if websocket not in self.connection_info:
            return

        connection_info = self.connection_info[websocket]
        user_id = connection_info.user_id
        username = connection_info.username

        # Remove from storage
        self._remove_connection(websocket)

        # Find rooms user was in and notify
        rooms_to_notify = []
        for room_id, users in self.rooms.items():
            if user_id in users and user_id not in self.user_connections:
                # If the user has no more connections, they have left the room.
                users.discard(user_id)
                rooms_to_notify.append(room_id)

        # Notify rooms about user leaving
        for room_id in rooms_to_notify:
            system_message = SystemMessage(
                content=f"{username} left the chat", level="info"
            )
            await self.broadcast_to_room(room_id, system_message)

        logger.info("User %s (%s) disconnected: %s", username, user_id, reason)

    async def _force_disconnect(self, websocket: WebSocket, reason: str) -> None:
        """Force disconnect a WebSocket connection."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1000, reason=reason)
        except (RuntimeError, ConnectionError, OSError):
            pass

        await self.disconnect(websocket, reason)

    def _remove_connection(self, websocket: WebSocket) -> None:
        """Remove a connection from all storage structures."""
        if websocket not in self.connection_info:
            return

        connection_info = self.connection_info[websocket]
        user_id = connection_info.user_id
        session_id = connection_info.session_id

        # Remove from all storage
        self.connection_info.pop(websocket, None)
        self.connections.pop(session_id, None)
        self.last_heartbeat.pop(websocket, None)

        # Remove from user connections
        if user_id in self.user_connections:
            try:
                self.user_connections[user_id].remove(websocket)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            except ValueError:
                pass

    async def send_to_connection(
        self, websocket: WebSocket, message: WebSocketMessage
    ) -> None:
        """Send a message to a specific WebSocket connection."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                message_str = message.model_dump_json()

                # Check message size
                if len(message_str.encode("utf-8")) > self.max_message_size:
                    error_msg = ErrorMessage(
                        error_code="MESSAGE_TOO_LARGE",
                        message="Message exceeds maximum size limit",
                    )
                    await websocket.send_text(error_msg.model_dump_json())
                    return

                await websocket.send_text(message_str)
            else:
                await self._force_disconnect(websocket, "Connection not active")
        except Exception:
            logger.exception("Error sending message")
            await self._force_disconnect(websocket, "Send error")

    async def send_to_user(self, user_id: str, message: WebSocketMessage) -> None:
        """Send a message to all connections of a specific user."""
        connections = self.user_connections.get(user_id, [])
        if not connections:
            return

        tasks = [
            self.send_to_connection(websocket, message) for websocket in connections
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_to_room(
        self, room_id: str, message: WebSocketMessage, exclude_user: str | None = None
    ) -> None:
        """Broadcast a message to all users in a room."""
        users_in_room = self.rooms.get(room_id, set())
        tasks = [
            self.send_to_user(user_id, message)
            for user_id in users_in_room
            if user_id != exclude_user
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_to_all(
        self, message: WebSocketMessage, exclude_user: str | None = None
    ) -> None:
        """Broadcast a message to all connected users."""
        tasks = [
            self.send_to_user(user_id, message)
            for user_id in self.user_connections
            if user_id != exclude_user
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_message(self, websocket: WebSocket, raw_message: str) -> bool:
        """Handle an incoming WebSocket message with validation and rate limiting.

        Returns:
            bool: True if message was processed successfully
        """
        if websocket not in self.connection_info:
            return False

        connection_info = self.connection_info[websocket]
        user_id = connection_info.user_id

        # Check message size
        if len(raw_message.encode("utf-8")) > self.max_message_size:
            error_msg = ErrorMessage(
                error_code="MESSAGE_TOO_LARGE",
                message="Message exceeds maximum size limit",
            )
            await self.send_to_connection(websocket, error_msg)
            return False

        # Check rate limiting
        if not self._check_rate_limit(user_id):
            error_msg = ErrorMessage(
                error_code="RATE_LIMIT_EXCEEDED",
                message="Too many messages, please slow down",
            )
            await self.send_to_connection(websocket, error_msg)
            return False

        # Record message for rate limiting
        self._record_message(user_id)

        # Update last seen
        connection_info.last_seen = datetime.now(UTC)
        self.last_heartbeat[websocket] = time.time()

        return True

    def get_room_users(self, room_id: str) -> list[str]:
        """Get list of users in a specific room."""
        return list(self.rooms.get(room_id, set()))

    def get_user_count(self) -> int:
        """Get total number of connected users."""
        return len(self.user_connections)

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.connection_info)

    def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get information about a connected user."""
        connections = self.user_connections.get(user_id, [])
        if not connections:
            return None

        # Get info from first connection
        first_connection = connections[0]
        if first_connection in self.connection_info:
            info = self.connection_info[first_connection]
            return {
                "user_id": info.user_id,
                "username": info.username,
                "connected_at": info.connected_at,
                "last_seen": info.last_seen,
                "connection_count": len(connections),
            }

        return None

    async def stop_background_tasks(self) -> None:
        """Stop background tasks (called during app shutdown)."""
        if not self._started:
            return

        logger.info("Stopping WebSocket background tasks...")

        # Cancel background tasks
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        self._started = False
        logger.info("WebSocket background tasks stopped")

    async def shutdown(self) -> None:
        """Gracefully shutdown the connection manager."""
        logger.info("Shutting down WebSocket connection manager...")

        # Stop background tasks first
        await self.stop_background_tasks()

        # Close all connections
        for websocket in list(self.connection_info.keys()):
            await self._force_disconnect(websocket, "Server shutdown")

        logger.info("WebSocket connection manager shutdown complete")


# Connection manager will be created during app startup
# Use get_connection_manager() dependency to access it
