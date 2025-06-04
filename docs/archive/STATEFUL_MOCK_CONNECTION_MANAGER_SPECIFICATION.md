# Stateful Mock ConnectionManager Specification - Clarity Loop Backend

## Date: 2025-06-04

## 1. Purpose

This document outlines the specification for `TestConnectionManager`, a stateful mock implementation of the production `ConnectionManager`. Its purpose is to enable reliable and accurate testing of WebSocket functionality in `tests/api/v1/test_websocket.py` by providing the `chat_handler.py` with a simulated environment that mirrors the essential state and behavior of the real `ConnectionManager`.

This mock directly addresses the root cause of persistent WebSocket test failures, which was the use of a stateless `MagicMock` that could not fulfill the state-dependent expectations of the `chat_handler.py`.

## 2. Scope

This mock will simulate the following aspects of the production `ConnectionManager` relevant to the `chat_handler.py`:

* Connection tracking (WebSocket objects, user IDs, room IDs).
* Storage and retrieval of `ConnectionInfo` objects.
* Basic room membership management.
* Message broadcasting and direct sending (primarily for recording/assertion).
* Heartbeat handling.
* Rate limiting (simplified for testing).

This mock is **not** intended to replicate:

* Complex background tasks (e.g., automated cleanup of stale connections, periodic heartbeats initiated by the manager).
* Deep database interactions or external service calls.
* Performance characteristics of the production manager.

## 3. Attributes

The `TestConnectionManager` class will maintain the following internal attributes to manage state:

* `active_websockets: Set[WebSocket]`
  * Description: A set of all currently "connected" WebSocket objects.
* `user_connections: Dict[str, List[WebSocket]]`
  * Description: Maps `user_id` (str) to a list of `WebSocket` objects associated with that user.
  * Implementation: Uses `collections.defaultdict(list)`.
* `connection_info: Dict[WebSocket, ConnectionInfo]`
  * Description: The most critical state. Maps a `WebSocket` object to its corresponding `ConnectionInfo` Pydantic model instance.
  * `ConnectionInfo` (from `src/clarity/api/v1/websocket/models.py`) includes fields like `user_id`, `room_id`, `session_id`, `username`, `connected_at`, `last_active`, `last_heartbeat_ack`, `message_timestamps`.
* `rooms: Dict[str, Set[str]]`
  * Description: Maps `room_id` (str) to a set of `user_id`s currently in that room.
  * Implementation: Uses `collections.defaultdict(set)`.
* `messages_sent: List[Dict[str, Any]]`
  * Description: A list to record messages sent via `send_to_connection` or `broadcast_to_room`. Each entry can be a dictionary detailing the target, message, and type of send (direct/broadcast). Useful for test assertions.
* `heartbeat_interval: int`
* `max_connections_per_user: int`
* `connection_timeout: int`
* `message_rate_limit: int`
* `max_message_size: int`
  * Description: Configuration parameters, initialized in `__init__`. Can be used to simulate certain checks if needed, but primarily for completeness of the interface.

## 4. Method Specifications

### 4.1. `__init__(self, heartbeat_interval: int = 30, ...)`

* **Behavior**: Initializes all state attributes (e.g., `active_websockets = set()`, `connection_info = {}`, `rooms = defaultdict(set)`, `user_connections = defaultdict(list)`, `messages_sent = []`) and configuration parameters.

### 4.2. `async connect(self, websocket: WebSocket, user_id: str, room_id: str, session_id: str, username: Optional[str]) -> ConnectionInfo:`

* **Behavior**:
    1. **DO NOT call `await websocket.accept()`**. The `TestClient` manages the WebSocket handshake. This mock simulates what happens *after* the handshake.
    2. Check if `max_connections_per_user` would be exceeded for `user_id`. If so, raise `WebSocketDisconnect` (or a custom exception that tests can expect).
    3. Create a `ConnectionInfo` instance using the provided arguments and current timestamp for `connected_at`, `last_active`. Initialize `message_timestamps` as an empty list.
    4. Store the `ConnectionInfo` object: `self.connection_info[websocket] = new_connection_info`.
    5. Add the WebSocket to tracking: `self.active_websockets.add(websocket)`.
    6. Update room membership: `self.rooms[room_id].add(user_id)`.
    7. Update user's connections: `self.user_connections[user_id].append(websocket)`.
    8. Log the connection event.
    9. Return the created `ConnectionInfo` object.

### 4.3. `async disconnect(self, websocket: WebSocket, reason: Optional[str] = None) -> None:`

* **Behavior**:
    1. Retrieve `connection_info = self.connection_info.get(websocket)`. If not found, log and return (or raise an error if strictness is desired).
    2. Remove from `self.connection_info`: `del self.connection_info[websocket]`.
    3. Remove from `self.active_websockets`: `self.active_websockets.discard(websocket)`.
    4. Update `self.user_connections`: Remove `websocket` from `self.user_connections[connection_info.user_id]`. If the list becomes empty, `del self.user_connections[connection_info.user_id]`.
    5. Update `self.rooms`:
        * Check if `connection_info.user_id` has any other active connections (via `self.user_connections`) to `connection_info.room_id`.
        * If not, remove `connection_info.user_id` from `self.rooms[connection_info.room_id]`.
        * If `self.rooms[connection_info.room_id]` becomes empty, `del self.rooms[connection_info.room_id]`.
    6. Log the disconnection event with the reason.

### 4.4. `async send_to_connection(self, websocket: WebSocket, message: Any) -> None:`

* **Behavior**:
    1. Verify `websocket` is in `self.connection_info`.
    2. Record the attempt: `self.messages_sent.append({"type": "direct", "target_ws": websocket, "message": message_content})`. (Here `message_content` would be the serialized form of `message` if it's a Pydantic model, or `message` itself if it's already a dict/str).
    3. Log the send attempt.
    4. (Optional) Simulate the actual send by calling `await websocket.send_json(message_content)` or `send_text` if tests need to verify client-side reception via `TestClient`'s WebSocket. For many handler tests, just recording is sufficient.

### 4.5. `async broadcast_to_room(self, room_id: str, message: Any, exclude_websocket: Optional[WebSocket] = None) -> None:`

* **Behavior**:
    1. Identify all WebSockets connected to `room_id` using `self.user_connections` and `self.connection_info` (or by iterating `self.connection_info` and checking `ci.room_id`).
    2. Record the broadcast: `self.messages_sent.append({"type": "broadcast", "room_id": room_id, "message": message_content, "excluded": exclude_websocket})`.
    3. For each target WebSocket (not `exclude_websocket`):
        * (Optional) Simulate send: `await target_ws.send_json(message_content)`.
    4. Log the broadcast attempt.

### 4.6. `async handle_heartbeat(self, websocket: WebSocket) -> None:`

* **Behavior**:
    1. Retrieve `connection_info = self.connection_info.get(websocket)`.
    2. If found, update `connection_info.last_heartbeat_ack = datetime.now(timezone.utc)` and `connection_info.last_active = datetime.now(timezone.utc)`.
    3. Log the heartbeat.

### 4.7. `def is_rate_limited(self, websocket: WebSocket) -> bool:`

* **Behavior (Simplified for most tests)**:
    1. Retrieve `connection_info = self.connection_info.get(websocket)`.
    2. If found:
        * Add `datetime.now(timezone.utc)` to `connection_info.message_timestamps`.
        * Implement a simple check (e.g., count timestamps within the last second). If it exceeds a test-friendly limit (e.g., `self.message_rate_limit` per second), return `True`.
        * Prune old timestamps from `connection_info.message_timestamps`.
    3. Default to `False` if no specific rate-limiting scenario is being tested.
    4. Log the check.

### 4.8. Getter Methods (Synchronous)

* `get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]`:
  * Can return a simplified dict based on `ConnectionInfo` for the first connection of the user, or just a placeholder.
* `get_room_users(self, room_id: str) -> Set[str]`:
  * Return `self.rooms.get(room_id, set())`.
* `get_user_count(self, room_id: str) -> int`:
  * Return `len(self.rooms.get(room_id, set()))`.
* `get_connection_count(self) -> int`:
  * Return `len(self.active_websockets)`.

## 5. Integration with Pytest Fixtures

* An instance of `TestConnectionManager` will be provided by a pytest fixture (e.g., `mock_test_connection_manager`).
* The `app` fixture in `test_websocket.py` will use this instance to override the `chat_handler.get_connection_manager` dependency.

## 6. Logging

All significant actions within the mock (connect, disconnect, send, broadcast, heartbeat) should include `logger.info()` or `logger.debug()` statements to aid in test debugging and tracing the mock's behavior.

This specification provides a solid foundation for building a `TestConnectionManager` that will significantly improve the reliability and accuracy of WebSocket tests in the Clarity Loop Backend.
