"""WebSocket data models for structured communication."""

# removed - breaks FastAPI

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MessageType(StrEnum):
    """Message types for WebSocket communication."""

    # Chat messages
    MESSAGE = "message"
    SYSTEM = "system"
    ERROR = "error"

    # Health insights
    HEALTH_INSIGHT = "health_insight"
    ANALYSIS_UPDATE = "analysis_update"

    # User status
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    TYPING = "typing"

    # Connection management
    CONNECTION_ACK = "connection_ack"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"


class BaseMessage(BaseModel):
    """Base message structure for all WebSocket communications."""

    model_config = ConfigDict(str_strip_whitespace=True)

    type: MessageType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    message_id: str | None = None


class ChatMessage(BaseMessage):
    """Chat message for user communication."""

    type: MessageType = MessageType.MESSAGE
    content: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., min_length=1, max_length=100)
    username: str | None = None
    metadata: dict[str, Any] | None = None


class SystemMessage(BaseMessage):
    """System notification message."""

    type: MessageType = MessageType.SYSTEM
    content: str
    level: str = Field(default="info")  # info, warning, error


class ErrorMessage(BaseMessage):
    """Error message for communication failures."""

    type: MessageType = MessageType.ERROR
    error_code: str
    message: str
    details: dict[str, Any] | None = None


class HealthInsightMessage(BaseMessage):
    """Health insight from AI analysis."""

    type: MessageType = MessageType.HEALTH_INSIGHT
    user_id: str
    insight: str = Field(..., min_length=1, max_length=5000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    category: str
    source_data: dict[str, Any] | None = None
    recommendations: list[str] | None = None


class AnalysisUpdateMessage(BaseMessage):
    """Real-time analysis progress update."""

    type: MessageType = MessageType.ANALYSIS_UPDATE
    user_id: str
    status: str  # "started", "processing", "completed", "failed"
    progress: int = Field(..., ge=0, le=100)
    details: str | None = None


class UserStatusMessage(BaseMessage):
    """User status change notification."""

    type: MessageType = MessageType.USER_JOINED
    user_id: str
    username: str
    status: str = "online"  # online, offline, away


class TypingMessage(BaseMessage):
    """Typing indicator message."""

    type: MessageType = MessageType.TYPING
    user_id: str
    username: str
    is_typing: bool


class ConnectionMessage(BaseMessage):
    """Connection acknowledgment message."""

    type: MessageType = MessageType.CONNECTION_ACK
    user_id: str
    session_id: str
    server_info: dict[str, Any] | None = None


class HeartbeatMessage(BaseMessage):
    """Heartbeat message for connection health."""

    type: MessageType = MessageType.HEARTBEAT
    client_timestamp: datetime | None = None


class HeartbeatAckMessage(BaseMessage):
    """Heartbeat acknowledgment message."""

    type: MessageType = MessageType.HEARTBEAT_ACK
    client_timestamp: datetime | None = None
    server_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# Union type for all possible WebSocket messages
WebSocketMessage = (
    ChatMessage
    | SystemMessage
    | ErrorMessage
    | HealthInsightMessage
    | AnalysisUpdateMessage
    | UserStatusMessage
    | TypingMessage
    | ConnectionMessage
    | HeartbeatMessage
    | HeartbeatAckMessage
)


class ConnectionInfo(BaseModel):
    """Information about a WebSocket connection."""

    user_id: str
    username: str
    session_id: str
    connected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))
    connection_count: int = 1
    metadata: dict[str, Any] | None = None


class RoomInfo(BaseModel):
    """Information about a chat room or channel."""

    room_id: str
    name: str
    description: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    max_users: int = 100
    active_users: list[str] = Field(default_factory=list)
    permissions: dict[str, Any] | None = None


# Added for health data validation from WebSocket messages
class InvalidWebSocketDataError(ValueError):
    """Custom exception for invalid WebSocket data payloads."""


class ActigraphyDataPointSchema(BaseModel):
    """Schema for individual actigraphy data points received via WebSocket."""

    timestamp: datetime
    value: float = Field(..., description="Activity count or acceleration value")


class WebSocketHealthDataPayload(BaseModel):
    """Schema for the 'data' or 'content' field of a health_data WebSocket message."""

    data_points: list[ActigraphyDataPointSchema] = Field(default_factory=list)
    steps: float | None = None  # Optional steps, can be derived or primary
    # Add other fields if the client might send them and trigger_health_analysis might use them
    # e.g., sampling_rate, duration_hours, if not to be determined server-side primarily
