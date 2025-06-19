"""WebSocket chat handler for real-time health insights and communication."""

# removed - breaks FastAPI

from datetime import UTC, datetime
import json
import logging
import os
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import ValidationError

from clarity.api.v1.websocket.connection_manager import ConnectionManager
from clarity.api.v1.websocket.lifespan import get_connection_manager
from clarity.api.v1.websocket.models import (
    ChatMessage,
    ErrorMessage,
    HeartbeatAckMessage,
    InvalidWebSocketDataError,
    MessageType,
    SystemMessage,
    TypingMessage,
    WebSocketHealthDataPayload,
)
from clarity.auth.aws_cognito_provider import CognitoAuthProvider
from clarity.core.config_aws import get_settings

# Removed circular import - will use direct initialization
from clarity.ml.gemini_service import (
    GeminiService,
    HealthInsightRequest,
)
from clarity.ml.pat_service import (
    ActigraphyAnalysis,
    ActigraphyInput,
    PATModelService,
    get_pat_service,
)
from clarity.ml.preprocessing import ActigraphyDataPoint
from clarity.models.auth import UserContext, UserRole

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


def get_gemini_service() -> GeminiService:
    # Direct initialization to avoid circular import
    return GeminiService(
        project_id=os.getenv("AWS_PROJECT_NAME", "clarity-digital-twin")
    )


def get_pat_model_service() -> PATModelService:
    # Direct initialization to avoid circular import
    service = get_pat_service()
    if not isinstance(service, PATModelService):
        msg = f"Expected PATModelService, got {type(service).__name__}"
        raise TypeError(msg)
    return service


class WebSocketChatHandler:
    """Handles WebSocket chat functionality with health insights integration."""

    def __init__(
        self,
        gemini_service: GeminiService,
        pat_service: PATModelService,
    ) -> None:
        self.gemini_service = gemini_service
        self.pat_service = pat_service

    async def process_chat_message(
        self,
        chat_message: ChatMessage,
        connection_manager: ConnectionManager,
    ) -> None:
        logger.info("Processing chat message for user %s", chat_message.user_id)
        # Add conversation context for Gemini
        user_query = chat_message.content

        # Use Gemini service to generate response
        gemini_request = HealthInsightRequest(
            user_id=chat_message.user_id,
            analysis_results={},  # Assuming chat messages don't have analysis results directly
            context=user_query,
            insight_type="chat_response",
        )
        try:
            gemini_response = await self.gemini_service.generate_health_insights(
                gemini_request
            )
            # Extract content from narrative or key_insights
            ai_response_content = gemini_response.narrative
        except Exception:
            logger.exception("Error generating Gemini response")
            ai_response_content = (
                "I am sorry, I could not generate a response at this time."
            )

        response_message = ChatMessage(
            user_id="AI",
            timestamp=datetime.now(UTC),
            type=MessageType.MESSAGE,
            content=ai_response_content,
        )
        await connection_manager.send_to_user(chat_message.user_id, response_message)

    @staticmethod
    async def process_typing_message(
        typing_message: TypingMessage,
        connection_manager: ConnectionManager,
        room_id: str,
    ) -> None:
        logger.info(
            "Processing typing indicator for user %s: is_typing=%s",
            typing_message.user_id,
            typing_message.is_typing,
        )
        # Broadcast typing status to the room where the user is connected
        # Note: We would need the websocket reference to exclude the sender properly
        await connection_manager.broadcast_to_room(
            room_id,
            typing_message,
        )

    @staticmethod
    async def process_heartbeat(
        websocket: WebSocket,
        message: dict[str, Any],
        connection_manager: ConnectionManager,
    ) -> None:
        client_ts_iso = message.get("client_timestamp")
        client_timestamp_dt: datetime | None = None
        if client_ts_iso:
            try:
                client_timestamp_dt = datetime.fromisoformat(client_ts_iso)
            except ValueError:
                logger.warning(
                    "Invalid client_timestamp format in heartbeat: %s", client_ts_iso
                )

        logger.info("Processing heartbeat for websocket %s", websocket)
        # The ConnectionManager handles last active time internally
        # Acknowledge heartbeat
        heartbeat_ack_message = HeartbeatAckMessage(
            type=MessageType.HEARTBEAT_ACK, client_timestamp=client_timestamp_dt
        )
        await connection_manager.send_to_connection(websocket, heartbeat_ack_message)

    async def trigger_health_analysis(
        self,
        user_id: str,
        health_data_payload: WebSocketHealthDataPayload,
        connection_manager: ConnectionManager,
    ) -> None:
        # HIPAA-compliant logging - no PHI data
        logger.info(
            "Triggering health analysis for user %s with %d data points",
            user_id,
            len(health_data_payload.data_points),
        )
        try:
            duration_hours = 24
            data_points_from_payload = health_data_payload.data_points

            if data_points_from_payload and len(data_points_from_payload) > 1:
                timestamps = [dp.timestamp for dp in data_points_from_payload]
                if timestamps:
                    min_ts = min(timestamps)
                    max_ts = max(timestamps)
                    duration_seconds = (max_ts - min_ts).total_seconds()
                    if duration_seconds > 0:
                        duration_hours = max(1, int(duration_seconds / 3600))
            elif data_points_from_payload and len(data_points_from_payload) == 1:
                duration_hours = 1

            actigraphy_data_points = [
                ActigraphyDataPoint(timestamp=dp.timestamp, value=dp.value)
                for dp in data_points_from_payload
            ]

            actigraphy_input = ActigraphyInput(
                user_id=user_id,
                data_points=actigraphy_data_points,
                sampling_rate=1.0,
                duration_hours=duration_hours,
            )

            pat_analysis_results: ActigraphyAnalysis = (
                await self.pat_service.analyze_actigraphy(actigraphy_input)
            )

            insight_request = HealthInsightRequest(
                user_id=user_id,
                analysis_results=pat_analysis_results.model_dump(),
                context="Based on recent health data.",
                insight_type="health_analysis",
            )
            insight_response = await self.gemini_service.generate_health_insights(
                insight_request
            )

            insight_message = ChatMessage(
                user_id="AI",
                timestamp=datetime.now(UTC),
                type=MessageType.MESSAGE,
                content=insight_response.narrative,
            )
            await connection_manager.send_to_user(user_id, insight_message)

        except WebSocketDisconnect:
            logger.info(
                "Health analysis interrupted by client disconnect for user %s", user_id
            )
            raise
        except Exception:
            logger.exception("Error during health analysis")
            error_msg = ErrorMessage(
                error_code="HEALTH_ANALYSIS_ERROR",
                message="Failed to perform health analysis",
            )
            await connection_manager.send_to_user(user_id, error_msg)


@router.websocket("/{room_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    room_id: str = "general",
    token: str | None = Query(...),
    connection_manager: ConnectionManager = Depends(get_connection_manager),
    gemini_service: GeminiService = Depends(get_gemini_service),
    pat_service: PATModelService = Depends(get_pat_model_service),
) -> None:
    """WebSocket endpoint for real-time chat with health insights.

    Features:
    - Real-time messaging
    - Health insights generation
    - Typing indicators
    - Connection management
    - Rate limiting
    - Heartbeat monitoring
    """
    # Authenticate the user first
    current_user = await _authenticate_websocket_user(token, websocket)
    if not current_user:
        # _authenticate_websocket_user already closed the connection
        return

    handler = WebSocketChatHandler(
        gemini_service=gemini_service, pat_service=pat_service
    )
    user_id = current_user.user_id
    username = str(current_user.email)  # Ensure username is string

    await connection_manager.connect(websocket, user_id, username, room_id)
    logger.info("User %s (%s) connected to room %s", user_id, username, room_id)

    try:  # noqa: PLR1702 - WebSocket handler complexity
        while True:
            try:
                raw_message = await websocket.receive_text()

                # SECURITY: Pre-parse message size validation to prevent DoS
                max_message_size = 64 * 1024  # 64KB limit
                if len(raw_message.encode("utf-8")) > max_message_size:
                    error_detail = f"Message size exceeds {max_message_size} bytes"
                    error_msg = ErrorMessage(
                        error_code="MESSAGE_TOO_LARGE",
                        message=error_detail,
                        details={"max_size": max_message_size},
                    )
                    await websocket.send_text(error_msg.model_dump_json())
                    continue

                if not await connection_manager.handle_message(websocket, raw_message):
                    continue

                try:
                    # SECURITY: Safe JSON parsing with try-catch for malformed data
                    message_data = json.loads(raw_message)
                    message_type = message_data.get("type")

                    if message_type == MessageType.MESSAGE.value:
                        chat_msg: ChatMessage = ChatMessage(**message_data)
                        await handler.process_chat_message(chat_msg, connection_manager)

                    elif message_type == MessageType.TYPING.value:
                        typing_msg: TypingMessage = TypingMessage(**message_data)
                        # Pass the room_id from the endpoint directly
                        await WebSocketChatHandler.process_typing_message(
                            typing_msg, connection_manager, room_id
                        )

                    elif message_type == MessageType.HEARTBEAT.value:
                        await WebSocketChatHandler.process_heartbeat(
                            websocket, message_data, connection_manager
                        )

                    elif message_type == MessageType.HEALTH_INSIGHT.value:
                        health_data_content = message_data.get("content", {})
                        user_id_from_message = message_data.get("user_id", "")
                        if user_id_from_message:
                            try:
                                # SECURITY: Validate data points count before processing
                                data_points = health_data_content.get("data_points", [])
                                max_data_points = 10080  # 1 week of minute-level data
                                if len(data_points) > max_data_points:
                                    error_message = f"Exceeded maximum data points limit of {max_data_points}"
                                    error_msg = ErrorMessage(
                                        error_code="TOO_MANY_DATA_POINTS",
                                        message=error_message,
                                        details={
                                            "received": len(data_points),
                                            "max_allowed": max_data_points,
                                        },
                                    )
                                    await websocket.send_text(
                                        error_msg.model_dump_json()
                                    )
                                    continue

                                validated_payload = (
                                    WebSocketHealthDataPayload.model_validate(
                                        health_data_content
                                    )
                                )
                                await handler.trigger_health_analysis(
                                    user_id_from_message,
                                    validated_payload,
                                    connection_manager,
                                )
                            except ValidationError as e:
                                logger.warning(
                                    "Invalid HEALTH_INSIGHT content payload: %s. Errors: %s",
                                    health_data_content,
                                    e.errors(),
                                )
                                error_msg = ErrorMessage(
                                    error_code="INVALID_HEALTH_INSIGHT_CONTENT",
                                    message="Health insight content validation failed.",
                                    details={
                                        "validation_errors": json.dumps(e.errors())
                                    },
                                )
                                await connection_manager.send_to_connection(
                                    websocket, error_msg
                                )
                            except InvalidWebSocketDataError as e:
                                logger.warning(
                                    "Invalid WebSocket health data for insight: %s", e
                                )
                                error_msg = ErrorMessage(
                                    error_code="INVALID_HEALTH_INSIGHT_DATA",
                                    message=str(e),
                                )
                                await connection_manager.send_to_connection(
                                    websocket, error_msg
                                )
                        else:
                            logger.warning("User ID not found in health data message.")

                    else:
                        logger.warning("Unknown message type: %s", message_type)
                        error_msg = ErrorMessage(
                            error_code="UNKNOWN_MESSAGE_TYPE",
                            message=f"Unknown message type: {message_type}",
                        )
                        await connection_manager.send_to_connection(
                            websocket, error_msg
                        )

                except json.JSONDecodeError:
                    error_msg = ErrorMessage(
                        error_code="INVALID_JSON", message="Invalid JSON format"
                    )
                    await connection_manager.send_to_connection(websocket, error_msg)

                except Exception:
                    logger.exception("Error parsing message")
                    error_msg = ErrorMessage(
                        error_code="MESSAGE_PARSE_ERROR",
                        message="Failed to parse message",
                    )
                    await connection_manager.send_to_connection(websocket, error_msg)

            except WebSocketDisconnect as e:
                logger.warning(
                    "WebSocket disconnected: code=%s, reason=%s", e.code, e.reason
                )
                break

            except Exception:
                logger.exception("Unexpected error in chat endpoint")
                await connection_manager.disconnect(websocket, "internal_error")
                break

    except Exception:
        logger.exception("Error in WebSocket connection")

    finally:
        await connection_manager.disconnect(websocket, "Connection closed")
        logger.info(
            "WebSocket connection closed for %s",
            username if "username" in locals() else "unknown user",
        )


def _extract_username(user: UserContext | None, user_id: str) -> str:
    # Ensure we have a username; fallback with generated username
    if user and user.email:
        return user.email
    return f"User_{user_id[:8]}"


async def _authenticate_websocket_user(
    token: str | None, websocket: WebSocket
) -> UserContext | None:
    """Authenticate WebSocket user and handle connection closing."""
    if not token:
        await websocket.close(code=4001, reason="Authentication token is required")
        return None

    try:
        # Get auth provider directly to avoid circular import
        auth_provider = CognitoAuthProvider(
            user_pool_id=os.getenv("COGNITO_USER_POOL_ID", ""),
            client_id=os.getenv("COGNITO_CLIENT_ID", ""),
            region=os.getenv("COGNITO_REGION", "us-east-1"),
        )
        user_info = await auth_provider.verify_token(token)

        if not user_info:
            await websocket.close(code=4003, reason="Invalid or expired token")
            return None

        # Use the provider to create the full user context
        if hasattr(auth_provider, "get_or_create_user_context"):
            user_context = await auth_provider.get_or_create_user_context(user_info)
            if not isinstance(user_context, UserContext):
                msg = f"Expected UserContext, got {type(user_context).__name__}"
                raise TypeError(msg)
            return user_context
        # Fallback for providers without the enhanced method
        # This part might need adjustment based on what verify_token returns
        # For now, assuming it returns a dict that can be used to build a basic context
        # Create basic user context from token info
        return UserContext(
            user_id=user_info.get("sub", user_info.get("user_id", "unknown")),
            email=user_info.get("email", ""),
            role=UserRole.PATIENT,  # Default role
            permissions=[],
            is_verified=user_info.get("email_verified", False),
            is_active=True,
            custom_claims=user_info,
        )

    except Exception:
        logger.exception("WebSocket authentication failed")
        await websocket.close(code=4003, reason="Authentication failed")
        return None


async def _handle_health_analysis_message(
    raw_message: str,
    user_id: str,
    handler: WebSocketChatHandler,
    connection_manager: ConnectionManager,
    websocket: WebSocket,
) -> None:
    try:
        message_data = json.loads(raw_message)
        msg_type = message_data.get("type")

        if msg_type == "health_data":
            health_data_content = message_data.get("data", {})
            try:
                validated_payload = WebSocketHealthDataPayload.model_validate(
                    health_data_content
                )
                await handler.trigger_health_analysis(
                    user_id, validated_payload, connection_manager
                )
            except ValidationError as e:
                logger.warning(
                    "Invalid health_data payload: %s. Errors: %s",
                    health_data_content,
                    e.errors(),
                )
                error_msg = ErrorMessage(
                    error_code="INVALID_HEALTH_DATA_PAYLOAD",
                    message="Health data payload validation failed.",
                    details={"validation_errors": json.dumps(e.errors())},
                )
                await connection_manager.send_to_connection(websocket, error_msg)
            except InvalidWebSocketDataError as e:
                logger.warning("Invalid WebSocket health data: %s", e)
                error_msg = ErrorMessage(
                    error_code="INVALID_HEALTH_DATA", message=str(e)
                )
                await connection_manager.send_to_connection(websocket, error_msg)

        elif msg_type == MessageType.HEARTBEAT.value:
            await WebSocketChatHandler.process_heartbeat(
                websocket, message_data, connection_manager
            )
        else:
            logger.warning("Unknown message type: %s", msg_type)
            error_msg = ErrorMessage(
                error_code="UNKNOWN_MESSAGE_TYPE",
                message=f"Unknown message type: {msg_type}",
            )
            await connection_manager.send_to_connection(websocket, error_msg)
    except json.JSONDecodeError:
        error_msg = ErrorMessage(
            error_code="INVALID_JSON", message="Invalid JSON format"
        )
        await connection_manager.send_to_connection(websocket, error_msg)
    except Exception:
        logger.exception("Error parsing message")
        error_msg = ErrorMessage(
            error_code="MESSAGE_PARSE_ERROR",
            message="Failed to parse message",
        )
        await connection_manager.send_to_connection(websocket, error_msg)


@router.websocket("/health-analysis/{user_id}")
async def websocket_health_analysis_endpoint(
    websocket: WebSocket,
    user_id: str,
    token: str | None = Query(...),
    connection_manager: ConnectionManager = Depends(get_connection_manager),
    gemini_service: GeminiService = Depends(get_gemini_service),
    pat_service: PATModelService = Depends(get_pat_model_service),
) -> None:
    """WebSocket endpoint for real-time health analysis updates.

    This endpoint provides real-time updates during health data processing,
    including PAT analysis and AI insight generation.
    """
    # Authenticate the user first
    current_user = await _authenticate_websocket_user(token, websocket)
    if not current_user or current_user.user_id != user_id:
        if current_user:
            logger.warning(
                "Authenticated user %s does not match path user %s",
                current_user.user_id,
                user_id,
            )
        await websocket.close(code=4003, reason="Unauthorized")
        return

    handler = WebSocketChatHandler(
        gemini_service=gemini_service, pat_service=pat_service
    )
    logger.info("WebSocket connection attempt: %s", token)
    try:
        await websocket.accept()
        username = _extract_username(current_user, user_id)
        await connection_manager.connect(
            websocket,
            user_id,
            username,
            f"health_analysis_{user_id}",
        )
        logger.info("Health analysis WebSocket connected for %s", username)
        welcome_msg = SystemMessage(
            content="Connected to health analysis service. Send health data to start analysis.",
            level="info",
        )
        await connection_manager.send_to_connection(websocket, welcome_msg)
        while True:
            try:
                raw_message = await websocket.receive_text()
                if not await connection_manager.handle_message(websocket, raw_message):
                    continue
                await _handle_health_analysis_message(
                    raw_message, user_id, handler, connection_manager, websocket
                )
            except WebSocketDisconnect as e:
                logger.warning(
                    "WebSocket disconnected: code=%s, reason=%s", e.code, e.reason
                )
                break
            except Exception:
                logger.exception("Error in health analysis WebSocket")
                await connection_manager.disconnect(websocket, "internal_error")
                break
    except Exception:
        logger.exception("Error in health analysis WebSocket connection")
    finally:
        await connection_manager.disconnect(
            websocket, "Health analysis connection closed"
        )
        logger.info("Health analysis WebSocket closed for %s", user_id)


@router.get("/chat/stats")
async def get_chat_stats(
    connection_manager: ConnectionManager = Depends(get_connection_manager),
) -> dict[str, Any]:
    """Get current chat statistics."""
    return {
        "total_users": connection_manager.get_user_count(),
        "total_connections": connection_manager.get_connection_count(),
        "rooms": {
            room_id: len(users) for room_id, users in connection_manager.rooms.items()
        },
    }


@router.get("/chat/users/{room_id}")
async def get_room_users(
    room_id: str,
    connection_manager: ConnectionManager = Depends(get_connection_manager),
) -> dict[str, Any]:
    """Get list of users in a specific room."""
    users = connection_manager.get_room_users(room_id)
    user_info = []

    for user_id in users:
        info = connection_manager.get_user_info(user_id)
        if info:
            user_info.append(info)

    return {"room_id": room_id, "users": user_info}
