"""WebSocket chat handler for real-time health insights and communication."""

import asyncio
from datetime import UTC, datetime
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)

from clarity.auth.firebase_auth import get_current_user_websocket
from clarity.core.config import get_settings
from clarity.core.container import get_container
from clarity.ml.gemini_service import (
    GeminiService,
    HealthInsightRequest,
)

if TYPE_CHECKING:
    from clarity.ml.pat_service import ActigraphyAnalysis
from clarity.api.v1.websocket.connection_manager import ConnectionManager
from clarity.api.v1.websocket.lifespan import get_connection_manager
from clarity.api.v1.websocket.models import (
    ChatMessage,
    ErrorMessage,
    HeartbeatMessage,
    MessageType,
    SystemMessage,
    TypingMessage,
)
from clarity.ml.pat_service import ActigraphyInput, PATModelService
from clarity.ml.preprocessing import ActigraphyDataPoint
from clarity.models.user import User

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/chat")


def get_gemini_service() -> GeminiService:
    return get_container().get_instance(GeminiService)


def get_pat_model_service() -> PATModelService:
    return get_container().get_instance(PATModelService)


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
        websocket: WebSocket,
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
        except Exception as e:
            logger.exception("Error generating Gemini response: %s", e)
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

    async def process_typing_message(
        self,
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

    async def process_heartbeat(
        self,
        websocket: WebSocket,
        message: dict[str, Any],
        connection_manager: ConnectionManager,
    ) -> None:
        user_id = message.get("user_id", "unknown")
        logger.info("Processing heartbeat for user %s", user_id)
        # The ConnectionManager handles last active time internally, so this explicit call is removed.
        # connection_manager.update_last_active(websocket)
        # Acknowledge heartbeat
        heartbeat_ack_message = HeartbeatMessage(
            timestamp=datetime.now(UTC),
            type=MessageType.HEARTBEAT_ACK,
        )
        await connection_manager.send_to_connection(websocket, heartbeat_ack_message)

    async def trigger_health_analysis(
        self,
        user_id: str,
        health_data: dict[str, Any],
        connection_manager: ConnectionManager,
    ) -> None:
        logger.info(
            "Triggering health analysis for user %s with data: %s", user_id, health_data
        )
        try:
            duration_hours = 24
            if (
                "data_points" in health_data
                and isinstance(health_data["data_points"], list)
                and len(health_data["data_points"]) > 1
            ):
                timestamps = []
                for dp in health_data["data_points"]:
                    try:
                        if isinstance(dp, dict) and "timestamp" in dp:
                            ts = dp["timestamp"]
                            if isinstance(ts, str):
                                timestamps.append(datetime.fromisoformat(ts))
                    except (ValueError, TypeError) as ts_e:
                        logger.warning(
                            "Invalid timestamp format in health data point: %s. Error: %s",
                            dp,
                            ts_e,
                        )
                        continue

                if len(timestamps) > 1:
                    min_ts = min(timestamps)
                    max_ts = max(timestamps)
                    duration_seconds = (max_ts - min_ts).total_seconds()
                    if duration_seconds > 0:
                        duration_hours = max(1, int(duration_seconds / 3600))
                elif len(timestamps) == 1:
                    duration_hours = 1

            actigraphy_input = ActigraphyInput(
                user_id=user_id,
                data_points=[
                    ActigraphyDataPoint(
                        timestamp=datetime.now(UTC),
                        value=float(health_data.get("steps", 0)),
                    )
                ],
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
        except Exception as e:
            logger.exception("Error during health analysis: %s", e)
            error_msg = ErrorMessage(
                error_code="HEALTH_ANALYSIS_ERROR",
                message="Failed to perform health analysis: %s" % e,
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
    current_user: User = Depends(get_current_user_websocket),
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
    handler = WebSocketChatHandler(
        gemini_service=gemini_service, pat_service=pat_service
    )
    user_id = current_user.uid
    username = str(
        current_user.display_name or current_user.email
    )  # Ensure username is string

    logger.info("WebSocket connection attempt: token=%s", token)

    try:
        await websocket.accept()
        await connection_manager.connect(
            websocket=websocket, user_id=user_id, username=username, room_id=room_id
        )

        logger.info(
            "WebSocket chat connection established for %s in room %s",
            username,
            room_id,
        )

        while True:
            try:
                raw_message = await websocket.receive_text()

                if not await connection_manager.handle_message(websocket, raw_message):
                    continue

                try:
                    message_data = json.loads(raw_message)
                    message_type = message_data.get("type")

                    if message_type == MessageType.MESSAGE.value:
                        chat_msg: ChatMessage = ChatMessage(**message_data)
                        await handler.process_chat_message(
                            websocket, chat_msg, connection_manager
                        )

                    elif message_type == MessageType.TYPING.value:
                        typing_msg: TypingMessage = TypingMessage(**message_data)
                        # Pass the room_id from the endpoint directly
                        await handler.process_typing_message(
                            typing_msg, connection_manager, room_id
                        )

                    elif message_type == MessageType.HEARTBEAT.value:
                        await handler.process_heartbeat(
                            websocket, message_data, connection_manager
                        )

                    elif message_type == MessageType.HEALTH_INSIGHT.value:
                        health_data_content = message_data.get("content", {})
                        user_id_from_message = message_data.get("user_id", "")
                        if user_id_from_message:
                            await handler.trigger_health_analysis(
                                user_id_from_message,
                                health_data_content,
                                connection_manager,
                            )
                        else:
                            logger.warning("User ID not found in health data message.")

                    else:
                        logger.warning("Unknown message type: %s", message_type)
                        error_msg = ErrorMessage(
                            error_code="UNKNOWN_MESSAGE_TYPE",
                            message="Unknown message type: %s" % message_type,
                        )
                        await connection_manager.send_to_connection(
                            websocket, error_msg
                        )

                except json.JSONDecodeError:
                    error_msg = ErrorMessage(
                        error_code="INVALID_JSON", message="Invalid JSON format"
                    )
                    await connection_manager.send_to_connection(websocket, error_msg)

                except Exception as e:
                    logger.exception("Error parsing message: %s", e)
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

            except Exception as e:
                logger.exception("Unexpected error in chat endpoint: %s", e)
                await connection_manager.disconnect(websocket, "internal_error")
                break

    except Exception as e:
        logger.exception("Error in WebSocket connection: %s", e)

    finally:
        await connection_manager.disconnect(websocket, "Connection closed")
        logger.info(
            "WebSocket connection closed for %s",
            username if "username" in locals() else "unknown user",
        )


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
    user: User | None = None
    handler = WebSocketChatHandler(
        gemini_service=gemini_service, pat_service=pat_service
    )

    logger.info("WebSocket connection attempt: %s", token)

    try:
        await websocket.accept()

        if token:
            try:
                auth_result = get_current_user_websocket(token)
                if asyncio.iscoroutine(auth_result) or inspect.isawaitable(auth_result):
                    user = await auth_result
                else:
                    user = auth_result
                if user and user.uid != user_id:
                    await websocket.close(code=1008, reason="User ID mismatch")
                    return
            except HTTPException:
                await websocket.close(code=1008, reason="Invalid authentication token")
                return

        username = (
            user.display_name if user and user.display_name else "User_%s" % user_id[:8]
        )

        await connection_manager.connect(
            websocket=websocket,
            user_id=user_id,
            username=username,
            room_id="health_analysis_%s" % user_id,
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

                try:
                    message_data = json.loads(raw_message)

                    if message_data.get("type") == "health_data":
                        health_data = message_data.get("data", {})
                        await handler.trigger_health_analysis(
                            user_id, health_data, connection_manager
                        )

                    elif message_data.get("type") == MessageType.HEARTBEAT.value:
                        await handler.process_heartbeat(
                            websocket, message_data, connection_manager
                        )

                except json.JSONDecodeError:
                    error_msg = ErrorMessage(
                        error_code="INVALID_JSON", message="Invalid JSON format"
                    )
                    await connection_manager.send_to_connection(websocket, error_msg)

            except WebSocketDisconnect as e:
                logger.warning(
                    "WebSocket disconnected: code=%s, reason=%s", e.code, e.reason
                )
                break

            except Exception as e:
                logger.exception("Error in health analysis WebSocket: %s", e)
                await connection_manager.disconnect(websocket, "internal_error")
                break

    except Exception as e:
        logger.exception("Error in health analysis WebSocket connection: %s", e)

    finally:
        await connection_manager.disconnect(
            websocket, "Health analysis connection closed"
        )
        logger.info(f"Health analysis WebSocket closed for {user_id}")


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
