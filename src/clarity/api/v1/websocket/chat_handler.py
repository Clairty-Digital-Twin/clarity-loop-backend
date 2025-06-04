"""WebSocket chat handler for real-time health insights and communication."""

import asyncio
from datetime import UTC, datetime
import inspect
import json
import logging
from typing import Any

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
from clarity.ml.pat_service import ActigraphyAnalysis, ActigraphyInput, PATModelService
from clarity.ml.preprocessing import ActigraphyDataPoint
from clarity.models.user import User

from .lifespan import get_connection_manager
from .models import (
    ChatMessage,
    ErrorMessage,
    HeartbeatMessage,
    MessageType,
    SystemMessage,
    TypingMessage,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="")


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
    ):
        self.gemini_service = gemini_service
        self.pat_service = pat_service

    async def process_chat_message(
        self, websocket: WebSocket, chat_message: ChatMessage, connection_manager: Any
    ) -> None:
        logger.info(f"Processing chat message for user {chat_message.user_id}")
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
            logger.error(f"Error generating Gemini response: {e}")
            ai_response_content = "I am sorry, I could not generate a response at this time."

        response_message = ChatMessage(
            user_id="AI",
            timestamp=datetime.now(UTC),
            type=MessageType.MESSAGE,
            content=ai_response_content,
        )
        await connection_manager.send_to_user(chat_message.user_id, response_message)

    async def process_typing_message(
        self, websocket: WebSocket, typing_message: TypingMessage, connection_manager: Any
    ) -> None:
        logger.info(
            f"Processing typing indicator for user {typing_message.user_id}: "
            f"is_typing={typing_message.is_typing}"
        )
        # Broadcast typing status to the room where the user is connected
        # (Assuming room_id is stored in connection_manager.connection_info)
        connection_info = connection_manager.connection_info.get(websocket)
        if connection_info and connection_info.room_id:
            await connection_manager.broadcast_to_room(
                connection_info.room_id,
                typing_message,
                exclude_websocket=websocket,  # Don't send back to sender
            )
        else:
            logger.warning("Could not find connection info or room_id for typing message.")

    async def process_heartbeat(
        self, websocket: WebSocket, message: dict[str, Any], connection_manager: Any
    ) -> None:
        user_id = message.get("user_id", "unknown")
        logger.info(f"Processing heartbeat for user {user_id}")
        # Update last active time for the connection
        connection_manager.update_last_active(websocket)
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
        connection_manager: Any
    ) -> None:
        logger.info(f"Triggering health analysis for user {user_id} with data: {health_data}")
        try:
            duration_hours = 24
            if "data_points" in health_data and isinstance(health_data["data_points"], list) and len(health_data["data_points"]) > 1:
                timestamps = []
                for dp in health_data["data_points"]:
                    try:
                        # Attempt to parse timestamp, handle potential errors
                        if isinstance(dp, dict) and "timestamp" in dp:
                            ts = dp["timestamp"]
                            if isinstance(ts, str):
                                timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                    except (ValueError, TypeError) as ts_e:
                        logger.warning(f"Invalid timestamp format in health data point: {dp}. Error: {ts_e}")
                        continue  # Skip invalid timestamp

                if len(timestamps) > 1:
                    min_ts = min(timestamps)
                    max_ts = max(timestamps)
                    duration_seconds = (max_ts - min_ts).total_seconds()
                    if duration_seconds > 0:
                        duration_hours = max(1, int(duration_seconds / 3600))
                elif len(timestamps) == 1:
                    duration_hours = 1  # For single data point, assume 1 hour of activity

            actigraphy_input = ActigraphyInput(
                user_id=user_id,
                data_points=[
                    ActigraphyDataPoint(
                        timestamp=datetime.now(UTC),
                        value=float(health_data.get("steps", 0))
                    )
                ],
                sampling_rate=1.0,
                duration_hours=duration_hours
            )

            pat_analysis_results: ActigraphyAnalysis = await self.pat_service.analyze_actigraphy(
                actigraphy_input
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
            logger.info(f"Health analysis interrupted by client disconnect for user {user_id}")
            raise  # Re-raise to be caught by outer handler
        except Exception as e:
            logger.error(f"Error during health analysis: {e}")
            error_msg = ErrorMessage(
                error_code="HEALTH_ANALYSIS_ERROR",
                message=f"Failed to perform health analysis: {e}",
            )
            await connection_manager.send_to_user(user_id, error_msg)


@router.websocket("/chat/{room_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    room_id: str = "general",
    token: str | None = Query(...),
    connection_manager: Any = Depends(get_connection_manager),
    gemini_service: GeminiService = Depends(get_gemini_service),
    pat_service: PATModelService = Depends(get_pat_model_service),
    current_user: User = Depends(get_current_user_websocket)
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
    handler = WebSocketChatHandler(gemini_service=gemini_service, pat_service=pat_service)
    user_id = current_user.uid
    username = current_user.display_name or current_user.email

    logger.info(f"WebSocket connection attempt: token={token}")

    try:
        connected = await connection_manager.connect(
            websocket=websocket, user_id=user_id, username=username, room_id=room_id
        )

        if not connected:
            return

        logger.info(
            f"WebSocket chat connection established for {username} in room {room_id}"
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
                        message = ChatMessage(**message_data)
                        await handler.process_chat_message(
                            websocket, message, connection_manager
                        )

                    elif message_type == MessageType.TYPING.value:
                        message = TypingMessage(**message_data)
                        await handler.process_typing_message(
                            websocket, message, connection_manager
                        )

                    elif message_type == MessageType.HEARTBEAT.value:
                        await handler.process_heartbeat(
                            websocket, message_data, connection_manager
                        )

                    elif message_type == MessageType.HEALTH_INSIGHT.value:
                        health_data_content = message_data.get("content", {})
                        user_id_from_message = message_data.get("user_id", "")
                        if user_id_from_message:
                            await handler.trigger_health_analysis(user_id_from_message, health_data_content, connection_manager)
                        else:
                            logger.warning("User ID not found in health data message.")

                    else:
                        logger.warning(f"Unknown message type: {message_type}")
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

                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    error_msg = ErrorMessage(
                        error_code="MESSAGE_PARSE_ERROR",
                        message="Failed to parse message",
                    )
                    await connection_manager.send_to_connection(websocket, error_msg)

            except WebSocketDisconnect as e:
                logger.warning(
                    f"WebSocket disconnected: code={e.code}, reason={e.reason}"
                )
                raise

            except Exception as e:
                logger.error(f"Unexpected error in chat endpoint: {e}")
                await connection_manager.disconnect(websocket, "internal_error")
                break

    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")

    finally:
        await connection_manager.disconnect(websocket, "Connection closed")
        logger.info(
            f"WebSocket connection closed for {username if 'username' in locals() else 'unknown user'}"
        )


@router.websocket("/health-analysis/{user_id}")
async def websocket_health_analysis_endpoint(
    websocket: WebSocket,
    user_id: str,
    token: str | None = Query(...),
    connection_manager: Any = Depends(get_connection_manager),
) -> None:
    """WebSocket endpoint for real-time health analysis updates.

    This endpoint provides real-time updates during health data processing,
    including PAT analysis and AI insight generation.
    """
    user: User | None = None
    handler = WebSocketChatHandler(
        gemini_service=Depends(get_gemini_service),
        pat_service=Depends(get_pat_model_service)
    )

    logger.info(f"WebSocket connection attempt: token={token}")

    try:
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
            user.display_name if user and user.display_name else f"User_{user_id[:8]}"
        )

        connected = await connection_manager.connect(
            websocket=websocket,
            user_id=user_id,
            username=username,
            room_id=f"health_analysis_{user_id}",
        )

        if not connected:
            return

        logger.info(f"Health analysis WebSocket connected for {username}")

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
                    f"WebSocket disconnected: code={e.code}, reason={e.reason}"
                )
                break  # Exit loop on disconnect

            except Exception as e:
                logger.error(f"Error in health analysis WebSocket: {e}")
                await connection_manager.disconnect(websocket, "internal_error")
                break

    except Exception as e:
        logger.error(f"Error in health analysis WebSocket connection: {e}")

    finally:
        await connection_manager.disconnect(
            websocket, "Health analysis connection closed"
        )
        logger.info(f"Health analysis WebSocket closed for {user_id}")


@router.get("/chat/stats")
async def get_chat_stats(connection_manager: Any = Depends(get_connection_manager)) -> dict[str, Any]:
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
    room_id: str, connection_manager: Any = Depends(get_connection_manager)
) -> dict[str, Any]:
    """Get list of users in a specific room."""
    users = connection_manager.get_room_users(room_id)
    user_info = []

    for user_id in users:
        info = connection_manager.get_user_info(user_id)
        if info:
            user_info.append(info)

    return {"room_id": room_id, "users": user_info}
