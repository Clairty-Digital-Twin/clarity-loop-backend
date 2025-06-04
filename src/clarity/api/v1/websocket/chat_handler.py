"""WebSocket chat handler for real-time health insights and communication."""

from datetime import datetime
import json
import logging
from typing import Optional
import uuid

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from starlette.websockets import WebSocketState

from clarity.auth.firebase_auth import get_current_user_websocket
from clarity.core.config import get_settings
from clarity.ml.gemini_service import GeminiService
from clarity.ml.pat_service import PATModelService
from clarity.models.user import User

from .lifespan import get_connection_manager
from .models import (
    AnalysisUpdateMessage,
    ChatMessage,
    ErrorMessage,
    HealthInsightMessage,
    HeartbeatAckMessage,
    MessageType,
    SystemMessage,
    TypingMessage,
    WebSocketMessage,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/ws", tags=["websocket"])


class WebSocketChatHandler:
    """Handles WebSocket chat functionality with health insights integration."""

    def __init__(self):
        self.gemini_service = GeminiService()
        self.pat_service = PATModelService()

    async def process_chat_message(
        self, websocket: WebSocket, message: ChatMessage, connection_manager
    ):
        """Process a chat message and potentially generate health insights."""
        try:
            connection_info = connection_manager.connection_info.get(websocket)
            if not connection_info:
                return

            # Update message with user info
            message.user_id = connection_info.user_id
            message.username = connection_info.username
            message.message_id = str(uuid.uuid4())

            # Broadcast message to room
            await connection_manager.broadcast_to_room("general", message)

            # Check if message contains health-related content for AI analysis
            await self._analyze_for_health_insights(message)

        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            error_msg = ErrorMessage(
                error_code="PROCESSING_ERROR", message="Failed to process message"
            )
            await connection_manager.send_to_connection(websocket, error_msg)

    async def _analyze_for_health_insights(self, message: ChatMessage):
        """Analyze message content for potential health insights."""
        # Keywords that might trigger health analysis
        health_keywords = [
            "sleep",
            "tired",
            "energy",
            "stress",
            "anxiety",
            "mood",
            "exercise",
            "workout",
            "heart rate",
            "steps",
            "activity",
            "pain",
            "headache",
            "feel",
            "feeling",
            "health",
            "wellness",
        ]

        message_lower = message.content.lower()
        if any(keyword in message_lower for keyword in health_keywords):
            try:
                # Generate health insight using Gemini
                insight_response = await self.gemini_service.generate_health_insights(
                    {
                        "user_query": message.content,
                        "context": "chat_message",
                        "timestamp": message.timestamp.isoformat(),
                    }
                )

                if insight_response.get("insights"):
                    insight_message = HealthInsightMessage(
                        user_id=message.user_id,
                        insight=insight_response["insights"][0]["content"],
                        confidence=insight_response["insights"][0].get(
                            "confidence", 0.8
                        ),
                        category="conversational_health",
                        recommendations=insight_response.get("recommendations", []),
                    )

                    # Send insight back to user
                    await connection_manager.send_to_user(
                        message.user_id, insight_message
                    )

            except Exception as e:
                logger.error(f"Error generating health insight: {e}")

    async def process_typing_message(
        self, websocket: WebSocket, message: TypingMessage, connection_manager
    ):
        """Process typing indicator message."""
        connection_info = connection_manager.connection_info.get(websocket)
        if not connection_info:
            return

        # Update message with user info
        message.user_id = connection_info.user_id
        message.username = connection_info.username

        # Broadcast typing status to room (excluding sender)
        await connection_manager.broadcast_to_room(
            "general", message, exclude_user=connection_info.user_id
        )

    async def process_heartbeat(self, websocket: WebSocket, message):
        """Process heartbeat message and send acknowledgment."""
        try:
            ack_message = HeartbeatAckMessage(
                client_timestamp=message.get("client_timestamp")
            )
            await connection_manager.send_to_connection(websocket, ack_message)
        except Exception as e:
            logger.error(f"Error processing heartbeat: {e}")

    async def trigger_health_analysis(
        self, user_id: str, health_data: dict, connection_manager
    ):
        """Trigger comprehensive health analysis and send real-time updates."""
        try:
            # Send analysis started message
            update_message = AnalysisUpdateMessage(
                user_id=user_id,
                status="started",
                progress=0,
                details="Starting health data analysis...",
            )
            await connection_manager.send_to_user(user_id, update_message)

            # Process with PAT service
            update_message.status = "processing"
            update_message.progress = 25
            update_message.details = "Analyzing activity patterns..."
            await connection_manager.send_to_user(user_id, update_message)

            pat_results = await self.pat_service.analyze_health_data(health_data)

            # Generate insights with Gemini
            update_message.progress = 75
            update_message.details = "Generating AI insights..."
            await connection_manager.send_to_user(user_id, update_message)

            gemini_insights = await self.gemini_service.generate_health_insights(
                {
                    "pat_analysis": pat_results,
                    "raw_data": health_data,
                    "user_id": user_id,
                }
            )

            # Send completion message
            update_message.status = "completed"
            update_message.progress = 100
            update_message.details = "Analysis complete!"
            await connection_manager.send_to_user(user_id, update_message)

            # Send detailed insights
            if gemini_insights.get("insights"):
                for insight_data in gemini_insights["insights"]:
                    insight_message = HealthInsightMessage(
                        user_id=user_id,
                        insight=insight_data["content"],
                        confidence=insight_data.get("confidence", 0.9),
                        category=insight_data.get("category", "health_analysis"),
                        source_data={
                            "pat_analysis": pat_results,
                            "analysis_timestamp": datetime.utcnow().isoformat(),
                        },
                        recommendations=gemini_insights.get("recommendations", []),
                    )
                    await connection_manager.send_to_user(user_id, insight_message)

        except Exception as e:
            logger.error(f"Error in health analysis: {e}")

            # Send error message
            error_update = AnalysisUpdateMessage(
                user_id=user_id,
                status="failed",
                progress=100,
                details=f"Analysis failed: {e!s}",
            )
            await connection_manager.send_to_user(user_id, error_update)


# Global chat handler instance
chat_handler = WebSocketChatHandler()


@router.websocket("/chat/{room_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    room_id: str = "general",
    token: str | None = Query(None),
    connection_manager=Depends(get_connection_manager),
):
    """WebSocket endpoint for real-time chat with health insights.

    Features:
    - Real-time messaging
    - Health insights generation
    - Typing indicators
    - Connection management
    - Rate limiting
    - Heartbeat monitoring
    """
    user: User | None = None

    try:
        # Authenticate user
        if token:
            try:
                user = await get_current_user_websocket(token)
            except HTTPException:
                await websocket.close(code=1008, reason="Invalid authentication token")
                return

        # Use authenticated user info or fallback to anonymous
        user_id = user.uid if user else f"anonymous_{uuid.uuid4().hex[:8]}"
        username = (
            user.display_name if user and user.display_name else f"User_{user_id[:8]}"
        )

        # Connect to chat
        connected = await connection_manager.connect(
            websocket=websocket, user_id=user_id, username=username, room_id=room_id
        )

        if not connected:
            return

        logger.info(
            f"WebSocket chat connection established for {username} in room {room_id}"
        )

        # Main message loop
        while True:
            try:
                # Receive message
                raw_message = await websocket.receive_text()

                # Validate and rate limit
                if not await connection_manager.handle_message(websocket, raw_message):
                    continue

                # Parse message
                try:
                    message_data = json.loads(raw_message)
                    message_type = message_data.get("type")

                    if message_type == MessageType.MESSAGE:
                        message = ChatMessage(**message_data)
                        await chat_handler.process_chat_message(
                            websocket, message, connection_manager
                        )

                    elif message_type == MessageType.TYPING:
                        message = TypingMessage(**message_data)
                        await chat_handler.process_typing_message(
                            websocket, message, connection_manager
                        )

                    elif message_type == MessageType.HEARTBEAT:
                        await chat_handler.process_heartbeat(websocket, message_data)

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

            except WebSocketDisconnect:
                break

            except Exception as e:
                logger.error(f"Error in WebSocket message loop: {e}")
                break

    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")

    finally:
        # Ensure cleanup
        await connection_manager.disconnect(websocket, "Connection closed")
        logger.info(
            f"WebSocket connection closed for {username if 'username' in locals() else 'unknown user'}"
        )


@router.websocket("/health-analysis/{user_id}")
async def websocket_health_analysis_endpoint(
    websocket: WebSocket,
    user_id: str,
    token: str | None = Query(None),
    connection_manager=Depends(get_connection_manager),
):
    """WebSocket endpoint for real-time health analysis updates.

    This endpoint provides real-time updates during health data processing,
    including PAT analysis and AI insight generation.
    """
    user: User | None = None

    try:
        # Authenticate user
        if token:
            try:
                user = await get_current_user_websocket(token)
                if user.uid != user_id:
                    await websocket.close(code=1008, reason="User ID mismatch")
                    return
            except HTTPException:
                await websocket.close(code=1008, reason="Invalid authentication token")
                return

        username = (
            user.display_name if user and user.display_name else f"User_{user_id[:8]}"
        )

        # Connect for health analysis updates
        connected = await connection_manager.connect(
            websocket=websocket,
            user_id=user_id,
            username=username,
            room_id=f"health_analysis_{user_id}",
        )

        if not connected:
            return

        logger.info(f"Health analysis WebSocket connected for {username}")

        # Send welcome message
        welcome_msg = SystemMessage(
            content="Connected to health analysis service. Send health data to start analysis.",
            level="info",
        )
        await connection_manager.send_to_connection(websocket, welcome_msg)

        # Handle health data analysis requests
        while True:
            try:
                raw_message = await websocket.receive_text()

                if not await connection_manager.handle_message(websocket, raw_message):
                    continue

                try:
                    message_data = json.loads(raw_message)

                    if message_data.get("type") == "health_data":
                        health_data = message_data.get("data", {})
                        await chat_handler.trigger_health_analysis(
                            user_id, health_data, connection_manager
                        )

                    elif message_data.get("type") == MessageType.HEARTBEAT:
                        await chat_handler.process_heartbeat(websocket, message_data)

                except json.JSONDecodeError:
                    error_msg = ErrorMessage(
                        error_code="INVALID_JSON", message="Invalid JSON format"
                    )
                    await connection_manager.send_to_connection(websocket, error_msg)

            except WebSocketDisconnect:
                break

            except Exception as e:
                logger.error(f"Error in health analysis WebSocket: {e}")
                break

    except Exception as e:
        logger.error(f"Error in health analysis WebSocket connection: {e}")

    finally:
        await connection_manager.disconnect(
            websocket, "Health analysis connection closed"
        )
        logger.info(f"Health analysis WebSocket closed for {user_id}")


@router.get("/chat/stats")
async def get_chat_stats(connection_manager=Depends(get_connection_manager)):
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
    room_id: str, connection_manager=Depends(get_connection_manager)
):
    """Get list of users in a specific room."""
    users = connection_manager.get_room_users(room_id)
    user_info = []

    for user_id in users:
        info = connection_manager.get_user_info(user_id)
        if info:
            user_info.append(info)

    return {"room_id": room_id, "users": user_info}
