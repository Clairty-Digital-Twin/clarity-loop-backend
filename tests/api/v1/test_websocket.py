"""Tests for WebSocket chat functionality."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import WebSocket

from clarity.api.v1.websocket.connection_manager import ConnectionManager
from clarity.api.v1.websocket.models import (
    ChatMessage,
    ErrorMessage,
    HeartbeatMessage,
    MessageType,
    SystemMessage,
    TypingMessage,
)
from clarity.main import create_app


@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def connection_manager():
    """Create connection manager for testing."""
    return ConnectionManager(
        heartbeat_interval=5,
        max_connections_per_user=2,
        connection_timeout=30,
        message_rate_limit=10,
        max_message_size=1024
    )


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.client_state = "CONNECTED"
        self.messages_sent = []
        self.closed = False
        self.close_code = None
        self.close_reason = None
    
    async def accept(self):
        """Mock accept method."""
        pass
    
    async def send_text(self, data: str):
        """Mock send_text method."""
        self.messages_sent.append(data)
    
    async def close(self, code: int = 1000, reason: str = ""):
        """Mock close method."""
        self.closed = True
        self.close_code = code
        self.close_reason = reason


class TestConnectionManager:
    """Test cases for WebSocket connection manager."""
    
    @pytest.mark.asyncio
    async def test_connect_success(self, connection_manager):
        """Test successful WebSocket connection."""
        websocket = MockWebSocket()
        
        result = await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User",
            room_id="test_room"
        )
        
        assert result is True
        assert "test_user" in connection_manager.user_connections
        assert websocket in connection_manager.connection_info
        assert len(websocket.messages_sent) == 1  # Connection ack message
        
        # Verify connection info
        info = connection_manager.connection_info[websocket]
        assert info.user_id == "test_user"
        assert info.username == "Test User"
    
    @pytest.mark.asyncio
    async def test_connect_max_connections_exceeded(self, connection_manager):
        """Test connection rejection when max connections exceeded."""
        user_id = "test_user"
        
        # Connect up to max connections
        for i in range(connection_manager.max_connections_per_user):
            websocket = MockWebSocket()
            result = await connection_manager.connect(
                websocket=websocket,
                user_id=user_id,
                username=f"User {i}",
            )
            assert result is True
        
        # Try to connect one more (should fail)
        websocket = MockWebSocket()
        result = await connection_manager.connect(
            websocket=websocket,
            user_id=user_id,
            username="Extra User",
        )
        
        assert result is False
        assert websocket.closed is True
        assert websocket.close_code == 1008
    
    @pytest.mark.asyncio
    async def test_disconnect(self, connection_manager):
        """Test WebSocket disconnection."""
        websocket = MockWebSocket()
        
        # Connect first
        await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User",
        )
        
        # Disconnect
        await connection_manager.disconnect(websocket)
        
        assert websocket not in connection_manager.connection_info
        assert len(connection_manager.user_connections.get("test_user", [])) == 0
    
    @pytest.mark.asyncio
    async def test_send_to_connection(self, connection_manager):
        """Test sending message to specific connection."""
        websocket = MockWebSocket()
        
        await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User",
        )
        
        # Clear initial messages
        websocket.messages_sent.clear()
        
        # Send message
        message = SystemMessage(content="Test message")
        await connection_manager.send_to_connection(websocket, message)
        
        assert len(websocket.messages_sent) == 1
        sent_data = json.loads(websocket.messages_sent[0])
        assert sent_data["type"] == MessageType.SYSTEM
        assert sent_data["content"] == "Test message"
    
    @pytest.mark.asyncio
    async def test_broadcast_to_room(self, connection_manager):
        """Test broadcasting message to room."""
        # Connect multiple users to same room
        websockets = []
        for i in range(3):
            websocket = MockWebSocket()
            await connection_manager.connect(
                websocket=websocket,
                user_id=f"user_{i}",
                username=f"User {i}",
                room_id="test_room"
            )
            websocket.messages_sent.clear()  # Clear connection ack
            websockets.append(websocket)
        
        # Broadcast message
        message = SystemMessage(content="Broadcast test")
        await connection_manager.broadcast_to_room("test_room", message)
        
        # Verify all users received message
        for websocket in websockets:
            assert len(websocket.messages_sent) == 1
            sent_data = json.loads(websocket.messages_sent[0])
            assert sent_data["content"] == "Broadcast test"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, connection_manager):
        """Test message rate limiting."""
        websocket = MockWebSocket()
        
        await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User",
        )
        
        # Send messages up to limit
        for i in range(connection_manager.message_rate_limit):
            result = await connection_manager.handle_message(
                websocket, f"Message {i}"
            )
            assert result is True
        
        # Next message should be rate limited
        result = await connection_manager.handle_message(
            websocket, "Rate limited message"
        )
        assert result is False
        
        # Should have received error message
        error_messages = [
            msg for msg in websocket.messages_sent 
            if "RATE_LIMIT_EXCEEDED" in msg
        ]
        assert len(error_messages) > 0
    
    @pytest.mark.asyncio
    async def test_message_size_limit(self, connection_manager):
        """Test message size limiting."""
        websocket = MockWebSocket()
        
        await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User",
        )
        
        # Create message larger than limit
        large_message = "x" * (connection_manager.max_message_size + 1)
        
        result = await connection_manager.handle_message(websocket, large_message)
        assert result is False
        
        # Should have received error message
        error_messages = [
            msg for msg in websocket.messages_sent 
            if "MESSAGE_TOO_LARGE" in msg
        ]
        assert len(error_messages) > 0
    
    @pytest.mark.asyncio
    async def test_heartbeat_functionality(self, connection_manager):
        """Test heartbeat monitoring."""
        websocket = MockWebSocket()
        
        await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User",
        )
        
        initial_message_count = len(websocket.messages_sent)
        
        # Trigger heartbeat
        await connection_manager._send_heartbeats()
        
        # Should receive heartbeat message
        assert len(websocket.messages_sent) > initial_message_count
        
        # Check last message is heartbeat
        last_message = json.loads(websocket.messages_sent[-1])
        assert last_message["type"] == MessageType.HEARTBEAT


class TestWebSocketModels:
    """Test cases for WebSocket message models."""
    
    def test_chat_message_validation(self):
        """Test ChatMessage validation."""
        # Valid message
        message = ChatMessage(
            content="Hello world",
            user_id="test_user",
            username="Test User"
        )
        assert message.type == MessageType.MESSAGE
        assert message.content == "Hello world"
        
        # Invalid message (too long)
        with pytest.raises(ValueError):
            ChatMessage(
                content="x" * 3000,  # Exceeds max_length
                user_id="test_user"
            )
        
        # Invalid message (empty content)
        with pytest.raises(ValueError):
            ChatMessage(
                content="",
                user_id="test_user"
            )
    
    def test_error_message_creation(self):
        """Test ErrorMessage creation."""
        error = ErrorMessage(
            error_code="TEST_ERROR",
            message="Test error message",
            details={"key": "value"}
        )
        
        assert error.type == MessageType.ERROR
        assert error.error_code == "TEST_ERROR"
        assert error.message == "Test error message"
        assert error.details == {"key": "value"}
    
    def test_typing_message(self):
        """Test TypingMessage model."""
        typing = TypingMessage(
            user_id="test_user",
            username="Test User",
            is_typing=True
        )
        
        assert typing.type == MessageType.TYPING
        assert typing.is_typing is True
    
    def test_heartbeat_message(self):
        """Test HeartbeatMessage model."""
        heartbeat = HeartbeatMessage()
        assert heartbeat.type == MessageType.HEARTBEAT
        
        # Test JSON serialization
        json_data = heartbeat.model_dump_json()
        assert isinstance(json_data, str)
        
        # Test deserialization
        parsed = json.loads(json_data)
        assert parsed["type"] == MessageType.HEARTBEAT


@pytest.mark.asyncio
class TestWebSocketEndpoints:
    """Test WebSocket endpoints."""
    
    @patch('clarity.auth.firebase_auth.get_current_user_websocket')
    async def test_websocket_chat_endpoint_authenticated(self, mock_auth):
        """Test WebSocket chat endpoint with authentication."""
        from clarity.models.user import User
        
        # Mock authenticated user
        mock_user = User(
            uid="test_user_123",
            email="test@example.com",
            display_name="Test User"
        )
        mock_auth.return_value = mock_user
        
        app = create_app()
        
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/chat/test_room?token=mock_token") as websocket:
                # Should receive connection acknowledgment
                data = websocket.receive_json()
                assert data["type"] == MessageType.CONNECTION_ACK
                assert data["user_id"] == "test_user_123"
                
                # Send a chat message
                chat_message = {
                    "type": MessageType.MESSAGE,
                    "content": "Hello from test!"
                }
                websocket.send_json(chat_message)
                
                # Should receive the message back (broadcast)
                data = websocket.receive_json()
                assert data["type"] == MessageType.MESSAGE
                assert data["content"] == "Hello from test!"
                assert data["user_id"] == "test_user_123"
    
    async def test_websocket_chat_endpoint_anonymous(self):
        """Test WebSocket chat endpoint without authentication."""
        app = create_app()
        
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/chat/test_room") as websocket:
                # Should receive connection acknowledgment
                data = websocket.receive_json()
                assert data["type"] == MessageType.CONNECTION_ACK
                assert "anonymous_" in data["user_id"]
    
    async def test_websocket_invalid_message_format(self):
        """Test WebSocket with invalid message format."""
        app = create_app()
        
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/chat/test_room") as websocket:
                # Skip connection ack
                websocket.receive_json()
                
                # Send invalid JSON
                websocket.send_text("invalid json")
                
                # Should receive error message
                data = websocket.receive_json()
                assert data["type"] == MessageType.ERROR
                assert data["error_code"] == "INVALID_JSON"
    
    async def test_websocket_typing_indicator(self):
        """Test typing indicator functionality."""
        app = create_app()
        
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/chat/test_room") as websocket:
                # Skip connection ack
                websocket.receive_json()
                
                # Send typing message
                typing_message = {
                    "type": MessageType.TYPING,
                    "is_typing": True
                }
                websocket.send_json(typing_message)
                
                # For single user, no typing broadcast back to sender
                # Would need multiple connections to test properly


@pytest.mark.asyncio
class TestChatHandler:
    """Test chat handler functionality."""
    
    @patch('clarity.services.gemini_service.GeminiService')
    async def test_health_insight_generation(self, mock_gemini_service):
        """Test health insight generation from chat messages."""
        from clarity.api.v1.websocket.chat_handler import WebSocketChatHandler
        
        # Mock Gemini service response
        mock_gemini_service.return_value.generate_health_insights.return_value = {
            "insights": [{
                "content": "Based on your message about feeling tired, consider improving your sleep schedule.",
                "confidence": 0.85,
                "category": "sleep"
            }],
            "recommendations": ["Maintain consistent sleep schedule", "Avoid caffeine before bed"]
        }
        
        handler = WebSocketChatHandler()
        websocket = MockWebSocket()
        
        # Create connection manager and connect
        connection_manager = ConnectionManager()
        await connection_manager.connect(
            websocket=websocket,
            user_id="test_user",
            username="Test User"
        )
        
        # Create health-related message
        message = ChatMessage(
            content="I'm feeling really tired lately and can't sleep well",
            user_id="test_user",
            username="Test User"
        )
        
        # Process message
        await handler.process_chat_message(websocket, message)
        
        # Should have generated health insight
        mock_gemini_service.return_value.generate_health_insights.assert_called_once()
        
        # Check if insight message was sent
        insight_messages = [
            msg for msg in websocket.messages_sent
            if "sleep schedule" in msg
        ]
        assert len(insight_messages) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])