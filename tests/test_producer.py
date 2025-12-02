"""
Unit tests for the RabbitMQ producer module.
Tests message publishing and event serialization.
"""

import pytest
import json
from unittest.mock import MagicMock, patch


class TestReclamationEvent:
    """Tests for ReclamationEvent dataclass."""
    
    def test_event_creation(self):
        """Test creating a reclamation event."""
        from src.producer import ReclamationEvent
        
        event = ReclamationEvent(
            reclamation_id=1,
            user_id=10,
            event_type="new_reclamation"
        )
        
        assert event.reclamation_id == 1
        assert event.user_id == 10
        assert event.event_type == "new_reclamation"
    
    def test_event_default_type(self):
        """Test default event type."""
        from src.producer import ReclamationEvent
        
        event = ReclamationEvent(reclamation_id=1, user_id=10)
        assert event.event_type == "new_reclamation"
    
    def test_event_to_json(self):
        """Test serializing event to JSON."""
        from src.producer import ReclamationEvent
        
        event = ReclamationEvent(reclamation_id=1, user_id=10)
        json_str = event.to_json()
        
        data = json.loads(json_str)
        assert data["reclamation_id"] == 1
        assert data["user_id"] == 10
        assert data["event_type"] == "new_reclamation"
    
    def test_event_from_json(self):
        """Test deserializing event from JSON."""
        from src.producer import ReclamationEvent
        
        json_str = '{"reclamation_id": 1, "user_id": 10, "event_type": "new_reclamation"}'
        event = ReclamationEvent.from_json(json_str)
        
        assert event.reclamation_id == 1
        assert event.user_id == 10
        assert event.event_type == "new_reclamation"
    
    def test_event_roundtrip(self):
        """Test event serialization roundtrip."""
        from src.producer import ReclamationEvent
        
        original = ReclamationEvent(reclamation_id=42, user_id=100)
        json_str = original.to_json()
        restored = ReclamationEvent.from_json(json_str)
        
        assert original.reclamation_id == restored.reclamation_id
        assert original.user_id == restored.user_id
        assert original.event_type == restored.event_type


class TestRabbitMQProducer:
    """Tests for RabbitMQProducer."""
    
    @pytest.fixture
    def mock_pika(self):
        """Mock pika library."""
        with patch('src.producer.pika') as mock:
            mock_conn = MagicMock()
            mock_channel = MagicMock()
            mock_conn.channel.return_value = mock_channel
            mock.BlockingConnection.return_value = mock_conn
            mock.PlainCredentials.return_value = MagicMock()
            mock.ConnectionParameters.return_value = MagicMock()
            mock.BasicProperties.return_value = MagicMock()
            
            yield mock, mock_conn, mock_channel
    
    def test_producer_connect(self, mock_pika):
        """Test producer connection."""
        from src.producer import RabbitMQProducer
        
        mock, mock_conn, mock_channel = mock_pika
        
        producer = RabbitMQProducer()
        producer.connect()
        
        mock.BlockingConnection.assert_called_once()
        mock_channel.queue_declare.assert_called()
    
    def test_producer_disconnect(self, mock_pika):
        """Test producer disconnection."""
        from src.producer import RabbitMQProducer
        
        mock, mock_conn, mock_channel = mock_pika
        mock_conn.is_open = True
        
        producer = RabbitMQProducer()
        producer.connect()
        producer.disconnect()
        
        mock_conn.close.assert_called_once()
    
    def test_producer_publish(self, mock_pika):
        """Test publishing an event."""
        from src.producer import RabbitMQProducer, ReclamationEvent
        
        mock, mock_conn, mock_channel = mock_pika
        mock_conn.is_open = True
        
        producer = RabbitMQProducer()
        producer.connect()
        
        event = ReclamationEvent(reclamation_id=1, user_id=10)
        result = producer.publish(event)
        
        assert result is True
        mock_channel.basic_publish.assert_called_once()
    
    def test_producer_publish_new_reclamation(self, mock_pika):
        """Test convenience method for publishing."""
        from src.producer import RabbitMQProducer
        
        mock, mock_conn, mock_channel = mock_pika
        mock_conn.is_open = True
        
        producer = RabbitMQProducer()
        producer.connect()
        
        result = producer.publish_new_reclamation(1, 10)
        
        assert result is True
        mock_channel.basic_publish.assert_called_once()
    
    def test_producer_publish_to_dlq(self, mock_pika):
        """Test publishing to dead letter queue."""
        from src.producer import RabbitMQProducer, ReclamationEvent
        
        mock, mock_conn, mock_channel = mock_pika
        mock_conn.is_open = True
        
        producer = RabbitMQProducer()
        producer.connect()
        
        event = ReclamationEvent(reclamation_id=1, user_id=10)
        result = producer.publish_to_dlq(event, "Test error")
        
        assert result is True
        # Should have published to DLQ
        mock_channel.basic_publish.assert_called()
    
    def test_producer_context_manager(self, mock_pika):
        """Test using producer as context manager."""
        from src.producer import RabbitMQProducer
        
        mock, mock_conn, mock_channel = mock_pika
        mock_conn.is_open = True
        
        with RabbitMQProducer() as producer:
            result = producer.publish_new_reclamation(1, 10)
            assert result is True
        
        mock_conn.close.assert_called()
    
    def test_producer_reconnect_on_failure(self, mock_pika):
        """Test auto-reconnection on connection failure."""
        from src.producer import RabbitMQProducer
        
        mock, mock_conn, mock_channel = mock_pika
        
        # First check returns False (disconnected), second returns True
        mock_conn.is_open = False
        
        producer = RabbitMQProducer()
        producer._connection = mock_conn
        producer._channel = mock_channel
        
        # This should trigger reconnection
        producer._ensure_connected()
        
        # Should have reconnected
        assert mock.BlockingConnection.call_count >= 1


class TestProducerFactoryFunctions:
    """Tests for producer factory functions."""
    
    def test_get_producer(self):
        """Test get_producer factory function."""
        from src.producer import get_producer, RabbitMQProducer
        
        producer = get_producer()
        assert isinstance(producer, RabbitMQProducer)
    
    def test_publish_reclamation_event(self):
        """Test publish_reclamation_event convenience function."""
        from src.producer import publish_reclamation_event
        
        with patch('src.producer.RabbitMQProducer') as MockProducer:
            mock_instance = MagicMock()
            mock_instance.publish_new_reclamation.return_value = True
            MockProducer.return_value.__enter__ = MagicMock(return_value=mock_instance)
            MockProducer.return_value.__exit__ = MagicMock(return_value=False)
            
            result = publish_reclamation_event(1, 10)
            
            assert result is True


class TestProducerErrorHandling:
    """Tests for producer error handling."""
    
    def test_publish_connection_error(self):
        """Test handling connection errors during publish."""
        from src.producer import RabbitMQProducer, ReclamationEvent
        from pika.exceptions import AMQPConnectionError
        
        with patch('src.producer.pika') as mock_pika:
            mock_conn = MagicMock()
            mock_channel = MagicMock()
            mock_channel.basic_publish.side_effect = AMQPConnectionError("Connection lost")
            mock_conn.channel.return_value = mock_channel
            mock_conn.is_open = True
            mock_pika.BlockingConnection.return_value = mock_conn
            mock_pika.PlainCredentials.return_value = MagicMock()
            mock_pika.ConnectionParameters.return_value = MagicMock()
            
            producer = RabbitMQProducer()
            producer.connect()
            
            event = ReclamationEvent(reclamation_id=1, user_id=10)
            result = producer.publish(event)
            
            assert result is False


