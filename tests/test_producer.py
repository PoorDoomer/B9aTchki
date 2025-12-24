"""
Unit tests for the Kafka producer module.
Tests message publishing and event serialization.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, PropertyMock


class TestReclamationEvent:
    """Tests for ReclamationEvent dataclass."""
    
    def test_event_creation(self):
        """Test creating a reclamation event."""
        from src.producer import ReclamationEvent
        
        event = ReclamationEvent(
            reclamation_id=1,
            reclamant_id=10,
            event_type="new_reclamation"
        )
        
        assert event.reclamation_id == 1
        assert event.reclamant_id == 10
        assert event.event_type == "new_reclamation"
    
    def test_event_default_type(self):
        """Test default event type."""
        from src.producer import ReclamationEvent
        
        event = ReclamationEvent(reclamation_id=1, reclamant_id=10)
        assert event.event_type == "new_reclamation"
    
    def test_event_to_json(self):
        """Test serializing event to JSON."""
        from src.producer import ReclamationEvent
        
        event = ReclamationEvent(reclamation_id=1, reclamant_id=10)
        json_str = event.to_json()
        
        data = json.loads(json_str)
        assert data["reclamation_id"] == 1
        assert data["reclamant_id"] == 10
        assert data["event_type"] == "new_reclamation"
    
    def test_event_from_json(self):
        """Test deserializing event from JSON."""
        from src.producer import ReclamationEvent
        
        json_str = '{"reclamation_id": 1, "reclamant_id": 10, "event_type": "new_reclamation"}'
        event = ReclamationEvent.from_json(json_str)
        
        assert event.reclamation_id == 1
        assert event.reclamant_id == 10
        assert event.event_type == "new_reclamation"
    
    def test_event_roundtrip(self):
        """Test event serialization roundtrip."""
        from src.producer import ReclamationEvent
        
        original = ReclamationEvent(reclamation_id=42, reclamant_id=100)
        json_str = original.to_json()
        restored = ReclamationEvent.from_json(json_str)
        
        assert original.reclamation_id == restored.reclamation_id
        assert original.reclamant_id == restored.reclamant_id
        assert original.event_type == restored.event_type


class TestKafkaEventProducer:
    """Tests for KafkaEventProducer."""
    
    @pytest.fixture
    def mock_kafka(self):
        """Mock kafka library."""
        with patch('src.producer.KafkaClient') as mock:
            mock_producer = MagicMock()
            mock_future = MagicMock()
            mock_future.get.return_value = None
            mock_producer.send.return_value = mock_future
            mock.return_value = mock_producer
            
            yield mock, mock_producer
    
    def test_producer_connect(self, mock_kafka):
        """Test producer connection."""
        from src.producer import KafkaEventProducer
        
        mock_class, mock_producer = mock_kafka
        
        producer = KafkaEventProducer()
        producer.connect()
        
        mock_class.assert_called_once()
        assert producer._producer is not None
    
    def test_producer_disconnect(self, mock_kafka):
        """Test producer disconnection."""
        from src.producer import KafkaEventProducer
        
        mock_class, mock_producer = mock_kafka
        
        producer = KafkaEventProducer()
        producer.connect()
        producer.disconnect()
        
        mock_producer.flush.assert_called_once()
        mock_producer.close.assert_called_once()
    
    def test_producer_publish(self, mock_kafka):
        """Test publishing an event."""
        from src.producer import KafkaEventProducer, ReclamationEvent
        
        mock_class, mock_producer = mock_kafka
        
        producer = KafkaEventProducer()
        producer.connect()
        
        event = ReclamationEvent(reclamation_id=1, reclamant_id=10)
        result = producer.publish(event)
        
        assert result is True
        mock_producer.send.assert_called_once()
    
    def test_producer_publish_new_reclamation(self, mock_kafka):
        """Test convenience method for publishing."""
        from src.producer import KafkaEventProducer
        
        mock_class, mock_producer = mock_kafka
        
        producer = KafkaEventProducer()
        producer.connect()
        
        result = producer.publish_new_reclamation(1, 10)
        
        assert result is True
        mock_producer.send.assert_called_once()
    
    def test_producer_publish_to_dlq(self, mock_kafka):
        """Test publishing to dead letter topic."""
        from src.producer import KafkaEventProducer, ReclamationEvent
        
        mock_class, mock_producer = mock_kafka
        
        producer = KafkaEventProducer()
        producer.connect()
        
        event = ReclamationEvent(reclamation_id=1, reclamant_id=10)
        result = producer.publish_to_dlq(event, "Test error")
        
        assert result is True
        # Should have published to DLQ topic
        mock_producer.send.assert_called()
    
    def test_producer_context_manager(self, mock_kafka):
        """Test using producer as context manager."""
        from src.producer import KafkaEventProducer
        
        mock_class, mock_producer = mock_kafka
        
        with KafkaEventProducer() as producer:
            result = producer.publish_new_reclamation(1, 10)
            assert result is True
        
        mock_producer.close.assert_called()
    
    def test_producer_auto_connect(self, mock_kafka):
        """Test auto-connection when publishing."""
        from src.producer import KafkaEventProducer
        
        mock_class, mock_producer = mock_kafka
        
        producer = KafkaEventProducer()
        # Not explicitly connected, should auto-connect
        producer.publish_new_reclamation(1, 10)
        
        mock_class.assert_called_once()


class TestProducerFactoryFunctions:
    """Tests for producer factory functions."""
    
    def test_get_producer(self):
        """Test get_producer factory function."""
        from src.producer import get_producer, KafkaEventProducer
        
        producer = get_producer()
        assert isinstance(producer, KafkaEventProducer)
    
    def test_publish_reclamation_event(self):
        """Test publish_reclamation_event convenience function."""
        from src.producer import publish_reclamation_event
        
        with patch('src.producer.KafkaEventProducer') as MockProducer:
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
        from src.producer import KafkaEventProducer, ReclamationEvent
        from kafka.errors import KafkaError
        
        with patch('src.producer.KafkaClient') as mock_kafka:
            mock_producer = MagicMock()
            mock_producer.send.side_effect = KafkaError("Connection lost")
            mock_kafka.return_value = mock_producer
            
            producer = KafkaEventProducer()
            producer.connect()
            
            event = ReclamationEvent(reclamation_id=1, reclamant_id=10)
            result = producer.publish(event)
            
            assert result is False
