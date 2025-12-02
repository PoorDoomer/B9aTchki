"""
Unit tests for the RabbitMQ worker module.
Tests message consumption and processing.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, call


class TestRabbitMQWorker:
    """Tests for RabbitMQWorker."""
    
    @pytest.fixture
    def mock_pika(self):
        """Mock pika library."""
        with patch('src.worker.pika') as mock:
            mock_conn = MagicMock()
            mock_channel = MagicMock()
            mock_conn.channel.return_value = mock_channel
            mock_conn.is_open = True
            mock.BlockingConnection.return_value = mock_conn
            mock.PlainCredentials.return_value = MagicMock()
            mock.ConnectionParameters.return_value = MagicMock()
            
            yield mock, mock_conn, mock_channel
    
    @pytest.fixture
    def mock_detector(self):
        """Mock duplicate detector."""
        from src.duplicate_detector import DuplicateDetectionResult, DuplicateAction
        
        mock = MagicMock()
        mock.process_reclamation.return_value = DuplicateDetectionResult(
            reclamation_id=1,
            is_duplicate=False,
            action=DuplicateAction.NO_ACTION,
            message="No duplicates found"
        )
        return mock
    
    def test_worker_connect(self, mock_pika):
        """Test worker connection."""
        from src.worker import RabbitMQWorker
        
        mock, mock_conn, mock_channel = mock_pika
        
        with patch('src.worker.RabbitMQProducer'):
            worker = RabbitMQWorker()
            worker.connect()
        
        mock.BlockingConnection.assert_called_once()
        mock_channel.basic_qos.assert_called_once_with(prefetch_count=1)
        mock_channel.queue_declare.assert_called()
    
    def test_worker_disconnect(self, mock_pika):
        """Test worker disconnection."""
        from src.worker import RabbitMQWorker
        
        mock, mock_conn, mock_channel = mock_pika
        
        with patch('src.worker.RabbitMQProducer') as MockProducer:
            mock_producer = MagicMock()
            MockProducer.return_value = mock_producer
            
            worker = RabbitMQWorker()
            worker.connect()
            worker.disconnect()
        
        mock_conn.close.assert_called_once()
        mock_producer.disconnect.assert_called_once()
    
    def test_worker_process_message_success(self, mock_pika, mock_detector):
        """Test successful message processing."""
        from src.worker import RabbitMQWorker
        from src.producer import ReclamationEvent
        
        mock, mock_conn, mock_channel = mock_pika
        
        # Create a test message
        event = ReclamationEvent(reclamation_id=1, user_id=10)
        body = event.to_json().encode('utf-8')
        
        method = MagicMock()
        method.delivery_tag = 1
        properties = MagicMock()
        
        with patch('src.worker.RabbitMQProducer'):
            worker = RabbitMQWorker(detector=mock_detector)
            worker.connect()
            
            # Process the message
            worker._process_message(mock_channel, method, properties, body)
        
        # Verify detector was called
        mock_detector.process_reclamation.assert_called_once_with(1)
        
        # Verify message was acknowledged
        mock_channel.basic_ack.assert_called_once_with(delivery_tag=1)
    
    def test_worker_process_message_with_callback(self, mock_pika, mock_detector):
        """Test message processing with result callback."""
        from src.worker import RabbitMQWorker
        from src.producer import ReclamationEvent
        from src.duplicate_detector import DuplicateDetectionResult
        
        mock, mock_conn, mock_channel = mock_pika
        
        # Create callback to capture result
        results = []
        def on_result(result: DuplicateDetectionResult):
            results.append(result)
        
        event = ReclamationEvent(reclamation_id=1, user_id=10)
        body = event.to_json().encode('utf-8')
        
        method = MagicMock()
        method.delivery_tag = 1
        properties = MagicMock()
        
        with patch('src.worker.RabbitMQProducer'):
            worker = RabbitMQWorker(detector=mock_detector, on_result=on_result)
            worker.connect()
            worker._process_message(mock_channel, method, properties, body)
        
        # Verify callback was called
        assert len(results) == 1
        assert results[0].reclamation_id == 1
    
    def test_worker_process_invalid_json(self, mock_pika):
        """Test handling of invalid JSON message."""
        from src.worker import RabbitMQWorker
        
        mock, mock_conn, mock_channel = mock_pika
        
        # Invalid JSON
        body = b'not valid json'
        
        method = MagicMock()
        method.delivery_tag = 1
        properties = MagicMock()
        
        with patch('src.worker.RabbitMQProducer') as MockProducer:
            mock_producer = MagicMock()
            MockProducer.return_value = mock_producer
            
            worker = RabbitMQWorker()
            worker.connect()
            worker._process_message(mock_channel, method, properties, body)
        
        # Message should still be acknowledged (removed from queue)
        mock_channel.basic_ack.assert_called_once_with(delivery_tag=1)
    
    def test_worker_process_error_sends_to_dlq(self, mock_pika):
        """Test that processing errors send to DLQ."""
        from src.worker import RabbitMQWorker
        from src.producer import ReclamationEvent
        
        mock, mock_conn, mock_channel = mock_pika
        
        # Create detector that raises error
        mock_detector = MagicMock()
        mock_detector.process_reclamation.side_effect = Exception("Processing error")
        
        event = ReclamationEvent(reclamation_id=1, user_id=10)
        body = event.to_json().encode('utf-8')
        
        method = MagicMock()
        method.delivery_tag = 1
        properties = MagicMock()
        
        with patch('src.worker.RabbitMQProducer') as MockProducer:
            mock_producer = MagicMock()
            MockProducer.return_value = mock_producer
            
            worker = RabbitMQWorker(detector=mock_detector)
            worker.connect()
            worker._producer = mock_producer
            worker._process_message(mock_channel, method, properties, body)
        
        # Should have sent to DLQ
        mock_producer.publish_to_dlq.assert_called_once()
        
        # Message should be acknowledged
        mock_channel.basic_ack.assert_called_once_with(delivery_tag=1)
    
    def test_worker_context_manager(self, mock_pika):
        """Test using worker as context manager."""
        from src.worker import RabbitMQWorker
        
        mock, mock_conn, mock_channel = mock_pika
        
        with patch('src.worker.RabbitMQProducer'):
            with RabbitMQWorker() as worker:
                assert worker._connection is not None
        
        mock_conn.close.assert_called()
    
    def test_worker_get_queue_size(self, mock_pika):
        """Test getting queue size."""
        from src.worker import RabbitMQWorker
        
        mock, mock_conn, mock_channel = mock_pika
        
        # Mock queue_declare to return message count
        mock_result = MagicMock()
        mock_result.method.message_count = 5
        mock_channel.queue_declare.return_value = mock_result
        
        with patch('src.worker.RabbitMQProducer'):
            worker = RabbitMQWorker()
            worker.connect()
            size = worker.get_queue_size()
        
        assert size == 5
    
    def test_worker_process_one(self, mock_pika, mock_detector):
        """Test processing single message."""
        from src.worker import RabbitMQWorker
        from src.producer import ReclamationEvent
        
        mock, mock_conn, mock_channel = mock_pika
        
        event = ReclamationEvent(reclamation_id=1, user_id=10)
        body = event.to_json().encode('utf-8')
        
        method = MagicMock()
        method.delivery_tag = 1
        properties = MagicMock()
        
        # Mock basic_get to return a message
        mock_channel.basic_get.return_value = (method, properties, body)
        
        with patch('src.worker.RabbitMQProducer'):
            worker = RabbitMQWorker(detector=mock_detector)
            result = worker.process_one()
        
        assert result is not None
        assert result.reclamation_id == 1
    
    def test_worker_process_one_no_message(self, mock_pika):
        """Test process_one with no messages in queue."""
        from src.worker import RabbitMQWorker
        
        mock, mock_conn, mock_channel = mock_pika
        
        # Mock basic_get to return no message
        mock_channel.basic_get.return_value = (None, None, None)
        
        with patch('src.worker.RabbitMQProducer'):
            worker = RabbitMQWorker()
            result = worker.process_one()
        
        assert result is None


class TestWorkerSignalHandling:
    """Tests for worker signal handling."""
    
    def test_signal_handler_sets_stop_flag(self):
        """Test that signal handler sets stop flag."""
        from src.worker import RabbitMQWorker
        
        with patch('src.worker.pika'):
            with patch('src.worker.RabbitMQProducer'):
                worker = RabbitMQWorker()
                
                # Simulate signal
                worker._signal_handler(2, None)
                
                assert worker._should_stop is True


class TestWorkerFactoryFunctions:
    """Tests for worker factory functions."""
    
    def test_get_worker(self):
        """Test get_worker factory function."""
        from src.worker import get_worker, RabbitMQWorker
        
        worker = get_worker()
        assert isinstance(worker, RabbitMQWorker)
    
    def test_get_worker_with_detector(self):
        """Test get_worker with custom detector."""
        from src.worker import get_worker, RabbitMQWorker
        
        mock_detector = MagicMock()
        worker = get_worker(detector=mock_detector)
        
        assert isinstance(worker, RabbitMQWorker)
        assert worker._detector is mock_detector


class TestWorkerDuplicateDetection:
    """Tests for duplicate detection integration."""
    
    def test_worker_detects_duplicate(self, mock_rabbitmq_connection):
        """Test worker correctly handles duplicate detection result."""
        from src.worker import RabbitMQWorker
        from src.producer import ReclamationEvent
        from src.duplicate_detector import DuplicateDetectionResult, DuplicateAction
        
        mock_conn, mock_channel = mock_rabbitmq_connection
        
        # Create detector that returns duplicate
        mock_detector = MagicMock()
        mock_detector.process_reclamation.return_value = DuplicateDetectionResult(
            reclamation_id=1,
            is_duplicate=True,
            action=DuplicateAction.AUTO_MARK_DUPLICATE,
            matched_id=2,
            similarity_score=0.96,
            message="Matched with reclamation 2"
        )
        
        event = ReclamationEvent(reclamation_id=1, user_id=10)
        body = event.to_json().encode('utf-8')
        
        method = MagicMock()
        method.delivery_tag = 1
        properties = MagicMock()
        
        with patch('src.worker.pika') as mock_pika:
            mock_pika.BlockingConnection.return_value = mock_conn
            mock_pika.PlainCredentials.return_value = MagicMock()
            mock_pika.ConnectionParameters.return_value = MagicMock()
            
            with patch('src.worker.RabbitMQProducer'):
                worker = RabbitMQWorker(detector=mock_detector)
                worker._connection = mock_conn
                worker._channel = mock_channel
                
                # Track results
                results = []
                worker._on_result = lambda r: results.append(r)
                
                worker._process_message(mock_channel, method, properties, body)
        
        # Verify the result
        assert len(results) == 1
        assert results[0].is_duplicate is True
        assert results[0].matched_id == 2
        assert results[0].similarity_score == 0.96


