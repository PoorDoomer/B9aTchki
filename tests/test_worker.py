"""
Unit tests for the Kafka worker module.
Tests message consumption and processing.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, call


class TestKafkaWorker:
    """Tests for KafkaWorker."""
    
    @pytest.fixture
    def mock_kafka(self):
        """Mock kafka library."""
        with patch('src.worker.KafkaConsumer') as mock_consumer_class:
            mock_consumer = MagicMock()
            mock_consumer_class.return_value = mock_consumer
            
            yield mock_consumer_class, mock_consumer
    
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
    
    def test_worker_connect(self, mock_kafka):
        """Test worker connection."""
        from src.worker import KafkaWorker
        
        mock_class, mock_consumer = mock_kafka
        
        with patch('src.worker.KafkaEventProducer') as MockProducer:
            mock_producer = MagicMock()
            MockProducer.return_value = mock_producer
            
            worker = KafkaWorker()
            worker.connect()
        
        mock_class.assert_called_once()
        mock_producer.connect.assert_called_once()
    
    def test_worker_disconnect(self, mock_kafka):
        """Test worker disconnection."""
        from src.worker import KafkaWorker
        
        mock_class, mock_consumer = mock_kafka
        
        with patch('src.worker.KafkaEventProducer') as MockProducer:
            mock_producer = MagicMock()
            MockProducer.return_value = mock_producer
            
            worker = KafkaWorker()
            worker.connect()
            worker.disconnect()
        
        mock_consumer.close.assert_called_once()
        mock_producer.disconnect.assert_called_once()
    
    def test_worker_process_message_success(self, mock_kafka, mock_detector):
        """Test successful message processing."""
        from src.worker import KafkaWorker
        
        mock_class, mock_consumer = mock_kafka
        
        # Create a mock Kafka message
        mock_message = MagicMock()
        mock_message.value = {
            'reclamation_id': 1,
            'reclamant_id': 10,
            'event_type': 'new_reclamation'
        }
        mock_message.topic = 'reclamation_processing'
        mock_message.partition = 0
        mock_message.offset = 0
        
        with patch('src.worker.KafkaEventProducer'):
            worker = KafkaWorker(detector=mock_detector)
            worker.connect()
            
            # Process the message
            result = worker._process_message(mock_message)
        
        # Verify detector was called
        mock_detector.process_reclamation.assert_called_once_with(1)
        
        # Verify result
        assert result is not None
        assert result.reclamation_id == 1
    
    def test_worker_process_message_with_callback(self, mock_kafka, mock_detector):
        """Test message processing with result callback."""
        from src.worker import KafkaWorker
        from src.duplicate_detector import DuplicateDetectionResult
        
        mock_class, mock_consumer = mock_kafka
        
        # Create callback to capture result
        results = []
        def on_result(result: DuplicateDetectionResult):
            results.append(result)
        
        mock_message = MagicMock()
        mock_message.value = {
            'reclamation_id': 1,
            'reclamant_id': 10,
            'event_type': 'new_reclamation'
        }
        mock_message.topic = 'reclamation_processing'
        mock_message.partition = 0
        mock_message.offset = 0
        
        with patch('src.worker.KafkaEventProducer'):
            worker = KafkaWorker(detector=mock_detector, on_result=on_result)
            worker.connect()
            worker._process_message(mock_message)
        
        # Verify callback was called
        assert len(results) == 1
        assert results[0].reclamation_id == 1
    
    def test_worker_process_invalid_json(self, mock_kafka):
        """Test handling of invalid JSON message."""
        from src.worker import KafkaWorker
        
        mock_class, mock_consumer = mock_kafka
        
        # Message with invalid structure
        mock_message = MagicMock()
        mock_message.value = "not a valid dict"
        mock_message.topic = 'reclamation_processing'
        mock_message.partition = 0
        mock_message.offset = 0
        
        with patch('src.worker.KafkaEventProducer') as MockProducer:
            mock_producer = MagicMock()
            MockProducer.return_value = mock_producer
            
            worker = KafkaWorker()
            worker.connect()
            result = worker._process_message(mock_message)
        
        # Should return None on error
        assert result is None
    
    def test_worker_context_manager(self, mock_kafka):
        """Test using worker as context manager."""
        from src.worker import KafkaWorker
        
        mock_class, mock_consumer = mock_kafka
        
        with patch('src.worker.KafkaEventProducer'):
            with KafkaWorker() as worker:
                assert worker._consumer is not None
        
        mock_consumer.close.assert_called()
    
    def test_worker_process_one(self, mock_kafka, mock_detector):
        """Test processing single message."""
        from src.worker import KafkaWorker
        
        mock_class, mock_consumer = mock_kafka
        
        mock_message = MagicMock()
        mock_message.value = {
            'reclamation_id': 1,
            'reclamant_id': 10,
            'event_type': 'new_reclamation'
        }
        mock_message.topic = 'reclamation_processing'
        mock_message.partition = 0
        mock_message.offset = 0
        
        # Mock poll to return one message
        mock_consumer.poll.return_value = {
            ('reclamation_processing', 0): [mock_message]
        }
        
        with patch('src.worker.KafkaEventProducer'):
            worker = KafkaWorker(detector=mock_detector)
            result = worker.process_one()
        
        assert result is not None
        assert result.reclamation_id == 1
    
    def test_worker_process_one_no_message(self, mock_kafka):
        """Test process_one with no messages in queue."""
        from src.worker import KafkaWorker
        
        mock_class, mock_consumer = mock_kafka
        
        # Mock poll to return no messages
        mock_consumer.poll.return_value = {}
        
        with patch('src.worker.KafkaEventProducer'):
            worker = KafkaWorker()
            result = worker.process_one()
        
        assert result is None


class TestWorkerSignalHandling:
    """Tests for worker signal handling."""
    
    def test_signal_handler_sets_stop_flag(self):
        """Test that signal handler sets stop flag."""
        from src.worker import KafkaWorker
        
        with patch('src.worker.KafkaConsumer'):
            with patch('src.worker.KafkaEventProducer'):
                worker = KafkaWorker()
                
                # Simulate signal
                worker._signal_handler(2, None)
                
                assert worker._should_stop is True


class TestWorkerFactoryFunctions:
    """Tests for worker factory functions."""
    
    def test_get_worker(self):
        """Test get_worker factory function."""
        from src.worker import get_worker, KafkaWorker
        
        worker = get_worker()
        assert isinstance(worker, KafkaWorker)
    
    def test_get_worker_with_detector(self):
        """Test get_worker with custom detector."""
        from src.worker import get_worker, KafkaWorker
        
        mock_detector = MagicMock()
        worker = get_worker(detector=mock_detector)
        
        assert isinstance(worker, KafkaWorker)
        assert worker._detector is mock_detector


class TestWorkerDuplicateDetection:
    """Tests for duplicate detection integration."""
    
    @pytest.fixture
    def mock_kafka_connection(self):
        """Mock Kafka consumer and producer."""
        mock_consumer = MagicMock()
        mock_producer = MagicMock()
        return mock_consumer, mock_producer
    
    def test_worker_detects_duplicate(self, mock_kafka_connection):
        """Test worker correctly handles duplicate detection result."""
        from src.worker import KafkaWorker
        from src.producer import ReclamationEvent
        from src.duplicate_detector import DuplicateDetectionResult, DuplicateAction
        
        mock_consumer, mock_producer = mock_kafka_connection
        
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
        
        mock_message = MagicMock()
        mock_message.value = {
            'reclamation_id': 1,
            'reclamant_id': 10,
            'event_type': 'new_reclamation'
        }
        mock_message.topic = 'reclamation_processing'
        mock_message.partition = 0
        mock_message.offset = 0
        
        with patch('src.worker.KafkaConsumer') as MockConsumer:
            MockConsumer.return_value = mock_consumer
            
            with patch('src.worker.KafkaEventProducer') as MockProducer:
                MockProducer.return_value = mock_producer
                
                worker = KafkaWorker(detector=mock_detector)
                worker._consumer = mock_consumer
                worker._producer = mock_producer
                
                # Track results
                results = []
                worker._on_result = lambda r: results.append(r)
                
                worker._process_message(mock_message)
        
        # Verify the result
        assert len(results) == 1
        assert results[0].is_duplicate is True
        assert results[0].matched_id == 2
        assert results[0].similarity_score == 0.96
