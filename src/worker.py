"""
Kafka Worker for consuming and processing reclamation events.

Consumes messages from the processing topic and executes the
duplicate detection pipeline for each new reclamation.
"""

import json
import logging
import signal
import sys
from typing import Optional, Callable

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from src.config import get_config, KafkaConfig
from src.producer import ReclamationEvent, KafkaEventProducer
from src.duplicate_detector import DuplicateDetector, DuplicateDetectionResult
from src.embeddings import get_embedding_model


logger = logging.getLogger(__name__)


class KafkaWorker:
    """
    Worker for consuming reclamation events from Kafka.
    
    Implements:
    - Reliable message consumption with manual commits
    - Graceful shutdown handling
    - Dead letter topic for failed messages
    - Connection recovery
    """
    
    def __init__(
        self,
        config: Optional[KafkaConfig] = None,
        detector: Optional[DuplicateDetector] = None,
        on_result: Optional[Callable[[DuplicateDetectionResult], None]] = None
    ):
        """
        Initialize the worker.
        
        Args:
            config: Optional Kafka configuration override.
            detector: Optional DuplicateDetector instance.
            on_result: Optional callback for processing results.
        """
        self._config = config or get_config().kafka
        self._detector = detector
        self._on_result = on_result
        self._consumer: Optional[KafkaConsumer] = None
        self._producer: Optional[KafkaEventProducer] = None
        self._should_stop = False
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._should_stop = True
    
    def _get_detector(self) -> DuplicateDetector:
        """Get or create the duplicate detector."""
        if self._detector is None:
            self._detector = DuplicateDetector()
        return self._detector
    
    def connect(self) -> None:
        """
        Establish connection to Kafka.
        
        Creates consumer and producer instances.
        """
        try:
            self._consumer = KafkaConsumer(
                self._config.topic,
                bootstrap_servers=self._config.bootstrap_servers,
                group_id=self._config.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='earliest',
                enable_auto_commit=False,  # Manual commit for reliability
                max_poll_interval_ms=300000,  # 5 minutes
                session_timeout_ms=30000
            )
            
            # Create producer for DLQ publishing
            self._producer = KafkaEventProducer(self._config)
            self._producer.connect()
            
            logger.info(f"Worker connected to Kafka at {self._config.bootstrap_servers}")
            
        except KafkaError as e:
            logger.error(f"Worker failed to connect to Kafka: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close connections."""
        if self._producer:
            self._producer.disconnect()
            self._producer = None
        
        if self._consumer:
            self._consumer.close()
            logger.info("Worker disconnected from Kafka")
        
        self._consumer = None
    
    def _process_message(self, message) -> Optional[DuplicateDetectionResult]:
        """
        Process a single message from Kafka.
        
        Args:
            message: The Kafka message to process.
            
        Returns:
            DuplicateDetectionResult if successful, None otherwise.
        """
        try:
            # Parse the event from message value
            data = message.value
            event = ReclamationEvent(
                reclamation_id=data['reclamation_id'],
                reclamant_id=data['reclamant_id'],
                event_type=data.get('event_type', 'new_reclamation')
            )
            logger.info(f"Processing reclamation {event.reclamation_id} for user {event.reclamant_id}")
            
            # Execute duplicate detection
            detector = self._get_detector()
            result = detector.process_reclamation(event.reclamation_id)
            
            # Invoke callback if provided
            if self._on_result:
                self._on_result(result)
            
            # Log the result
            logger.info(
                f"Reclamation {result.reclamation_id}: "
                f"action={result.action.value}, "
                f"is_duplicate={result.is_duplicate}"
            )
            if result.matched_id:
                logger.info(
                    f"  -> Matched with {result.matched_id} "
                    f"(score={result.similarity_score:.4f})"
                )
            
            return result
            
        except json.JSONDecodeError as e:
            # Invalid JSON - send to DLQ
            logger.error(f"Invalid JSON in message: {e}")
            self._send_to_dlq_raw(message, f"Invalid JSON: {str(e)}")
            return None
            
        except Exception as e:
            # Processing error - send to DLQ
            logger.error(f"Error processing message: {e}", exc_info=True)
            
            try:
                data = message.value
                event = ReclamationEvent(
                    reclamation_id=data['reclamation_id'],
                    reclamant_id=data['reclamant_id']
                )
                if self._producer:
                    self._producer.publish_to_dlq(event, str(e))
            except Exception:
                self._send_to_dlq_raw(message, f"Processing error: {str(e)}")
            
            return None
    
    def _send_to_dlq_raw(self, message, error: str) -> None:
        """Send a raw message to the DLQ."""
        if self._producer and self._producer._producer:
            try:
                dlq_message = {
                    "original_value": str(message.value),
                    "original_topic": message.topic,
                    "original_partition": message.partition,
                    "original_offset": message.offset,
                    "error": error
                }
                self._producer._producer.send(
                    topic=self._config.dlq_topic,
                    value=dlq_message
                ).get(timeout=10)
            except Exception as e:
                logger.error(f"Failed to send to DLQ: {e}")
    
    def start(self) -> None:
        """
        Start consuming messages from the topic.
        
        Blocks until shutdown signal received or error occurs.
        """
        logger.info(f"Starting worker, listening on topic: {self._config.topic}")
        
        try:
            self.connect()
            
            logger.info("Worker started, waiting for messages...")
            
            while not self._should_stop:
                try:
                    # Poll for messages with timeout
                    messages = self._consumer.poll(timeout_ms=1000)
                    
                    for topic_partition, records in messages.items():
                        for message in records:
                            if self._should_stop:
                                break
                            
                            self._process_message(message)
                            
                            # Commit the offset after successful processing
                            self._consumer.commit()
                    
                except KafkaError as e:
                    if not self._should_stop:
                        logger.warning(f"Kafka error, reconnecting: {e}")
                        self.disconnect()
                        self.connect()
            
            logger.info("Worker stopped gracefully")
            
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            raise
        
        finally:
            self.disconnect()
    
    def process_one(self, timeout: float = 5.0) -> Optional[DuplicateDetectionResult]:
        """
        Process a single message and return.
        
        Useful for testing and batch processing.
        
        Args:
            timeout: Maximum time to wait for a message in seconds.
        
        Returns:
            DuplicateDetectionResult if a message was processed, None if timeout.
        """
        result = None
        
        try:
            self.connect()
            
            # Poll for one message
            messages = self._consumer.poll(timeout_ms=int(timeout * 1000), max_records=1)
            
            for topic_partition, records in messages.items():
                for message in records:
                    result = self._process_message(message)
                    self._consumer.commit()
                    break
                break
            
            return result
            
        finally:
            self.disconnect()
    
    def process_batch(self, batch_size: int = 10) -> list[DuplicateDetectionResult]:
        """
        Process multiple messages in a batch.
        
        Efficiently processes up to batch_size messages before disconnecting.
        Uses the same detector instance and connection for all messages.
        
        Args:
            batch_size: Maximum number of messages to process.
        
        Returns:
            List of DuplicateDetectionResult objects for processed messages.
        """
        results = []
        
        try:
            self.connect()
            
            # Pre-initialize the detector (which loads the model)
            detector = self._get_detector()
            
            processed = 0
            while processed < batch_size:
                # Poll for messages
                messages = self._consumer.poll(timeout_ms=1000, max_records=batch_size - processed)
                
                if not messages:
                    # No more messages
                    logger.info(f"Batch complete: processed {processed} messages (no more messages)")
                    break
                
                for topic_partition, records in messages.items():
                    for message in records:
                        if processed >= batch_size:
                            break
                        
                        try:
                            data = message.value
                            event = ReclamationEvent(
                                reclamation_id=data['reclamation_id'],
                                reclamant_id=data['reclamant_id']
                            )
                            logger.info(f"Batch processing reclamation {event.reclamation_id}")
                            
                            result = detector.process_reclamation(event.reclamation_id)
                            results.append(result)
                            
                            self._consumer.commit()
                            processed += 1
                            
                        except Exception as e:
                            logger.error(f"Error in batch processing: {e}")
                            self._consumer.commit()
                            # Send to DLQ
                            try:
                                event = ReclamationEvent(
                                    reclamation_id=data['reclamation_id'],
                                    reclamant_id=data['reclamant_id']
                                )
                                if self._producer:
                                    self._producer.publish_to_dlq(event, str(e))
                            except Exception:
                                self._send_to_dlq_raw(message, f"Batch error: {str(e)}")
            
            logger.info(f"Batch processing complete: {processed} messages processed")
            return results
            
        finally:
            self.disconnect()

    def __enter__(self) -> "KafkaWorker":
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()


def get_worker(
    config: Optional[KafkaConfig] = None,
    detector: Optional[DuplicateDetector] = None
) -> KafkaWorker:
    """
    Factory function to get a KafkaWorker instance.
    
    Args:
        config: Optional configuration override.
        detector: Optional DuplicateDetector instance.
    
    Returns:
        KafkaWorker instance.
    """
    return KafkaWorker(config, detector)


def run_worker() -> None:
    """
    Main entry point to run the worker.
    
    Sets up logging, preloads the embedding model, and starts the worker.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting de-duplication worker...")
    
    # Preload the embedding model to avoid cold start latency
    logger.info("Preloading LaBSE embedding model...")
    try:
        model = get_embedding_model()
        # Trigger actual model loading by accessing the model property
        _ = model.model
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}", exc_info=True)
        sys.exit(1)
    
    try:
        worker = KafkaWorker()
        worker.start()
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_worker()
