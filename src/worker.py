"""
RabbitMQ Worker for consuming and processing reclamation events.

Consumes messages from the processing queue and executes the
duplicate detection pipeline for each new reclamation.
"""

import json
import logging
import signal
import sys
from typing import Optional, Callable

import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from src.config import get_config, RabbitMQConfig
from src.producer import ReclamationEvent, RabbitMQProducer
from src.duplicate_detector import DuplicateDetector, DuplicateDetectionResult


logger = logging.getLogger(__name__)


class RabbitMQWorker:
    """
    Worker for consuming reclamation events from RabbitMQ.
    
    Implements:
    - Reliable message consumption with acknowledgments
    - Graceful shutdown handling
    - Dead letter queue for failed messages
    - Connection recovery
    """
    
    def __init__(
        self,
        config: Optional[RabbitMQConfig] = None,
        detector: Optional[DuplicateDetector] = None,
        on_result: Optional[Callable[[DuplicateDetectionResult], None]] = None
    ):
        """
        Initialize the worker.
        
        Args:
            config: Optional RabbitMQ configuration override.
            detector: Optional DuplicateDetector instance.
            on_result: Optional callback for processing results.
        """
        self._config = config or get_config().rabbitmq
        self._detector = detector
        self._on_result = on_result
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.channel.Channel] = None
        self._producer: Optional[RabbitMQProducer] = None
        self._should_stop = False
        self._consumer_tag: Optional[str] = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._should_stop = True
        if self._channel:
            try:
                self._channel.stop_consuming()
            except Exception:
                pass
    
    def _get_detector(self) -> DuplicateDetector:
        """Get or create the duplicate detector."""
        if self._detector is None:
            self._detector = DuplicateDetector()
        return self._detector
    
    def connect(self) -> None:
        """
        Establish connection to RabbitMQ.
        
        Creates connection, channel, and declares queues.
        """
        try:
            credentials = pika.PlainCredentials(
                self._config.user,
                self._config.password
            )
            parameters = pika.ConnectionParameters(
                host=self._config.host,
                port=self._config.port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            
            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()
            
            # Set QoS - process one message at a time
            self._channel.basic_qos(prefetch_count=1)
            
            # Declare queues (idempotent)
            self._channel.queue_declare(
                queue=self._config.queue,
                durable=True
            )
            self._channel.queue_declare(
                queue=self._config.dlq,
                durable=True
            )
            
            # Create producer for DLQ publishing
            self._producer = RabbitMQProducer(self._config)
            self._producer.connect()
            
            logger.info(f"Worker connected to RabbitMQ at {self._config.host}:{self._config.port}")
            
        except AMQPConnectionError as e:
            logger.error(f"Worker failed to connect to RabbitMQ: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close connections."""
        if self._producer:
            self._producer.disconnect()
            self._producer = None
        
        if self._connection and self._connection.is_open:
            self._connection.close()
            logger.info("Worker disconnected from RabbitMQ")
        
        self._connection = None
        self._channel = None
    
    def _process_message(
        self,
        channel: pika.channel.Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes
    ) -> None:
        """
        Process a single message from the queue.
        
        Args:
            channel: The channel object.
            method: Delivery method.
            properties: Message properties.
            body: Message body.
        """
        delivery_tag = method.delivery_tag
        
        try:
            # Parse the event
            event = ReclamationEvent.from_json(body.decode('utf-8'))
            logger.info(f"Processing reclamation {event.reclamation_id} for user {event.user_id}")
            
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
            
            # Acknowledge successful processing
            channel.basic_ack(delivery_tag=delivery_tag)
            
        except json.JSONDecodeError as e:
            # Invalid JSON - send to DLQ
            logger.error(f"Invalid JSON in message: {e}")
            self._send_to_dlq(body, f"Invalid JSON: {str(e)}")
            channel.basic_ack(delivery_tag=delivery_tag)
            
        except Exception as e:
            # Processing error - send to DLQ
            logger.error(f"Error processing message: {e}", exc_info=True)
            
            try:
                event = ReclamationEvent.from_json(body.decode('utf-8'))
                if self._producer:
                    self._producer.publish_to_dlq(event, str(e))
            except Exception:
                self._send_to_dlq(body, f"Processing error: {str(e)}")
            
            # Acknowledge to remove from main queue
            channel.basic_ack(delivery_tag=delivery_tag)
    
    def _send_to_dlq(self, body: bytes, error: str) -> None:
        """Send a raw message to the DLQ."""
        if self._channel and self._channel.is_open:
            try:
                dlq_message = {
                    "original_body": body.decode('utf-8', errors='replace'),
                    "error": error
                }
                self._channel.basic_publish(
                    exchange='',
                    routing_key=self._config.dlq,
                    body=json.dumps(dlq_message),
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                        content_type='application/json'
                    )
                )
            except Exception as e:
                logger.error(f"Failed to send to DLQ: {e}")
    
    def start(self) -> None:
        """
        Start consuming messages from the queue.
        
        Blocks until shutdown signal received or error occurs.
        """
        logger.info(f"Starting worker, listening on queue: {self._config.queue}")
        
        try:
            self.connect()
            
            # Set up consumer
            self._consumer_tag = self._channel.basic_consume(
                queue=self._config.queue,
                on_message_callback=self._process_message,
                auto_ack=False  # Manual acknowledgment
            )
            
            logger.info("Worker started, waiting for messages...")
            
            # Start consuming
            while not self._should_stop:
                try:
                    self._channel.start_consuming()
                except AMQPConnectionError:
                    if not self._should_stop:
                        logger.warning("Connection lost, reconnecting...")
                        self.connect()
                        self._consumer_tag = self._channel.basic_consume(
                            queue=self._config.queue,
                            on_message_callback=self._process_message,
                            auto_ack=False
                        )
            
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
        result_holder = [None]
        
        def callback(result: DuplicateDetectionResult):
            result_holder[0] = result
        
        original_callback = self._on_result
        self._on_result = callback
        
        try:
            self.connect()
            
            # Try to get one message
            method, properties, body = self._channel.basic_get(
                queue=self._config.queue,
                auto_ack=False
            )
            
            if method:
                self._process_message(self._channel, method, properties, body)
            
            return result_holder[0]
            
        finally:
            self._on_result = original_callback
            self.disconnect()
    
    def get_queue_size(self) -> int:
        """
        Get the number of messages in the queue.
        
        Returns:
            Number of messages waiting in the queue.
        """
        try:
            if not self._channel or not self._channel.is_open:
                self.connect()
            
            queue_state = self._channel.queue_declare(
                queue=self._config.queue,
                durable=True,
                passive=True  # Don't create, just check
            )
            return queue_state.method.message_count
            
        except Exception as e:
            logger.error(f"Failed to get queue size: {e}")
            return -1
    
    def __enter__(self) -> "RabbitMQWorker":
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()


def get_worker(
    config: Optional[RabbitMQConfig] = None,
    detector: Optional[DuplicateDetector] = None
) -> RabbitMQWorker:
    """
    Factory function to get a RabbitMQWorker instance.
    
    Args:
        config: Optional configuration override.
        detector: Optional DuplicateDetector instance.
    
    Returns:
        RabbitMQWorker instance.
    """
    return RabbitMQWorker(config, detector)


def run_worker() -> None:
    """
    Main entry point to run the worker.
    
    Sets up logging and starts the worker.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting de-duplication worker...")
    
    try:
        worker = RabbitMQWorker()
        worker.start()
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_worker()


