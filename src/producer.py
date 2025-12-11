"""
RabbitMQ Producer for publishing reclamation events.

Publishes "new_reclamation" events to the processing queue
when new reclamations are created.
"""

import json
import logging
from typing import Optional
from dataclasses import dataclass, asdict

import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from src.config import get_config, RabbitMQConfig


logger = logging.getLogger(__name__)


@dataclass
class ReclamationEvent:
    """Event data for a new reclamation."""
    reclamation_id: int
    reclamant_id: int
    event_type: str = "new_reclamation"
    
    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> "ReclamationEvent":
        """Deserialize event from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class RabbitMQProducer:
    """
    Producer for publishing reclamation events to RabbitMQ.
    
    Handles connection management, queue declaration, and message publishing.
    """
    
    def __init__(self, config: Optional[RabbitMQConfig] = None):
        """
        Initialize the producer.
        
        Args:
            config: Optional RabbitMQ configuration override.
        """
        self._config = config or get_config().rabbitmq
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.channel.Channel] = None
        
    def connect(self) -> None:
        """
        Establish connection to RabbitMQ.
        
        Creates connection, channel, and declares the queue.
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
            
            # Declare the main processing queue
            self._channel.queue_declare(
                queue=self._config.queue,
                durable=True  # Persist queue across broker restarts
            )
            
            # Declare the dead letter queue
            self._channel.queue_declare(
                queue=self._config.dlq,
                durable=True
            )
            
            logger.info(f"Connected to RabbitMQ at {self._config.host}:{self._config.port}")
            
        except AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close the connection to RabbitMQ."""
        if self._connection and self._connection.is_open:
            self._connection.close()
            logger.info("Disconnected from RabbitMQ")
        self._connection = None
        self._channel = None
    
    def _ensure_connected(self) -> None:
        """Ensure we have an active connection."""
        if not self._connection or not self._connection.is_open:
            self.connect()
    
    def publish(self, event: ReclamationEvent) -> bool:
        """
        Publish a reclamation event to the queue.
        
        Args:
            event: The ReclamationEvent to publish.
        
        Returns:
            True if published successfully, False otherwise.
        """
        try:
            self._ensure_connected()
            
            message = event.to_json()
            
            self._channel.basic_publish(
                exchange='',
                routing_key=self._config.queue,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    content_type='application/json'
                )
            )
            
            logger.debug(
                f"Published event: reclamation_id={event.reclamation_id}, "
                f"user_id={event.reclamant_id}"
            )
            return True
            
        except (AMQPConnectionError, AMQPChannelError) as e:
            logger.error(f"Failed to publish event: {e}")
            self._connection = None
            self._channel = None
            return False
    
    def publish_new_reclamation(self, reclamation_id: int, reclamant_id: int) -> bool:
        """
        Convenience method to publish a new reclamation event.
        
        Args:
            reclamation_id: The ID of the new reclamation.
            user_id: The user ID associated with the reclamation.
        
        Returns:
            True if published successfully, False otherwise.
        """
        event = ReclamationEvent(
            reclamation_id=reclamation_id,
            reclamant_id=reclamant_id,
            event_type="new_reclamation"
        )
        return self.publish(event)
    
    def publish_to_dlq(self, event: ReclamationEvent, error_message: str) -> bool:
        """
        Publish a failed event to the dead letter queue.
        
        Args:
            event: The original ReclamationEvent that failed.
            error_message: Description of the failure.
        
        Returns:
            True if published successfully, False otherwise.
        """
        try:
            self._ensure_connected()
            
            # Add error information to the event
            dlq_data = asdict(event)
            dlq_data["error"] = error_message
            dlq_data["original_queue"] = self._config.queue
            
            message = json.dumps(dlq_data)
            
            self._channel.basic_publish(
                exchange='',
                routing_key=self._config.dlq,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type='application/json'
                )
            )
            
            logger.warning(
                f"Published to DLQ: reclamation_id={event.reclamation_id}, "
                f"error={error_message}"
            )
            return True
            
        except (AMQPConnectionError, AMQPChannelError) as e:
            logger.error(f"Failed to publish to DLQ: {e}")
            return False
    
    def __enter__(self) -> "RabbitMQProducer":
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()


def get_producer(config: Optional[RabbitMQConfig] = None) -> RabbitMQProducer:
    """
    Factory function to get a RabbitMQProducer instance.
    
    Args:
        config: Optional configuration override.
    
    Returns:
        RabbitMQProducer instance.
    """
    return RabbitMQProducer(config)


def publish_reclamation_event(reclamation_id: int, reclamant_id: int) -> bool:
    """
    Convenience function to publish a reclamation event.
    
    Creates a producer, publishes the event, and closes the connection.
    
    Args:
        reclamation_id: The ID of the reclamation.
        reclamant_id: The user ID.
    
    Returns:
        True if published successfully, False otherwise.
    """
    try:
        with RabbitMQProducer() as producer:
            return producer.publish_new_reclamation(reclamation_id, reclamant_id)
    except Exception as e:
        logger.error(f"Failed to publish reclamation event: {e}")
        return False


