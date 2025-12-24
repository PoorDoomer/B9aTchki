"""
Kafka Producer for publishing reclamation events.

Publishes "new_reclamation" events to the processing topic
when new reclamations are created.
"""

import json
import logging
from typing import Optional
from dataclasses import dataclass, asdict

from kafka import KafkaProducer as KafkaClient
from kafka.errors import KafkaError

from src.config import get_config, KafkaConfig


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


class KafkaEventProducer:
    """
    Producer for publishing reclamation events to Kafka.
    
    Handles connection management and message publishing.
    """
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        """
        Initialize the producer.
        
        Args:
            config: Optional Kafka configuration override.
        """
        self._config = config or get_config().kafka
        self._producer: Optional[KafkaClient] = None
        
    def connect(self) -> None:
        """
        Establish connection to Kafka.
        
        Creates the Kafka producer instance.
        """
        try:
            self._producer = KafkaClient(
                bootstrap_servers=self._config.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8') if isinstance(v, dict) else v,
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas to acknowledge
                retries=3,
                retry_backoff_ms=500
            )
            
            logger.info(f"Connected to Kafka at {self._config.bootstrap_servers}")
            
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close the connection to Kafka."""
        if self._producer:
            self._producer.flush()
            self._producer.close()
            logger.info("Disconnected from Kafka")
        self._producer = None
    
    def _ensure_connected(self) -> None:
        """Ensure we have an active connection."""
        if not self._producer:
            self.connect()
    
    def publish(self, event: ReclamationEvent) -> bool:
        """
        Publish a reclamation event to the topic.
        
        Args:
            event: The ReclamationEvent to publish.
        
        Returns:
            True if published successfully, False otherwise.
        """
        try:
            self._ensure_connected()
            
            message = asdict(event)
            
            future = self._producer.send(
                topic=self._config.topic,
                key=str(event.reclamation_id),
                value=message
            )
            
            # Wait for the message to be sent (with timeout)
            future.get(timeout=10)
            
            logger.debug(
                f"Published event: reclamation_id={event.reclamation_id}, "
                f"reclamant_id={event.reclamant_id}"
            )
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to publish event: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error publishing event: {e}")
            return False
    
    def publish_new_reclamation(self, reclamation_id: int, reclamant_id: int) -> bool:
        """
        Convenience method to publish a new reclamation event.
        
        Args:
            reclamation_id: The ID of the new reclamation.
            reclamant_id: The reclamant ID associated with the reclamation.
        
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
        Publish a failed event to the dead letter topic.
        
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
            dlq_data["original_topic"] = self._config.topic
            
            future = self._producer.send(
                topic=self._config.dlq_topic,
                key=str(event.reclamation_id),
                value=dlq_data
            )
            
            future.get(timeout=10)
            
            logger.warning(
                f"Published to DLQ: reclamation_id={event.reclamation_id}, "
                f"error={error_message}"
            )
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to publish to DLQ: {e}")
            return False
    
    def __enter__(self) -> "KafkaEventProducer":
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()


def get_producer(config: Optional[KafkaConfig] = None) -> KafkaEventProducer:
    """
    Factory function to get a KafkaEventProducer instance.
    
    Args:
        config: Optional configuration override.
    
    Returns:
        KafkaEventProducer instance.
    """
    return KafkaEventProducer(config)


def publish_reclamation_event(reclamation_id: int, reclamant_id: int) -> bool:
    """
    Convenience function to publish a reclamation event.
    
    Creates a producer, publishes the event, and closes the connection.
    
    Args:
        reclamation_id: The ID of the reclamation.
        reclamant_id: The reclamant ID.
    
    Returns:
        True if published successfully, False otherwise.
    """
    try:
        with KafkaEventProducer() as producer:
            return producer.publish_new_reclamation(reclamation_id, reclamant_id)
    except Exception as e:
        logger.error(f"Failed to publish reclamation event: {e}")
        return False
