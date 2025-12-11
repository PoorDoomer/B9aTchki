"""
PostgreSQL LISTEN Service for Reclamation Events.

Listens for PostgreSQL NOTIFY events on the 'new_reclamation' channel
and publishes corresponding events to RabbitMQ for processing.

This service bridges PostgreSQL triggers to RabbitMQ, enabling
automatic event-driven processing when new reclamations are inserted.
"""

import json
import logging
import select
import signal
import sys
import time
from typing import Optional

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from src.config import get_config, PostgresConfig, RabbitMQConfig
from src.producer import RabbitMQProducer


logger = logging.getLogger(__name__)


class PostgresListener:
    """
    Service that listens for PostgreSQL NOTIFY events and publishes to RabbitMQ.
    
    Implements:
    - Reliable connection with auto-reconnect
    - Graceful shutdown handling
    - JSON payload parsing
    - RabbitMQ publishing
    """
    
    CHANNEL_NAME = "new_reclamation"
    RECONNECT_DELAY = 5  # seconds
    
    def __init__(
        self,
        pg_config: Optional[PostgresConfig] = None,
        rabbitmq_config: Optional[RabbitMQConfig] = None
    ):
        """
        Initialize the listener.
        
        Args:
            pg_config: Optional PostgreSQL configuration override.
            rabbitmq_config: Optional RabbitMQ configuration override.
        """
        self._pg_config = pg_config or get_config().postgres
        self._rabbitmq_config = rabbitmq_config or get_config().rabbitmq
        
        self._pg_conn: Optional[psycopg2.extensions.connection] = None
        self._producer: Optional[RabbitMQProducer] = None
        self._should_stop = False
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._should_stop = True
    
    def _connect_postgres(self) -> bool:
        """
        Establish connection to PostgreSQL.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self._pg_conn = psycopg2.connect(
                host=self._pg_config.host,
                port=self._pg_config.port,
                database=self._pg_config.database,
                user=self._pg_config.user,
                password=self._pg_config.password
            )
            
            # Set autocommit mode required for LISTEN
            self._pg_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            # Subscribe to the notification channel
            with self._pg_conn.cursor() as cur:
                cur.execute(f"LISTEN {self.CHANNEL_NAME};")
            
            logger.info(
                f"Connected to PostgreSQL at {self._pg_config.host}:{self._pg_config.port}, "
                f"listening on channel '{self.CHANNEL_NAME}'"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def _disconnect_postgres(self) -> None:
        """Close PostgreSQL connection."""
        if self._pg_conn and not self._pg_conn.closed:
            try:
                self._pg_conn.close()
                logger.info("Disconnected from PostgreSQL")
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL connection: {e}")
        self._pg_conn = None
    
    def _connect_rabbitmq(self) -> bool:
        """
        Establish connection to RabbitMQ.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self._producer = RabbitMQProducer(self._rabbitmq_config)
            self._producer.connect()
            logger.info(
                f"Connected to RabbitMQ at {self._rabbitmq_config.host}:{self._rabbitmq_config.port}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    def _disconnect_rabbitmq(self) -> None:
        """Close RabbitMQ connection."""
        if self._producer:
            try:
                self._producer.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting from RabbitMQ: {e}")
            self._producer = None
    
    def _handle_notification(self, payload: str) -> bool:
        """
        Handle a notification from PostgreSQL.
        
        Args:
            payload: JSON string with reclamation details.
            
        Returns:
            True if handled successfully, False otherwise.
        """
        try:
            data = json.loads(payload)
            reclamation_id = data.get("reclamation_id")
            reclamant_id = data.get("reclamant_id")
            operation = data.get("operation", "INSERT")
            
            if reclamation_id is None or reclamant_id is None:
                logger.warning(f"Invalid notification payload: {payload}")
                return False
            
            logger.info(
                f"Received notification: {operation} reclamation_id={reclamation_id}, "
                f"reclamant_id={reclamant_id}"
            )
            
            # Publish to RabbitMQ
            if self._producer:
                success = self._producer.publish_new_reclamation(
                    reclamation_id=reclamation_id,
                    reclamant_id=reclamant_id
                )
                
                if success:
                    logger.info(f"Published event to RabbitMQ for reclamation {reclamation_id}")
                else:
                    logger.error(f"Failed to publish event for reclamation {reclamation_id}")
                
                return success
            else:
                logger.error("RabbitMQ producer not available")
                return False
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse notification payload: {e}")
            return False
        except Exception as e:
            logger.error(f"Error handling notification: {e}", exc_info=True)
            return False
    
    def _process_notifications(self) -> int:
        """
        Process any pending notifications.
        
        Returns:
            Number of notifications processed.
        """
        if not self._pg_conn:
            return 0
        
        count = 0
        self._pg_conn.poll()
        
        while self._pg_conn.notifies:
            notify = self._pg_conn.notifies.pop(0)
            logger.debug(f"Received notify on channel {notify.channel}: {notify.payload}")
            
            if self._handle_notification(notify.payload):
                count += 1
        
        return count
    
    def start(self) -> None:
        """
        Start listening for notifications.
        
        Blocks until shutdown signal received or error occurs.
        """
        logger.info("Starting PostgreSQL listener service...")
        
        while not self._should_stop:
            # Connect to both services
            if not self._pg_conn or self._pg_conn.closed:
                if not self._connect_postgres():
                    logger.warning(f"Retrying PostgreSQL connection in {self.RECONNECT_DELAY}s...")
                    time.sleep(self.RECONNECT_DELAY)
                    continue
            
            if not self._producer:
                if not self._connect_rabbitmq():
                    logger.warning(f"Retrying RabbitMQ connection in {self.RECONNECT_DELAY}s...")
                    time.sleep(self.RECONNECT_DELAY)
                    continue
            
            try:
                # Wait for notifications with timeout
                # Using select for cross-platform compatibility
                if select.select([self._pg_conn], [], [], 1.0) == ([], [], []):
                    # Timeout - no notifications, check for shutdown
                    continue
                
                # Process any pending notifications
                self._process_notifications()
                
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                logger.warning(f"PostgreSQL connection lost: {e}")
                self._disconnect_postgres()
                
            except Exception as e:
                logger.error(f"Error in listener loop: {e}", exc_info=True)
                time.sleep(1)  # Brief pause before retry
        
        logger.info("Listener stopped")
        self._disconnect_postgres()
        self._disconnect_rabbitmq()
    
    def stop(self) -> None:
        """Signal the listener to stop."""
        self._should_stop = True


def run_listener() -> None:
    """
    Main entry point to run the listener.
    
    Sets up logging and starts the listener service.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting PostgreSQL -> RabbitMQ listener...")
    
    try:
        listener = PostgresListener()
        listener.start()
    except KeyboardInterrupt:
        logger.info("Listener interrupted by user")
    except Exception as e:
        logger.error(f"Listener failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_listener()
