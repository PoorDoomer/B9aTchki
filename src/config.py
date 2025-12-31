"""
Configuration management for the De-duplication Pipeline.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Try to load .env file from project directory only (not home directory)
try:
    from dotenv import load_dotenv
    # Explicitly look for .env in the project root (where src/ is located)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        try:
            load_dotenv(env_file, encoding='utf-8')
        except Exception:
            # If .env file has encoding issues, skip it and use environment variables
            pass
except ImportError:
    pass


@dataclass
class PostgresConfig:
    """PostgreSQL connection configuration."""
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    
    def __post_init__(self):
        """Load values from environment if not provided."""
        if self.host is None:
            self.host = os.getenv("POSTGRES_HOST", "localhost")
        if self.port is None:
            self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        if self.user is None:
            self.user = os.getenv("POSTGRES_USER", "dedup_user")
        if self.password is None:
            self.password = os.getenv("POSTGRES_PASSWORD", "dedup_password")
        if self.database is None:
            self.database = os.getenv("POSTGRES_DB", "dedup_db")
    
    @property
    def connection_string(self) -> str:
        """Return psycopg2 connection string."""
        return f"host={self.host} port={self.port} dbname={self.database} user={self.user} password={self.password}"
    
    @property
    def dsn(self) -> str:
        """Return DSN format connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class KafkaConfig:
    """Kafka connection configuration."""
    bootstrap_servers: Optional[str] = None
    topic: Optional[str] = None
    dlq_topic: Optional[str] = None
    consumer_group: Optional[str] = None
    
    def __post_init__(self):
        """Load values from environment if not provided."""
        if self.bootstrap_servers is None:
            self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        if self.topic is None:
            self.topic = os.getenv("KAFKA_TOPIC", "reclamation_processing")
        if self.dlq_topic is None:
            self.dlq_topic = os.getenv("KAFKA_DLQ_TOPIC", "dead_letter_topic")
        if self.consumer_group is None:
            self.consumer_group = os.getenv("KAFKA_CONSUMER_GROUP", "dedup_workers")


@dataclass
class MLConfig:
    """Machine Learning configuration."""
    model_name: Optional[str] = None
    embedding_dimension: Optional[int] = None
    max_seq_tokens: Optional[int] = None
    def __post_init__(self):
        """Load values from environment if not provided."""
        if self.model_name is None:
            self.model_name = os.getenv("LABSE_MODEL", "sentence-transformers/LaBSE")
        if self.embedding_dimension is None:
            self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))
        if self.max_seq_tokens is None:
            self.max_seq_tokens = int(os.getenv("MAX_SEQ_TOKENS", "256"))

@dataclass  
class DuplicateDetectionConfig:
    """Duplicate detection thresholds and parameters."""
    threshold_auto_duplicate: Optional[float] = None
    threshold_review: Optional[float] = None
    time_window_days: Optional[int] = None
    
    def __post_init__(self):
        """Load values from environment if not provided."""
        # Thresholds lowered to better capture cross-lingual paraphrased duplicates:
        # - LaBSE gives ~0.55-0.80 for FR/AR insurance descriptions about same topic
        # - AUTO_DUPLICATE: 0.85+ means very similar phrasing (was 0.95)
        # - REVIEW: 0.70+ means same topic with different wording (was 0.85)
        if self.threshold_auto_duplicate is None:
            self.threshold_auto_duplicate = float(os.getenv("THRESHOLD_AUTO_DUPLICATE", "0.85"))
        if self.threshold_review is None:
            self.threshold_review = float(os.getenv("THRESHOLD_REVIEW", "0.70"))
        if self.time_window_days is None:
            self.time_window_days = int(os.getenv("TIME_WINDOW_DAYS", "7"))


@dataclass
class Config:
    """Main configuration container."""
    postgres: PostgresConfig
    kafka: KafkaConfig
    ml: MLConfig
    detection: DuplicateDetectionConfig
    
    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment."""
        return cls(
            postgres=PostgresConfig(),
            kafka=KafkaConfig(),
            ml=MLConfig(),
            detection=DuplicateDetectionConfig()
        )


# Singleton config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the singleton configuration instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reset_config() -> None:
    """Reset config for testing purposes."""
    global _config
    _config = None

