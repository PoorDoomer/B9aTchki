"""
Unit tests for the configuration module.
Tests environment variable loading and configuration defaults.
"""

import os
import pytest
from unittest.mock import patch


class TestPostgresConfig:
    """Tests for PostgreSQL configuration."""
    
    def test_default_values(self, postgres_config):
        """Test that default values are set correctly."""
        assert postgres_config.host == "localhost"
        assert postgres_config.port == 5432
        assert postgres_config.user == "dedup_user"
        assert postgres_config.database == "dedup_db"
    
    def test_connection_string_format(self, postgres_config):
        """Test connection string format."""
        conn_str = postgres_config.connection_string
        assert "host=localhost" in conn_str
        assert "port=5432" in conn_str
        assert "dbname=dedup_db" in conn_str
        assert "user=dedup_user" in conn_str
    
    def test_dsn_format(self, postgres_config):
        """Test DSN format connection string."""
        dsn = postgres_config.dsn
        assert dsn.startswith("postgresql://")
        assert "dedup_user" in dsn
        assert "localhost:5432" in dsn
        assert "dedup_db" in dsn


class TestRabbitMQConfig:
    """Tests for RabbitMQ configuration."""
    
    def test_default_values(self, rabbitmq_config):
        """Test that default values are set correctly."""
        assert rabbitmq_config.host == "localhost"
        assert rabbitmq_config.port == 5672
        assert rabbitmq_config.user == "dedup_user"
        assert rabbitmq_config.queue == "reclamation_processing"
        assert rabbitmq_config.dlq == "dead_letter_queue"
    
    def test_connection_url_format(self, rabbitmq_config):
        """Test AMQP connection URL format."""
        url = rabbitmq_config.connection_url
        assert url.startswith("amqp://")
        assert "dedup_user" in url
        assert "localhost:5672" in url


class TestMLConfig:
    """Tests for ML configuration."""
    
    def test_default_values(self, ml_config):
        """Test that default ML values are set correctly."""
        assert ml_config.model_name == "sentence-transformers/LaBSE"
        assert ml_config.embedding_dimension == 768


class TestDuplicateDetectionConfig:
    """Tests for duplicate detection configuration."""
    
    def test_default_thresholds(self, detection_config):
        """Test that default thresholds are set correctly."""
        assert detection_config.threshold_auto_duplicate == 0.95
        assert detection_config.threshold_review == 0.85
        assert detection_config.time_window_days == 7
    
    def test_threshold_ordering(self, detection_config):
        """Test that auto_duplicate threshold is higher than review threshold."""
        assert detection_config.threshold_auto_duplicate > detection_config.threshold_review


class TestMainConfig:
    """Tests for the main Config class."""
    
    def test_config_load(self, config):
        """Test that config loads all sub-configurations."""
        assert config.postgres is not None
        assert config.rabbitmq is not None
        assert config.ml is not None
        assert config.detection is not None
    
    def test_get_config_singleton(self):
        """Test that get_config returns singleton instance."""
        from src.config import get_config, reset_config
        
        reset_config()
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_reset_config(self):
        """Test that reset_config creates new instance."""
        from src.config import get_config, reset_config
        
        config1 = get_config()
        reset_config()
        config2 = get_config()
        
        assert config1 is not config2


class TestEnvironmentOverrides:
    """Tests for environment variable overrides."""
    
    def test_postgres_host_override(self):
        """Test that POSTGRES_HOST env var overrides default."""
        from src.config import PostgresConfig, reset_config
        
        with patch.dict(os.environ, {"POSTGRES_HOST": "custom-host"}):
            reset_config()
            config = PostgresConfig()
            assert config.host == "custom-host"
    
    def test_threshold_override(self):
        """Test that threshold env vars override defaults."""
        from src.config import DuplicateDetectionConfig, reset_config
        
        with patch.dict(os.environ, {"THRESHOLD_AUTO_DUPLICATE": "0.98"}):
            reset_config()
            config = DuplicateDetectionConfig()
            assert config.threshold_auto_duplicate == 0.98
    
    def test_ml_model_override(self):
        """Test that ML model env var overrides default."""
        from src.config import MLConfig, reset_config
        
        with patch.dict(os.environ, {"LABSE_MODEL": "custom-model"}):
            reset_config()
            config = MLConfig()
            assert config.model_name == "custom-model"


