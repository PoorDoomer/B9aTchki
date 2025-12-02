"""
Pytest configuration and fixtures for the de-duplication pipeline tests.
"""

import os
import pytest
from typing import Generator, Optional
from unittest.mock import MagicMock, patch

# Set test environment variables before importing config
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_USER", "dedup_user")
os.environ.setdefault("POSTGRES_PASSWORD", "dedup_password")
os.environ.setdefault("POSTGRES_DB", "dedup_db")
os.environ.setdefault("RABBITMQ_HOST", "localhost")
os.environ.setdefault("RABBITMQ_PORT", "5672")


@pytest.fixture
def config():
    """Provide a fresh configuration instance for tests."""
    from src.config import Config, reset_config
    reset_config()
    return Config.load()


@pytest.fixture
def postgres_config():
    """Provide PostgreSQL configuration."""
    from src.config import PostgresConfig
    return PostgresConfig()


@pytest.fixture
def rabbitmq_config():
    """Provide RabbitMQ configuration."""
    from src.config import RabbitMQConfig
    return RabbitMQConfig()


@pytest.fixture
def ml_config():
    """Provide ML configuration."""
    from src.config import MLConfig
    return MLConfig()


@pytest.fixture
def detection_config():
    """Provide duplicate detection configuration."""
    from src.config import DuplicateDetectionConfig
    return DuplicateDetectionConfig()


@pytest.fixture
def sample_french_texts() -> list[str]:
    """Sample French reclamation texts for testing."""
    return [
        "Ma connexion internet ne fonctionne pas depuis hier",
        "L'internet est coupé depuis 24 heures",
        "Je n'ai plus d'accès à internet",
        "Mon abonnement téléphonique a un problème de facturation",
        "La facture de mon téléphone est incorrecte",
    ]


@pytest.fixture
def sample_arabic_texts() -> list[str]:
    """Sample Arabic reclamation texts for testing."""
    return [
        "الإنترنت مقطوع منذ أمس",
        "لا يوجد اتصال بالإنترنت",
        "خدمة الإنترنت لا تعمل",
        "فاتورة الهاتف خاطئة",
        "هناك مشكلة في فاتورة الاشتراك",
    ]


@pytest.fixture
def sample_reclamation_data() -> list[dict]:
    """Sample reclamation records for testing."""
    return [
        {"user_id": 1, "message_libre": "Ma connexion internet ne fonctionne pas depuis hier", "status": "PENDING"},
        {"user_id": 1, "message_libre": "L'internet est coupé depuis 24 heures", "status": "PENDING"},
        {"user_id": 1, "message_libre": "الإنترنت مقطوع منذ أمس", "status": "PENDING"},
        {"user_id": 2, "message_libre": "Mon abonnement téléphonique a un problème de facturation", "status": "PENDING"},
        {"user_id": 2, "message_libre": "فاتورة الهاتف خاطئة", "status": "PENDING"},
    ]


@pytest.fixture
def mock_db_connection():
    """Mock database connection for unit tests."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn, mock_cursor


@pytest.fixture
def mock_rabbitmq_connection():
    """Mock RabbitMQ connection for unit tests."""
    mock_conn = MagicMock()
    mock_channel = MagicMock()
    mock_conn.channel.return_value = mock_channel
    return mock_conn, mock_channel


