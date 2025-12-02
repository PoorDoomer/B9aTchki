"""
Unit tests for the database module.
Tests are performed using mocks to avoid requiring a real database.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime
from decimal import Decimal


class TestDatabasePool:
    """Tests for DatabasePool singleton."""
    
    def test_pool_singleton_pattern(self):
        """Test that DatabasePool implements singleton pattern."""
        from src.database import DatabasePool
        
        # Reset any existing pool
        DatabasePool.reset()
        
        with patch('src.database.pool.ThreadedConnectionPool'):
            with patch('src.database.get_config') as mock_config:
                mock_pg_config = MagicMock()
                mock_pg_config.host = "localhost"
                mock_pg_config.port = 5432
                mock_pg_config.database = "test_db"
                mock_pg_config.user = "test_user"
                mock_pg_config.password = "test_pass"
                mock_config.return_value.postgres = mock_pg_config
                
                pool1 = DatabasePool()
                pool2 = DatabasePool()
                
                assert pool1 is pool2
        
        DatabasePool.reset()
    
    def test_pool_reset(self):
        """Test that reset creates new instance."""
        from src.database import DatabasePool
        
        DatabasePool.reset()
        
        with patch('src.database.pool.ThreadedConnectionPool'):
            with patch('src.database.get_config') as mock_config:
                mock_pg_config = MagicMock()
                mock_pg_config.host = "localhost"
                mock_pg_config.port = 5432
                mock_pg_config.database = "test_db"
                mock_pg_config.user = "test_user"
                mock_pg_config.password = "test_pass"
                mock_config.return_value.postgres = mock_pg_config
                
                pool1 = DatabasePool()
                DatabasePool.reset()
                pool2 = DatabasePool()
                
                assert pool1 is not pool2
        
        DatabasePool.reset()


class TestReclamationRepository:
    """Tests for ReclamationRepository."""
    
    @pytest.fixture
    def mock_pool(self):
        """Create a mock database pool."""
        from src.database import DatabasePool
        DatabasePool.reset()
        
        mock = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock.get_connection.return_value.__exit__ = MagicMock(return_value=False)
        
        return mock, mock_conn, mock_cursor
    
    def test_create_reclamation(self, mock_pool):
        """Test creating a reclamation."""
        from src.database import ReclamationRepository
        
        mock, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.return_value = (123,)
        
        repo = ReclamationRepository(mock)
        result = repo.create(user_id=1, message_libre="Test message")
        
        assert result == 123
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    def test_get_by_id_found(self, mock_pool):
        """Test getting a reclamation by ID when found."""
        from src.database import ReclamationRepository, Reclamation
        
        mock, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "user_id": 10,
            "message_libre": "Test",
            "created_at": datetime.now(),
            "status": "PENDING"
        }
        
        repo = ReclamationRepository(mock)
        result = repo.get_by_id(1)
        
        assert result is not None
        assert isinstance(result, Reclamation)
        assert result.id == 1
        assert result.user_id == 10
    
    def test_get_by_id_not_found(self, mock_pool):
        """Test getting a reclamation by ID when not found."""
        from src.database import ReclamationRepository
        
        mock, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.return_value = None
        
        repo = ReclamationRepository(mock)
        result = repo.get_by_id(999)
        
        assert result is None
    
    def test_update_status(self, mock_pool):
        """Test updating reclamation status."""
        from src.database import ReclamationRepository
        
        mock, mock_conn, mock_cursor = mock_pool
        mock_cursor.rowcount = 1
        
        repo = ReclamationRepository(mock)
        result = repo.update_status(1, "PROCESSED")
        
        assert result is True
        mock_conn.commit.assert_called_once()
    
    def test_update_status_not_found(self, mock_pool):
        """Test updating status for non-existent reclamation."""
        from src.database import ReclamationRepository
        
        mock, mock_conn, mock_cursor = mock_pool
        mock_cursor.rowcount = 0
        
        repo = ReclamationRepository(mock)
        result = repo.update_status(999, "PROCESSED")
        
        assert result is False
    
    def test_delete_reclamation(self, mock_pool):
        """Test deleting a reclamation."""
        from src.database import ReclamationRepository
        
        mock, mock_conn, mock_cursor = mock_pool
        mock_cursor.rowcount = 1
        
        repo = ReclamationRepository(mock)
        result = repo.delete(1)
        
        assert result is True
        mock_conn.commit.assert_called_once()


class TestEmbeddingRepository:
    """Tests for EmbeddingRepository."""
    
    @pytest.fixture
    def mock_pool(self):
        """Create a mock database pool."""
        mock = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock.get_connection.return_value.__exit__ = MagicMock(return_value=False)
        
        return mock, mock_conn, mock_cursor
    
    def test_save_embedding(self, mock_pool):
        """Test saving an embedding."""
        from src.database import EmbeddingRepository
        
        mock, mock_conn, mock_cursor = mock_pool
        
        repo = EmbeddingRepository(mock)
        embedding = [0.1, 0.2, 0.3] * 256  # 768 dimensions
        
        repo.save_embedding(1, embedding)
        
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    def test_get_embedding_found(self, mock_pool):
        """Test getting an embedding when found."""
        from src.database import EmbeddingRepository
        
        mock, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.return_value = ("[0.1,0.2,0.3]",)
        
        repo = EmbeddingRepository(mock)
        result = repo.get_embedding(1)
        
        assert result is not None
        assert len(result) == 3
        assert result[0] == 0.1
    
    def test_get_embedding_not_found(self, mock_pool):
        """Test getting an embedding when not found."""
        from src.database import EmbeddingRepository
        
        mock, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.return_value = None
        
        repo = EmbeddingRepository(mock)
        result = repo.get_embedding(999)
        
        assert result is None
    
    def test_find_similar_with_results(self, mock_pool):
        """Test finding similar embeddings with results."""
        from src.database import EmbeddingRepository, SimilarityMatch
        
        mock, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchall.return_value = [
            {"reclamation_id": 2, "score": 0.95, "message_libre": "Similar", "status": "PENDING"},
            {"reclamation_id": 3, "score": 0.87, "message_libre": "Also similar", "status": "PENDING"}
        ]
        
        repo = EmbeddingRepository(mock)
        embedding = [0.1] * 768
        results = repo.find_similar(
            embedding=embedding,
            user_id=1,
            exclude_id=1,
            min_score=0.85
        )
        
        assert len(results) == 2
        assert all(isinstance(r, SimilarityMatch) for r in results)
        assert results[0].score == 0.95
        assert results[1].score == 0.87
    
    def test_find_similar_no_results(self, mock_pool):
        """Test finding similar embeddings with no results."""
        from src.database import EmbeddingRepository
        
        mock, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchall.return_value = []
        
        repo = EmbeddingRepository(mock)
        embedding = [0.1] * 768
        results = repo.find_similar(
            embedding=embedding,
            user_id=1,
            exclude_id=1,
            min_score=0.85
        )
        
        assert len(results) == 0


class TestDuplicationLogRepository:
    """Tests for DuplicationLogRepository."""
    
    @pytest.fixture
    def mock_pool(self):
        """Create a mock database pool."""
        mock = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock.get_connection.return_value.__exit__ = MagicMock(return_value=False)
        
        return mock, mock_conn, mock_cursor
    
    def test_create_log(self, mock_pool):
        """Test creating a duplication log entry."""
        from src.database import DuplicationLogRepository
        
        mock, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchone.return_value = (1,)
        
        repo = DuplicationLogRepository(mock)
        result = repo.create(
            source_reclamation_id=1,
            matched_reclamation_id=2,
            similarity_score=0.95,
            action="AUTO_MARK_DUPLICATE"
        )
        
        assert result == 1
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    def test_get_by_source(self, mock_pool):
        """Test getting log entries by source reclamation."""
        from src.database import DuplicationLogRepository, DuplicationLog
        
        mock, mock_conn, mock_cursor = mock_pool
        mock_cursor.fetchall.return_value = [
            {
                "id": 1,
                "source_reclamation_id": 1,
                "matched_reclamation_id": 2,
                "similarity_score": Decimal("0.9500"),
                "action": "AUTO_MARK_DUPLICATE",
                "detected_at": datetime.now()
            }
        ]
        
        repo = DuplicationLogRepository(mock)
        results = repo.get_by_source(1)
        
        assert len(results) == 1
        assert isinstance(results[0], DuplicationLog)
        assert results[0].source_reclamation_id == 1


class TestDataClasses:
    """Tests for data classes."""
    
    def test_reclamation_dataclass(self):
        """Test Reclamation dataclass."""
        from src.database import Reclamation
        
        rec = Reclamation(
            id=1,
            user_id=10,
            message_libre="Test",
            created_at=datetime.now(),
            status="PENDING"
        )
        
        assert rec.id == 1
        assert rec.user_id == 10
        assert rec.message_libre == "Test"
        assert rec.status == "PENDING"
    
    def test_similarity_match_dataclass(self):
        """Test SimilarityMatch dataclass."""
        from src.database import SimilarityMatch
        
        match = SimilarityMatch(
            reclamation_id=2,
            score=0.95,
            message_libre="Similar text",
            status="PENDING"
        )
        
        assert match.reclamation_id == 2
        assert match.score == 0.95
        assert match.message_libre == "Similar text"
    
    def test_similarity_match_optional_fields(self):
        """Test SimilarityMatch with optional fields."""
        from src.database import SimilarityMatch
        
        match = SimilarityMatch(reclamation_id=2, score=0.95)
        
        assert match.reclamation_id == 2
        assert match.score == 0.95
        assert match.message_libre is None
        assert match.status is None


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_get_reclamation_repo(self):
        """Test get_reclamation_repo factory function."""
        from src.database import get_reclamation_repo, ReclamationRepository
        
        mock_pool = MagicMock()
        repo = get_reclamation_repo(mock_pool)
        
        assert isinstance(repo, ReclamationRepository)
    
    def test_get_embedding_repo(self):
        """Test get_embedding_repo factory function."""
        from src.database import get_embedding_repo, EmbeddingRepository
        
        mock_pool = MagicMock()
        repo = get_embedding_repo(mock_pool)
        
        assert isinstance(repo, EmbeddingRepository)
    
    def test_get_duplication_log_repo(self):
        """Test get_duplication_log_repo factory function."""
        from src.database import get_duplication_log_repo, DuplicationLogRepository
        
        mock_pool = MagicMock()
        repo = get_duplication_log_repo(mock_pool)
        
        assert isinstance(repo, DuplicationLogRepository)


