"""
Unit tests for the duplicate detector module.
Tests threshold logic and action determination.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


class TestDuplicateAction:
    """Tests for DuplicateAction enum."""
    
    def test_action_values(self):
        """Test DuplicateAction enum values."""
        from src.duplicate_detector import DuplicateAction
        
        assert DuplicateAction.AUTO_MARK_DUPLICATE.value == "AUTO_MARK_DUPLICATE"
        assert DuplicateAction.FLAG_FOR_REVIEW.value == "FLAG_FOR_REVIEW"
        assert DuplicateAction.NO_ACTION.value == "NO_ACTION"


class TestReclamationStatus:
    """Tests for ReclamationStatus enum."""
    
    def test_status_values(self):
        """Test ReclamationStatus enum values."""
        from src.duplicate_detector import ReclamationStatus
        
        assert ReclamationStatus.PENDING.value == "PENDING"
        assert ReclamationStatus.DUPLICATE.value == "DUPLICATE"
        assert ReclamationStatus.POTENTIAL_DUPLICATE.value == "POTENTIAL_DUPLICATE"
        assert ReclamationStatus.PROCESSED.value == "PROCESSED"


class TestDuplicateDetectionResult:
    """Tests for DuplicateDetectionResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a detection result."""
        from src.duplicate_detector import DuplicateDetectionResult, DuplicateAction
        
        result = DuplicateDetectionResult(
            reclamation_id=1,
            is_duplicate=True,
            action=DuplicateAction.AUTO_MARK_DUPLICATE,
            matched_id=2,
            similarity_score=0.96,
            message="Matched with reclamation 2"
        )
        
        assert result.reclamation_id == 1
        assert result.is_duplicate is True
        assert result.action == DuplicateAction.AUTO_MARK_DUPLICATE
        assert result.matched_id == 2
        assert result.similarity_score == 0.96
    
    def test_result_defaults(self):
        """Test result default values."""
        from src.duplicate_detector import DuplicateDetectionResult, DuplicateAction
        
        result = DuplicateDetectionResult(
            reclamation_id=1,
            is_duplicate=False,
            action=DuplicateAction.NO_ACTION
        )
        
        assert result.matched_id is None
        assert result.similarity_score is None
        assert result.message == ""


class TestDuplicateDetectorThresholds:
    """Tests for threshold logic in DuplicateDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create a DuplicateDetector with mocked dependencies."""
        from src.duplicate_detector import DuplicateDetector
        from src.config import DuplicateDetectionConfig
        
        config = DuplicateDetectionConfig(
            threshold_auto_duplicate=0.95,
            threshold_review=0.85,
            time_window_days=7
        )
        
        with patch('src.duplicate_detector.ReclamationRepository'):
            with patch('src.duplicate_detector.EmbeddingRepository'):
                with patch('src.duplicate_detector.DuplicationLogRepository'):
                    detector = DuplicateDetector(config=config, db_pool=MagicMock())
                    yield detector
    
    def test_determine_action_auto_duplicate(self, detector):
        """Test action determination for high similarity score."""
        from src.duplicate_detector import DuplicateAction
        
        # Score >= 0.95 should be AUTO_MARK_DUPLICATE
        assert detector.determine_action(0.95) == DuplicateAction.AUTO_MARK_DUPLICATE
        assert detector.determine_action(0.96) == DuplicateAction.AUTO_MARK_DUPLICATE
        assert detector.determine_action(1.0) == DuplicateAction.AUTO_MARK_DUPLICATE
    
    def test_determine_action_flag_for_review(self, detector):
        """Test action determination for medium similarity score."""
        from src.duplicate_detector import DuplicateAction
        
        # Score >= 0.85 and < 0.95 should be FLAG_FOR_REVIEW
        assert detector.determine_action(0.85) == DuplicateAction.FLAG_FOR_REVIEW
        assert detector.determine_action(0.90) == DuplicateAction.FLAG_FOR_REVIEW
        assert detector.determine_action(0.94) == DuplicateAction.FLAG_FOR_REVIEW
    
    def test_determine_action_no_action(self, detector):
        """Test action determination for low similarity score."""
        from src.duplicate_detector import DuplicateAction
        
        # Score < 0.85 should be NO_ACTION
        assert detector.determine_action(0.84) == DuplicateAction.NO_ACTION
        assert detector.determine_action(0.50) == DuplicateAction.NO_ACTION
        assert detector.determine_action(0.0) == DuplicateAction.NO_ACTION
    
    def test_threshold_boundary_95(self, detector):
        """Test exact boundary at 0.95 threshold."""
        from src.duplicate_detector import DuplicateAction
        
        # Exactly 0.95 should be AUTO_MARK_DUPLICATE
        assert detector.determine_action(0.95) == DuplicateAction.AUTO_MARK_DUPLICATE
        # Just below should be FLAG_FOR_REVIEW
        assert detector.determine_action(0.9499) == DuplicateAction.FLAG_FOR_REVIEW
    
    def test_threshold_boundary_85(self, detector):
        """Test exact boundary at 0.85 threshold."""
        from src.duplicate_detector import DuplicateAction
        
        # Exactly 0.85 should be FLAG_FOR_REVIEW
        assert detector.determine_action(0.85) == DuplicateAction.FLAG_FOR_REVIEW
        # Just below should be NO_ACTION
        assert detector.determine_action(0.8499) == DuplicateAction.NO_ACTION


class TestStatusForAction:
    """Tests for status determination based on action."""
    
    @pytest.fixture
    def detector(self):
        """Create a DuplicateDetector with mocked dependencies."""
        from src.duplicate_detector import DuplicateDetector
        from src.config import DuplicateDetectionConfig
        
        config = DuplicateDetectionConfig()
        
        with patch('src.duplicate_detector.ReclamationRepository'):
            with patch('src.duplicate_detector.EmbeddingRepository'):
                with patch('src.duplicate_detector.DuplicationLogRepository'):
                    detector = DuplicateDetector(config=config, db_pool=MagicMock())
                    yield detector
    
    def test_status_for_auto_duplicate(self, detector):
        """Test status for AUTO_MARK_DUPLICATE action."""
        from src.duplicate_detector import DuplicateAction, ReclamationStatus
        
        status = detector.get_status_for_action(DuplicateAction.AUTO_MARK_DUPLICATE)
        assert status == ReclamationStatus.DUPLICATE
    
    def test_status_for_flag_review(self, detector):
        """Test status for FLAG_FOR_REVIEW action."""
        from src.duplicate_detector import DuplicateAction, ReclamationStatus
        
        status = detector.get_status_for_action(DuplicateAction.FLAG_FOR_REVIEW)
        assert status == ReclamationStatus.POTENTIAL_DUPLICATE
    
    def test_status_for_no_action(self, detector):
        """Test status for NO_ACTION action."""
        from src.duplicate_detector import DuplicateAction, ReclamationStatus
        
        status = detector.get_status_for_action(DuplicateAction.NO_ACTION)
        assert status == ReclamationStatus.PENDING


class TestProcessReclamation:
    """Tests for the main process_reclamation method."""
    
    @pytest.fixture
    def mock_setup(self):
        """Set up all mocks for process_reclamation tests."""
        from src.duplicate_detector import DuplicateDetector
        from src.config import DuplicateDetectionConfig
        from src.database import Reclamation
        
        config = DuplicateDetectionConfig(
            threshold_auto_duplicate=0.95,
            threshold_review=0.85,
            time_window_days=7
        )
        
        mock_rec_repo = MagicMock()
        mock_emb_repo = MagicMock()
        mock_log_repo = MagicMock()
        
        with patch('src.duplicate_detector.ReclamationRepository', return_value=mock_rec_repo):
            with patch('src.duplicate_detector.EmbeddingRepository', return_value=mock_emb_repo):
                with patch('src.duplicate_detector.DuplicationLogRepository', return_value=mock_log_repo):
                    detector = DuplicateDetector(config=config, db_pool=MagicMock())
                    detector._reclamation_repo = mock_rec_repo
                    detector._embedding_repo = mock_emb_repo
                    detector._log_repo = mock_log_repo
                    
                    yield detector, mock_rec_repo, mock_emb_repo, mock_log_repo
    
    def test_process_not_found(self, mock_setup):
        """Test processing a non-existent reclamation."""
        from src.duplicate_detector import DuplicateAction
        
        detector, mock_rec_repo, _, _ = mock_setup
        mock_rec_repo.get_by_id.return_value = None
        
        result = detector.process_reclamation(999)
        
        assert result.reclamation_id == 999
        assert result.is_duplicate is False
        assert result.action == DuplicateAction.NO_ACTION
        assert "not found" in result.message
    
    def test_process_empty_text(self, mock_setup):
        """Test processing a reclamation with empty text after normalization."""
        from src.duplicate_detector import DuplicateAction
        from src.database import Reclamation
        
        detector, mock_rec_repo, _, _ = mock_setup
        mock_rec_repo.get_by_id.return_value = Reclamation(
            id=1,
            user_id=10,
            message_libre="   ",  # Will be empty after normalization
            created_at=datetime.now(),
            status="PENDING"
        )
        
        result = detector.process_reclamation(1)
        
        assert result.reclamation_id == 1
        assert result.is_duplicate is False
        assert result.action == DuplicateAction.NO_ACTION
        assert "empty" in result.message.lower()
    
    def test_process_no_matches(self, mock_setup):
        """Test processing when no similar reclamations found."""
        from src.duplicate_detector import DuplicateAction
        from src.database import Reclamation
        
        detector, mock_rec_repo, mock_emb_repo, _ = mock_setup
        
        mock_rec_repo.get_by_id.return_value = Reclamation(
            id=1,
            user_id=10,
            message_libre="Ma connexion internet ne fonctionne pas",
            created_at=datetime.now(),
            status="PENDING"
        )
        mock_emb_repo.find_similar.return_value = []
        
        with patch('src.duplicate_detector.get_embedding_model') as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode_to_list.return_value = [0.1] * 768
            mock_model.return_value = mock_instance
            
            result = detector.process_reclamation(1)
        
        assert result.reclamation_id == 1
        assert result.is_duplicate is False
        assert result.action == DuplicateAction.NO_ACTION
        assert result.matched_id is None
    
    def test_process_auto_duplicate(self, mock_setup):
        """Test processing when high similarity match found."""
        from src.duplicate_detector import DuplicateAction
        from src.database import Reclamation, SimilarityMatch
        
        detector, mock_rec_repo, mock_emb_repo, mock_log_repo = mock_setup
        
        mock_rec_repo.get_by_id.return_value = Reclamation(
            id=1,
            user_id=10,
            message_libre="Ma connexion internet ne fonctionne pas",
            created_at=datetime.now(),
            status="PENDING"
        )
        mock_emb_repo.find_similar.return_value = [
            SimilarityMatch(reclamation_id=2, score=0.96)
        ]
        
        with patch('src.duplicate_detector.get_embedding_model') as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode_to_list.return_value = [0.1] * 768
            mock_model.return_value = mock_instance
            
            result = detector.process_reclamation(1)
        
        assert result.reclamation_id == 1
        assert result.is_duplicate is True
        assert result.action == DuplicateAction.AUTO_MARK_DUPLICATE
        assert result.matched_id == 2
        assert result.similarity_score == 0.96
        
        # Verify status was updated
        mock_rec_repo.update_status.assert_called_once_with(1, "DUPLICATE")
        
        # Verify log was created
        mock_log_repo.create.assert_called_once()
    
    def test_process_flag_for_review(self, mock_setup):
        """Test processing when medium similarity match found."""
        from src.duplicate_detector import DuplicateAction
        from src.database import Reclamation, SimilarityMatch
        
        detector, mock_rec_repo, mock_emb_repo, mock_log_repo = mock_setup
        
        mock_rec_repo.get_by_id.return_value = Reclamation(
            id=1,
            user_id=10,
            message_libre="Ma connexion internet ne fonctionne pas",
            created_at=datetime.now(),
            status="PENDING"
        )
        mock_emb_repo.find_similar.return_value = [
            SimilarityMatch(reclamation_id=2, score=0.90)
        ]
        
        with patch('src.duplicate_detector.get_embedding_model') as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode_to_list.return_value = [0.1] * 768
            mock_model.return_value = mock_instance
            
            result = detector.process_reclamation(1)
        
        assert result.reclamation_id == 1
        assert result.is_duplicate is False  # Not auto-duplicate
        assert result.action == DuplicateAction.FLAG_FOR_REVIEW
        assert result.matched_id == 2
        assert result.similarity_score == 0.90
        
        # Verify status was updated to POTENTIAL_DUPLICATE
        mock_rec_repo.update_status.assert_called_once_with(1, "POTENTIAL_DUPLICATE")


class TestCheckSimilarity:
    """Tests for the check_similarity method."""
    
    @pytest.fixture
    def detector(self):
        """Create a DuplicateDetector with mocked dependencies."""
        from src.duplicate_detector import DuplicateDetector
        from src.config import DuplicateDetectionConfig
        
        config = DuplicateDetectionConfig()
        
        with patch('src.duplicate_detector.ReclamationRepository'):
            with patch('src.duplicate_detector.EmbeddingRepository'):
                with patch('src.duplicate_detector.DuplicationLogRepository'):
                    detector = DuplicateDetector(config=config, db_pool=MagicMock())
                    yield detector
    
    def test_check_similarity_high_score(self, detector):
        """Test check_similarity with high similarity texts."""
        from src.duplicate_detector import DuplicateAction
        
        with patch('src.duplicate_detector.get_embedding_model') as mock_model:
            mock_instance = MagicMock()
            mock_instance.similarity.return_value = 0.96
            mock_model.return_value = mock_instance
            
            score, action = detector.check_similarity(
                "Ma connexion internet ne fonctionne pas",
                "L'internet est coupé"
            )
        
        assert score == 0.96
        assert action == DuplicateAction.AUTO_MARK_DUPLICATE
    
    def test_check_similarity_empty_text(self, detector):
        """Test check_similarity with empty text."""
        from src.duplicate_detector import DuplicateAction
        
        score, action = detector.check_similarity("", "Some text")
        
        assert score == 0.0
        assert action == DuplicateAction.NO_ACTION


class TestBatchProcess:
    """Tests for batch processing."""
    
    def test_batch_process_success(self):
        """Test batch processing multiple reclamations."""
        from src.duplicate_detector import DuplicateDetector, DuplicateAction
        from src.config import DuplicateDetectionConfig
        
        config = DuplicateDetectionConfig()
        
        with patch('src.duplicate_detector.ReclamationRepository'):
            with patch('src.duplicate_detector.EmbeddingRepository'):
                with patch('src.duplicate_detector.DuplicationLogRepository'):
                    detector = DuplicateDetector(config=config, db_pool=MagicMock())
                    
                    # Mock process_reclamation to return results
                    detector._reclamation_repo.get_by_id.return_value = None
                    
                    results = detector.batch_process([1, 2, 3])
        
        assert len(results) == 3
        assert all(r.action == DuplicateAction.NO_ACTION for r in results)


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_get_duplicate_detector(self):
        """Test get_duplicate_detector factory function."""
        from src.duplicate_detector import get_duplicate_detector, DuplicateDetector
        
        with patch('src.duplicate_detector.ReclamationRepository'):
            with patch('src.duplicate_detector.EmbeddingRepository'):
                with patch('src.duplicate_detector.DuplicationLogRepository'):
                    detector = get_duplicate_detector(db_pool=MagicMock())
        
        assert isinstance(detector, DuplicateDetector)


