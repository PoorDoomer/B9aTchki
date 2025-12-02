"""
Integration tests for the de-duplication pipeline.

These tests verify the end-to-end functionality of the pipeline
with mock data and timeout assertions.

Note: Some tests require actual infrastructure (PostgreSQL, RabbitMQ)
and are marked with appropriate markers.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch


# Mark for tests requiring full infrastructure
# pytest-timeout is auto-loaded via pip install


class TestPreprocessingToEmbedding:
    """Integration tests for preprocessing to embedding pipeline."""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        from src.embeddings import EmbeddingModel
        
        EmbeddingModel.reset()
        
        with patch('src.embeddings.get_config') as mock_config:
            mock_ml_config = MagicMock()
            mock_ml_config.model_name = "test-model"
            mock_ml_config.embedding_dimension = 768
            mock_config.return_value.ml = mock_ml_config
            
            model = EmbeddingModel()
            
            # Mock the internal model with realistic behavior
            mock_st = MagicMock()
            
            def mock_encode(texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                # Return normalized random vectors
                vecs = np.random.randn(len(texts), 768).astype(np.float32)
                vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs
            
            mock_st.encode = mock_encode
            model._model = mock_st
            
            yield model
        
        EmbeddingModel.reset()
    
    @pytest.mark.timeout(5)
    def test_text_to_embedding_french(self, mock_embedding_model):
        """Test processing French text to embedding."""
        from src.preprocessing import normalize_text
        
        text = "Ma connexion internet ne fonctionne pas depuis hier"
        normalized = normalize_text(text)
        
        assert normalized == "ma connexion internet ne fonctionne pas depuis hier"
        
        embedding = mock_embedding_model.encode(normalized)
        assert len(embedding) == 768
    
    @pytest.mark.timeout(5)
    def test_text_to_embedding_arabic(self, mock_embedding_model):
        """Test processing Arabic text to embedding."""
        from src.preprocessing import normalize_text
        
        text = "الإنترنت مقطوع منذ أمس"
        normalized = normalize_text(text)
        
        assert len(normalized) > 0
        
        embedding = mock_embedding_model.encode(normalized)
        assert len(embedding) == 768
    
    @pytest.mark.timeout(5)
    def test_batch_preprocessing_and_embedding(self, mock_embedding_model):
        """Test batch processing multiple texts."""
        from src.preprocessing import preprocess_batch
        
        texts = [
            "Ma connexion internet ne fonctionne pas",
            "L'internet est coupé",
            "الإنترنت مقطوع",
        ]
        
        normalized = preprocess_batch(texts)
        assert len(normalized) == 3
        
        embeddings = mock_embedding_model.encode(normalized)
        assert embeddings.shape == (3, 768)


class TestSimilarityDetection:
    """Integration tests for similarity detection logic."""
    
    @pytest.fixture
    def detector_with_mock_deps(self):
        """Create a detector with mocked dependencies."""
        from src.duplicate_detector import DuplicateDetector
        from src.config import DuplicateDetectionConfig
        
        config = DuplicateDetectionConfig(
            threshold_auto_duplicate=0.95,
            threshold_review=0.85,
            time_window_days=7
        )
        
        with patch('src.duplicate_detector.ReclamationRepository') as mock_rec:
            with patch('src.duplicate_detector.EmbeddingRepository') as mock_emb:
                with patch('src.duplicate_detector.DuplicationLogRepository') as mock_log:
                    detector = DuplicateDetector(config=config, db_pool=MagicMock())
                    detector._reclamation_repo = mock_rec.return_value
                    detector._embedding_repo = mock_emb.return_value
                    detector._log_repo = mock_log.return_value
                    
                    yield detector
    
    @pytest.mark.timeout(5)
    def test_threshold_boundaries(self, detector_with_mock_deps):
        """Test that thresholds are correctly applied."""
        from src.duplicate_detector import DuplicateAction
        
        detector = detector_with_mock_deps
        
        # Test all threshold ranges
        test_cases = [
            (0.99, DuplicateAction.AUTO_MARK_DUPLICATE),
            (0.96, DuplicateAction.AUTO_MARK_DUPLICATE),
            (0.95, DuplicateAction.AUTO_MARK_DUPLICATE),
            (0.94, DuplicateAction.FLAG_FOR_REVIEW),
            (0.90, DuplicateAction.FLAG_FOR_REVIEW),
            (0.85, DuplicateAction.FLAG_FOR_REVIEW),
            (0.84, DuplicateAction.NO_ACTION),
            (0.50, DuplicateAction.NO_ACTION),
        ]
        
        for score, expected_action in test_cases:
            action = detector.determine_action(score)
            assert action == expected_action, f"Score {score} should be {expected_action}, got {action}"
    
    @pytest.mark.timeout(10)
    def test_full_detection_pipeline_no_match(self, detector_with_mock_deps):
        """Test full pipeline when no matches found."""
        from src.duplicate_detector import DuplicateAction
        from src.database import Reclamation
        
        detector = detector_with_mock_deps
        
        # Setup mocks
        detector._reclamation_repo.get_by_id.return_value = Reclamation(
            id=1,
            user_id=10,
            message_libre="Test message",
            created_at=datetime.now(),
            status="PENDING"
        )
        detector._embedding_repo.find_similar.return_value = []
        
        with patch('src.duplicate_detector.get_embedding_model') as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode_to_list.return_value = [0.1] * 768
            mock_model.return_value = mock_instance
            
            result = detector.process_reclamation(1)
        
        assert result.is_duplicate is False
        assert result.action == DuplicateAction.NO_ACTION
        assert result.matched_id is None
    
    @pytest.mark.timeout(10)
    def test_full_detection_pipeline_auto_duplicate(self, detector_with_mock_deps):
        """Test full pipeline when high-confidence match found."""
        from src.duplicate_detector import DuplicateAction, ReclamationStatus
        from src.database import Reclamation, SimilarityMatch
        
        detector = detector_with_mock_deps
        
        # Setup mocks
        detector._reclamation_repo.get_by_id.return_value = Reclamation(
            id=1,
            user_id=10,
            message_libre="Ma connexion internet ne fonctionne pas",
            created_at=datetime.now(),
            status="PENDING"
        )
        detector._embedding_repo.find_similar.return_value = [
            SimilarityMatch(reclamation_id=2, score=0.96)
        ]
        
        with patch('src.duplicate_detector.get_embedding_model') as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode_to_list.return_value = [0.1] * 768
            mock_model.return_value = mock_instance
            
            result = detector.process_reclamation(1)
        
        assert result.is_duplicate is True
        assert result.action == DuplicateAction.AUTO_MARK_DUPLICATE
        assert result.matched_id == 2
        assert result.similarity_score == 0.96
        
        # Verify status was updated
        detector._reclamation_repo.update_status.assert_called_with(1, "DUPLICATE")
        
        # Verify log was created
        detector._log_repo.create.assert_called_once()


class TestCrossLingualMatching:
    """Tests for cross-lingual (French-Arabic) matching."""
    
    @pytest.mark.timeout(5)
    def test_french_arabic_normalization(self):
        """Test that both French and Arabic texts are properly normalized."""
        from src.preprocessing import normalize_text, detect_primary_script
        
        french = "Ma connexion INTERNET ne fonctionne pas"
        arabic = "الإنترنت مقطوع منذ أمس"
        
        french_normalized = normalize_text(french)
        arabic_normalized = normalize_text(arabic)
        
        # French should be lowercased
        assert french_normalized == french_normalized.lower()
        
        # Arabic should have Alif normalized
        assert "ا" in arabic_normalized  # Normalized Alif
        
        # Script detection
        assert detect_primary_script(french) == "latin"
        assert detect_primary_script(arabic) == "arabic"
    
    @pytest.mark.timeout(5)
    def test_mixed_language_text(self):
        """Test handling of mixed language text."""
        from src.preprocessing import normalize_text, detect_primary_script
        
        mixed = "Internet مقطوع problem"
        normalized = normalize_text(mixed)
        
        assert len(normalized) > 0
        script = detect_primary_script(mixed)
        assert script in ["arabic", "latin", "mixed"]


class TestPipelineTimeout:
    """Tests with explicit timeout to ensure pipeline performance."""
    
    @pytest.mark.timeout(2)
    def test_preprocessing_performance(self):
        """Test that preprocessing is fast enough."""
        from src.preprocessing import normalize_text
        
        texts = [
            "Ma connexion internet ne fonctionne pas" * 10,
            "الإنترنت مقطوع منذ أمس" * 10,
        ] * 50
        
        for text in texts:
            result = normalize_text(text)
            assert len(result) > 0
    
    @pytest.mark.timeout(2)
    def test_threshold_logic_performance(self):
        """Test that threshold logic is fast."""
        from src.duplicate_detector import DuplicateDetector
        from src.config import DuplicateDetectionConfig
        
        config = DuplicateDetectionConfig()
        
        with patch('src.duplicate_detector.ReclamationRepository'):
            with patch('src.duplicate_detector.EmbeddingRepository'):
                with patch('src.duplicate_detector.DuplicationLogRepository'):
                    detector = DuplicateDetector(config=config, db_pool=MagicMock())
        
        # Test many threshold evaluations
        scores = np.random.uniform(0, 1, 10000)
        for score in scores:
            action = detector.determine_action(score)
            assert action is not None


class TestMockDataGeneration:
    """Tests for mock data generator."""
    
    @pytest.mark.timeout(5)
    def test_french_templates_exist(self):
        """Test that French templates are defined."""
        from scripts.generate_mock_data import FRENCH_TEMPLATES
        
        assert len(FRENCH_TEMPLATES) > 0
        
        # Check structure
        for text, category in FRENCH_TEMPLATES:
            assert isinstance(text, str)
            assert isinstance(category, str)
            assert len(text) > 0
    
    @pytest.mark.timeout(5)
    def test_arabic_templates_exist(self):
        """Test that Arabic templates are defined."""
        from scripts.generate_mock_data import ARABIC_TEMPLATES
        
        assert len(ARABIC_TEMPLATES) > 0
        
        # Check structure
        for text, category in ARABIC_TEMPLATES:
            assert isinstance(text, str)
            assert isinstance(category, str)
            assert len(text) > 0
    
    @pytest.mark.timeout(5)
    def test_categories_match(self):
        """Test that French and Arabic have matching categories."""
        from scripts.generate_mock_data import FRENCH_TEMPLATES, ARABIC_TEMPLATES
        
        french_categories = set(cat for _, cat in FRENCH_TEMPLATES)
        arabic_categories = set(cat for _, cat in ARABIC_TEMPLATES)
        
        # Categories should overlap
        common = french_categories & arabic_categories
        assert len(common) > 0, "French and Arabic should share some categories"


class TestEndToEndScenarios:
    """End-to-end scenario tests."""
    
    @pytest.fixture
    def full_mock_setup(self):
        """Setup complete mocked environment."""
        from src.duplicate_detector import DuplicateDetector
        from src.config import DuplicateDetectionConfig
        from src.database import Reclamation, SimilarityMatch
        from src.embeddings import EmbeddingModel
        
        EmbeddingModel.reset()
        
        config = DuplicateDetectionConfig()
        
        # Mock all components
        mock_rec_repo = MagicMock()
        mock_emb_repo = MagicMock()
        mock_log_repo = MagicMock()
        
        with patch('src.duplicate_detector.ReclamationRepository', return_value=mock_rec_repo):
            with patch('src.duplicate_detector.EmbeddingRepository', return_value=mock_emb_repo):
                with patch('src.duplicate_detector.DuplicationLogRepository', return_value=mock_log_repo):
                    with patch('src.duplicate_detector.get_embedding_model') as mock_model:
                        mock_instance = MagicMock()
                        mock_instance.encode_to_list.return_value = [0.1] * 768
                        mock_model.return_value = mock_instance
                        
                        detector = DuplicateDetector(config=config, db_pool=MagicMock())
                        detector._reclamation_repo = mock_rec_repo
                        detector._embedding_repo = mock_emb_repo
                        detector._log_repo = mock_log_repo
                        
                        yield {
                            'detector': detector,
                            'rec_repo': mock_rec_repo,
                            'emb_repo': mock_emb_repo,
                            'log_repo': mock_log_repo,
                            'model': mock_instance
                        }
        
        EmbeddingModel.reset()
    
    @pytest.mark.timeout(10)
    def test_scenario_exact_duplicate_french(self, full_mock_setup):
        """Scenario: User submits exact duplicate in French."""
        from src.duplicate_detector import DuplicateAction
        from src.database import Reclamation, SimilarityMatch
        
        setup = full_mock_setup
        
        setup['rec_repo'].get_by_id.return_value = Reclamation(
            id=2,
            user_id=10,
            message_libre="Ma connexion internet ne fonctionne pas",
            created_at=datetime.now(),
            status="PENDING"
        )
        
        # Simulate exact duplicate found (score = 0.99)
        setup['emb_repo'].find_similar.return_value = [
            SimilarityMatch(
                reclamation_id=1,
                score=0.99,
                message_libre="Ma connexion internet ne fonctionne pas",
                status="PROCESSED"
            )
        ]
        
        result = setup['detector'].process_reclamation(2)
        
        assert result.is_duplicate is True
        assert result.action == DuplicateAction.AUTO_MARK_DUPLICATE
        assert result.similarity_score >= 0.95
    
    @pytest.mark.timeout(10)
    def test_scenario_semantic_duplicate_cross_lingual(self, full_mock_setup):
        """Scenario: User submits semantically similar in Arabic to French original."""
        from src.duplicate_detector import DuplicateAction
        from src.database import Reclamation, SimilarityMatch
        
        setup = full_mock_setup
        
        setup['rec_repo'].get_by_id.return_value = Reclamation(
            id=2,
            user_id=10,
            message_libre="الإنترنت مقطوع منذ أمس",  # Arabic: Internet is cut since yesterday
            created_at=datetime.now(),
            status="PENDING"
        )
        
        # Simulate cross-lingual match found (score = 0.92)
        setup['emb_repo'].find_similar.return_value = [
            SimilarityMatch(
                reclamation_id=1,
                score=0.92,
                message_libre="Ma connexion internet ne fonctionne pas",  # French original
                status="PENDING"
            )
        ]
        
        result = setup['detector'].process_reclamation(2)
        
        # Should flag for review (not auto-duplicate)
        assert result.is_duplicate is False
        assert result.action == DuplicateAction.FLAG_FOR_REVIEW
        assert 0.85 <= result.similarity_score < 0.95
    
    @pytest.mark.timeout(10)
    def test_scenario_unique_reclamation(self, full_mock_setup):
        """Scenario: User submits unique reclamation."""
        from src.duplicate_detector import DuplicateAction
        from src.database import Reclamation
        
        setup = full_mock_setup
        
        setup['rec_repo'].get_by_id.return_value = Reclamation(
            id=1,
            user_id=10,
            message_libre="Je veux changer mon forfait mobile",
            created_at=datetime.now(),
            status="PENDING"
        )
        
        # No similar reclamations found
        setup['emb_repo'].find_similar.return_value = []
        
        result = setup['detector'].process_reclamation(1)
        
        assert result.is_duplicate is False
        assert result.action == DuplicateAction.NO_ACTION
        assert result.matched_id is None
    
    @pytest.mark.timeout(10)
    def test_scenario_different_user_no_match(self, full_mock_setup):
        """Scenario: Same text but different user - should not match due to grouping."""
        from src.duplicate_detector import DuplicateAction
        from src.database import Reclamation
        
        setup = full_mock_setup
        
        # User 20's reclamation (different from user 10)
        setup['rec_repo'].get_by_id.return_value = Reclamation(
            id=5,
            user_id=20,  # Different user
            message_libre="Ma connexion internet ne fonctionne pas",
            created_at=datetime.now(),
            status="PENDING"
        )
        
        # No matches (because search is filtered by user_id)
        setup['emb_repo'].find_similar.return_value = []
        
        result = setup['detector'].process_reclamation(5)
        
        # Should be unique (no duplicates for this user)
        assert result.is_duplicate is False
        assert result.action == DuplicateAction.NO_ACTION

