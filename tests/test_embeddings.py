"""
Unit tests for the LaBSE embedding module.
Tests the singleton pattern, embedding generation, and similarity computation.

Note: These tests use mocked models for unit testing.
Integration tests with real models are in test_integration.py.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestEmbeddingModelSingleton:
    """Tests for the EmbeddingModel singleton pattern."""
    
    def test_singleton_returns_same_instance(self):
        """Test that EmbeddingModel returns the same instance."""
        from src.embeddings import EmbeddingModel
        
        # Reset first to ensure clean state
        EmbeddingModel.reset()
        
        with patch('src.embeddings.get_config') as mock_config:
            mock_ml_config = MagicMock()
            mock_ml_config.model_name = "test-model"
            mock_ml_config.embedding_dimension = 768
            mock_config.return_value.ml = mock_ml_config
            
            model1 = EmbeddingModel()
            model2 = EmbeddingModel()
            
            assert model1 is model2
        
        EmbeddingModel.reset()
    
    def test_reset_creates_new_instance(self):
        """Test that reset allows creating a new instance."""
        from src.embeddings import EmbeddingModel
        
        EmbeddingModel.reset()
        
        with patch('src.embeddings.get_config') as mock_config:
            mock_ml_config = MagicMock()
            mock_ml_config.model_name = "test-model"
            mock_ml_config.embedding_dimension = 768
            mock_config.return_value.ml = mock_ml_config
            
            model1 = EmbeddingModel()
            EmbeddingModel.reset()
            model2 = EmbeddingModel()
            
            assert model1 is not model2
        
        EmbeddingModel.reset()
    
    def test_model_name_from_config(self):
        """Test that model name is loaded from config."""
        from src.embeddings import EmbeddingModel
        
        EmbeddingModel.reset()
        
        with patch('src.embeddings.get_config') as mock_config:
            mock_ml_config = MagicMock()
            mock_ml_config.model_name = "sentence-transformers/LaBSE"
            mock_ml_config.embedding_dimension = 768
            mock_config.return_value.ml = mock_ml_config
            
            model = EmbeddingModel()
            assert model.model_name == "sentence-transformers/LaBSE"
            assert model.dimension == 768
        
        EmbeddingModel.reset()


class TestEmbeddingModelEncode:
    """Tests for the encode functionality with mocked model."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock EmbeddingModel with mocked SentenceTransformer."""
        from src.embeddings import EmbeddingModel
        
        EmbeddingModel.reset()
        
        with patch('src.embeddings.get_config') as mock_config:
            mock_ml_config = MagicMock()
            mock_ml_config.model_name = "test-model"
            mock_ml_config.embedding_dimension = 768
            mock_config.return_value.ml = mock_ml_config
            
            model = EmbeddingModel()
            
            # Mock the internal model
            mock_st = MagicMock()
            mock_st.encode = MagicMock(return_value=np.random.randn(1, 768).astype(np.float32))
            model._model = mock_st
            
            yield model
        
        EmbeddingModel.reset()
    
    def test_encode_single_text(self, mock_model):
        """Test encoding a single text."""
        # Mock to return single vector
        mock_model._model.encode = MagicMock(
            return_value=np.random.randn(1, 768).astype(np.float32)
        )
        
        result = mock_model.encode("Hello world")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (768,)
        mock_model._model.encode.assert_called_once()
    
    def test_encode_multiple_texts(self, mock_model):
        """Test encoding multiple texts."""
        # Mock to return multiple vectors
        mock_model._model.encode = MagicMock(
            return_value=np.random.randn(3, 768).astype(np.float32)
        )
        
        texts = ["Hello", "World", "Test"]
        result = mock_model.encode(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 768)
    
    def test_encode_to_list_single(self, mock_model):
        """Test encode_to_list returns Python list for single text."""
        mock_model._model.encode = MagicMock(
            return_value=np.random.randn(1, 768).astype(np.float32)
        )
        
        result = mock_model.encode_to_list("Hello world")
        
        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)
    
    def test_encode_to_list_multiple(self, mock_model):
        """Test encode_to_list returns list of lists for multiple texts."""
        mock_model._model.encode = MagicMock(
            return_value=np.random.randn(3, 768).astype(np.float32)
        )
        
        texts = ["Hello", "World", "Test"]
        result = mock_model.encode_to_list(texts)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(x, list) for x in result)
        assert all(len(x) == 768 for x in result)


class TestSimilarityComputation:
    """Tests for similarity computation."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock EmbeddingModel for similarity tests."""
        from src.embeddings import EmbeddingModel
        
        EmbeddingModel.reset()
        
        with patch('src.embeddings.get_config') as mock_config:
            mock_ml_config = MagicMock()
            mock_ml_config.model_name = "test-model"
            mock_ml_config.embedding_dimension = 768
            mock_config.return_value.ml = mock_ml_config
            
            model = EmbeddingModel()
            
            # Mock the internal model
            mock_st = MagicMock()
            model._model = mock_st
            
            yield model
        
        EmbeddingModel.reset()
    
    def test_similarity_identical_vectors(self, mock_model):
        """Test similarity of identical texts returns ~1.0."""
        # Create identical normalized vectors
        vec = np.random.randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        mock_model._model.encode = MagicMock(
            return_value=np.vstack([vec, vec])
        )
        
        result = mock_model.similarity("test", "test")
        
        assert isinstance(result, float)
        assert result >= 0.99  # Should be very close to 1
    
    def test_similarity_orthogonal_vectors(self, mock_model):
        """Test similarity of orthogonal vectors returns ~0."""
        # Create orthogonal normalized vectors
        vec1 = np.zeros(768, dtype=np.float32)
        vec1[0] = 1.0
        vec2 = np.zeros(768, dtype=np.float32)
        vec2[1] = 1.0
        
        mock_model._model.encode = MagicMock(
            return_value=np.vstack([vec1, vec2])
        )
        
        result = mock_model.similarity("text1", "text2")
        
        assert isinstance(result, float)
        assert abs(result) < 0.01  # Should be very close to 0
    
    def test_similarity_range(self, mock_model):
        """Test similarity is in valid range [-1, 1]."""
        # Random normalized vectors
        vec1 = np.random.randn(768).astype(np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = np.random.randn(768).astype(np.float32)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        mock_model._model.encode = MagicMock(
            return_value=np.vstack([vec1, vec2])
        )
        
        result = mock_model.similarity("text1", "text2")
        
        assert -1.0 <= result <= 1.0


class TestSimilarityMatrix:
    """Tests for similarity matrix computation."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock EmbeddingModel for matrix tests."""
        from src.embeddings import EmbeddingModel
        
        EmbeddingModel.reset()
        
        with patch('src.embeddings.get_config') as mock_config:
            mock_ml_config = MagicMock()
            mock_ml_config.model_name = "test-model"
            mock_ml_config.embedding_dimension = 768
            mock_config.return_value.ml = mock_ml_config
            
            model = EmbeddingModel()
            model._model = MagicMock()
            
            yield model
        
        EmbeddingModel.reset()
    
    def test_similarity_matrix_shape(self, mock_model):
        """Test similarity matrix has correct shape."""
        # Create 5 random normalized vectors
        vecs = np.random.randn(5, 768).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        
        mock_model._model.encode = MagicMock(return_value=vecs)
        
        texts = ["a", "b", "c", "d", "e"]
        result = mock_model.similarity_matrix(texts)
        
        assert result.shape == (5, 5)
    
    def test_similarity_matrix_diagonal(self, mock_model):
        """Test similarity matrix diagonal is ~1 (self-similarity)."""
        # Create random normalized vectors
        vecs = np.random.randn(3, 768).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        
        mock_model._model.encode = MagicMock(return_value=vecs)
        
        texts = ["a", "b", "c"]
        result = mock_model.similarity_matrix(texts)
        
        # Diagonal should be ~1
        for i in range(3):
            assert result[i, i] >= 0.99
    
    def test_similarity_matrix_symmetric(self, mock_model):
        """Test similarity matrix is symmetric."""
        # Create random normalized vectors
        vecs = np.random.randn(4, 768).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        
        mock_model._model.encode = MagicMock(return_value=vecs)
        
        texts = ["a", "b", "c", "d"]
        result = mock_model.similarity_matrix(texts)
        
        # Matrix should be symmetric
        assert np.allclose(result, result.T)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_embedding_model_returns_singleton(self):
        """Test get_embedding_model returns singleton."""
        from src.embeddings import get_embedding_model, EmbeddingModel
        
        EmbeddingModel.reset()
        
        with patch('src.embeddings.get_config') as mock_config:
            mock_ml_config = MagicMock()
            mock_ml_config.model_name = "test-model"
            mock_ml_config.embedding_dimension = 768
            mock_config.return_value.ml = mock_ml_config
            
            model1 = get_embedding_model()
            model2 = get_embedding_model()
            
            assert model1 is model2
        
        EmbeddingModel.reset()


class TestVectorDimensions:
    """Tests for vector dimension validation."""
    
    def test_labse_dimension_is_768(self, ml_config):
        """Test that LaBSE embedding dimension is 768."""
        assert ml_config.embedding_dimension == 768
    
    def test_embedding_output_dimension(self):
        """Test that mocked embeddings have correct dimension."""
        from src.embeddings import EmbeddingModel
        
        EmbeddingModel.reset()
        
        with patch('src.embeddings.get_config') as mock_config:
            mock_ml_config = MagicMock()
            mock_ml_config.model_name = "test-model"
            mock_ml_config.embedding_dimension = 768
            mock_config.return_value.ml = mock_ml_config
            
            model = EmbeddingModel()
            
            # Mock with correct dimension
            mock_st = MagicMock()
            mock_st.encode = MagicMock(
                return_value=np.random.randn(1, 768).astype(np.float32)
            )
            model._model = mock_st
            
            result = model.encode("test")
            assert len(result) == 768
        
        EmbeddingModel.reset()


