"""
LaBSE Embedding module for cross-lingual text embeddings.

Uses Google's Language-agnostic BERT Sentence Embedding (LaBSE) model
to generate 768-dimensional vectors that map semantically similar
sentences across 109 languages to nearby points in vector space.
"""

import logging
from typing import Optional, Union
import numpy as np

from src.config import get_config, MLConfig


logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Singleton wrapper for the LaBSE sentence embedding model.
    
    This class provides a thread-safe singleton pattern for loading
    and using the LaBSE model, ensuring the model is only loaded once
    per process for efficiency.
    
    Attributes:
        model: The underlying SentenceTransformer model instance.
        model_name: Name of the model being used.
        dimension: Output embedding dimension (768 for LaBSE).
    """
    
    _instance: Optional["EmbeddingModel"] = None
    _initialized: bool = False
    
    def __new__(cls, model_name: Optional[str] = None) -> "EmbeddingModel":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Optional model name override. If not provided,
                uses the configured model name from environment.
        """
        # Skip if already initialized
        if EmbeddingModel._initialized:
            return
        
        config = get_config().ml if model_name is None else MLConfig(model_name=model_name)
        self.model_name = config.model_name
        self.dimension = config.embedding_dimension
        self._model = None
        self.max_seq_tokens = config.max_seq_tokens
        EmbeddingModel._initialized = True
        logger.info(f"EmbeddingModel initialized with model: {self.model_name}")
    
    @property
    def model(self):
        """Lazy-load the model on first access."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self) -> None:
        """Load the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")
            
        except ImportError as e:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise ImportError(
                "sentence-transformers library required. "
                "Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise



    def encode(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for one or more texts.
        
        Args:
            texts: Single text string or list of text strings.
            batch_size: Batch size for encoding multiple texts.
            show_progress: Whether to show a progress bar.
            normalize: Whether to L2-normalize the embeddings.
        
        Returns:
            numpy array of shape (n_texts, dimension) or (dimension,)
            for single text input.
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        token_numbers = [self.count_tokens(text) for text in texts]
        logger.debug(f"Token counts for inputs: {token_numbers}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize
        )
        
        if single_input:
            return embeddings[0]
        
        return embeddings
    
    def encode_to_list(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> Union[list[float], list[list[float]]]:
        """
        Generate embeddings and return as Python lists.
        
        Useful for database storage where lists are preferred over numpy arrays.
        
        Args:
            texts: Single text string or list of text strings.
            batch_size: Batch size for encoding multiple texts.
            normalize: Whether to L2-normalize the embeddings.
        
        Returns:
            List of floats for single input, or list of lists for batch input.
        """
        embeddings = self.encode(texts, batch_size=batch_size, normalize=normalize)
        
        if isinstance(texts, str):
            return embeddings.tolist()
        
        return [emb.tolist() for emb in embeddings]
    
    def similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
        
        Returns:
            Cosine similarity score between 0 and 1.
        """
        embeddings = self.encode([text1, text2], normalize=True)
        # Cosine similarity of normalized vectors is just dot product
        return float(np.dot(embeddings[0], embeddings[1]))
    
    def similarity_matrix(
        self,
        texts: list[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a list of texts.
        
        Args:
            texts: List of texts.
            batch_size: Batch size for encoding.
        
        Returns:
            Square numpy array of shape (n_texts, n_texts) with
            cosine similarity scores.
        """
        embeddings = self.encode(texts, batch_size=batch_size, normalize=True)
        # For normalized vectors, cosine similarity = dot product
        return np.dot(embeddings, embeddings.T)
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.
        
        Useful for testing or when model configuration changes.
        """
        cls._instance = None
        cls._initialized = False
        logger.info("EmbeddingModel singleton reset")


def get_embedding_model(model_name: Optional[str] = None) -> EmbeddingModel:
    """
    Get the singleton embedding model instance.
    
    Args:
        model_name: Optional model name override (only used on first call).
    
    Returns:
        The singleton EmbeddingModel instance.
    """
    return EmbeddingModel(model_name)


def encode_text(
    text: str,
    normalize: bool = True
) -> list[float]:
    """
    Convenience function to encode a single text.
    
    Args:
        text: Text to encode.
        normalize: Whether to L2-normalize the embedding.
    
    Returns:
        List of floats representing the embedding.
    """
    model = get_embedding_model()
    return model.encode_to_list(text, normalize=normalize)


def encode_texts(
    texts: list[str],
    batch_size: int = 32,
    normalize: bool = True
) -> list[list[float]]:
    """
    Convenience function to encode multiple texts.
    
    Args:
        texts: List of texts to encode.
        batch_size: Batch size for encoding.
        normalize: Whether to L2-normalize the embeddings.
    
    Returns:
        List of embedding lists.
    """
    model = get_embedding_model()
    return model.encode_to_list(texts, batch_size=batch_size, normalize=normalize)


def compute_similarity(text1: str, text2: str) -> float:
    """
    Convenience function to compute similarity between two texts.
    
    Args:
        text1: First text.
        text2: Second text.
    
    Returns:
        Cosine similarity score between 0 and 1.
    """
    model = get_embedding_model()
    return model.similarity(text1, text2)


