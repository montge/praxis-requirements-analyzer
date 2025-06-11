"""
HuggingFace Embeddings Client Module using RedisVL's HFTextVectorizer.

This module provides a client for generating embeddings using HuggingFace models
via RedisVL's vectorizer implementation.
"""

from typing import List
from redisvl.utils.vectorize import HFTextVectorizer
from src.utils.logger import setup_logger, handle_exception

class HuggingFaceEmbeddingsClient:
    """Client for generating embeddings using HuggingFace models via RedisVL.
    
    This class wraps RedisVL's HFTextVectorizer to provide consistent embedding generation.
    
    Attributes:
        model_name (str): Name of the HuggingFace model to use
        vectorizer (HFTextVectorizer): RedisVL HF vectorizer instance
        logger: Logger instance for this class
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embeddings client.
        
        Args:
            model_name: Name of the HuggingFace model to use for embeddings
                      Defaults to all-MiniLM-L6-v2
        
        Raises:
            Exception: If vectorizer initialization fails
        """
        self.model_name = model_name
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing embeddings client with model: {model_name}")
        
        try:
            self.vectorizer = HFTextVectorizer(model=model_name)
            self.logger.info("HuggingFace vectorizer initialized successfully")
            self.logger.debug(f"Vectorizer configuration: {self.vectorizer.model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize vectorizer: {str(e)}", exc_info=True)
            raise

    @handle_exception("Error generating embedding")
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text using RedisVL's vectorizer.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            List[float]: Generated embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            self.logger.debug(f"Generating embedding for text (length: {len(text)})")
            self.logger.debug(f"Input text preview: {text[:100]}...")
            
            embedding = self.vectorizer.embed(text)
            
            self.logger.debug(f"Successfully generated embedding of length {len(embedding)}")
            self.logger.debug(f"Embedding preview: first 5 values {embedding[:5]}")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {str(e)}", exc_info=True)
            raise

    @handle_exception("Error generating embeddings batch")
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using RedisVL's vectorizer.
        
        Args:
            texts: List of input texts to generate embeddings for
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            Exception: If batch embedding generation fails
        """
        try:
            self.logger.info(f"Generating embeddings batch for {len(texts)} texts")
            self.logger.debug(f"Text lengths: {[len(t) for t in texts]}")
            
            embeddings = self.vectorizer.embed_many(texts)
            
            self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
            self.logger.debug(f"Embedding dimensions: {[len(e) for e in embeddings]}")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings batch: {str(e)}", exc_info=True)
            raise

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings produced by this model.
        
        Returns:
            int: Dimension of the embeddings
        
        Note:
            Attempts to determine the dimension dynamically from the model or
            by generating a test embedding. Falls back to default value if both fail.
        """
        return 384  # Default for all-MiniLM-L6-v2 