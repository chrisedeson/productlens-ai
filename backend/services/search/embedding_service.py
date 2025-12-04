"""
Embedding Service for ProductLens AI.

Handles generating text embeddings using OpenAI's embedding models.
Embeddings are vector representations of text used for semantic similarity search.
"""

from typing import List, Optional
import logging

from openai import OpenAI, OpenAIError

from core.base_service import BaseService, RetryMixin
from core.exceptions import ExternalAPIError, ValidationError

logging.basicConfig(level=logging.INFO)


class EmbeddingService(BaseService, RetryMixin):
    """
    Service for generating text embeddings using OpenAI.
    
    Uses the text-embedding-3-small model which produces 1536-dimensional vectors.
    These embeddings are used for semantic similarity search in the vector database.
    
    Attributes:
        model: The OpenAI embedding model name.
        dimensions: Output dimensions of the embedding vectors.
    
    Example:
        >>> service = EmbeddingService(api_key="sk-...")
        >>> embedding = service.embed_query("running shoes")
        >>> len(embedding)
        1536
    """
    
    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSIONS = 1536
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS
    ):
        """
        Initialize the EmbeddingService.
        
        Args:
            api_key: OpenAI API key.
            model: Embedding model to use. Defaults to text-embedding-3-small.
            dimensions: Expected output dimensions.
            
        Raises:
            ValidationError: If api_key is empty.
        """
        super().__init__("EmbeddingService")
        
        if not api_key or not api_key.strip():
            raise ValidationError("api_key", "API key cannot be empty")
        
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        
        self._mark_initialized()
    
    def embed_query(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding for a single text query.
        
        This is the primary method for generating embeddings for user queries
        in the search functionality.
        
        Args:
            text: Input text to embed.
            
        Returns:
            List of floats representing the embedding vector, or None on error.
            
        Example:
            >>> embedding = service.embed_query("blue sneakers for running")
            >>> print(f"Generated {len(embedding)}-dimensional embedding")
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return None
        
        try:
            # Clean the text
            cleaned_text = self._preprocess_text(text)
            
            response = self._with_retry(
                lambda: self._client.embeddings.create(
                    input=cleaned_text,
                    model=self.model
                ),
                max_retries=3,
                exceptions=(OpenAIError,)
            )
            
            embedding = response.data[0].embedding
            self._log_operation("embed_query", {"text_length": len(text)}, "debug")
            
            return embedding
            
        except OpenAIError as e:
            raise ExternalAPIError(
                api_name="OpenAI Embeddings",
                message=f"Failed to generate embedding: {str(e)}",
                original_error=e
            )
        except Exception as e:
            self.logger.error(f"Unexpected error generating embedding: {e}")
            return None
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Efficiently processes large numbers of texts by batching API calls.
        
        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process per API call.
                       Maximum is 2048 per OpenAI docs.
            
        Returns:
            List of embedding vectors. None for texts that failed.
            
        Example:
            >>> texts = ["product 1", "product 2", "product 3"]
            >>> embeddings = service.embed_batch(texts)
            >>> print(f"Generated {len(embeddings)} embeddings")
        """
        if not texts:
            return []
        
        all_embeddings: List[Optional[List[float]]] = []
        batch_size = min(batch_size, 2048)  # OpenAI limit
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Clean the batch
            cleaned_batch = [self._preprocess_text(t) for t in batch]
            
            # Filter out empty texts and track indices
            valid_indices = []
            valid_texts = []
            for idx, text in enumerate(cleaned_batch):
                if text:
                    valid_indices.append(idx)
                    valid_texts.append(text)
            
            try:
                if valid_texts:
                    response = self._with_retry(
                        lambda: self._client.embeddings.create(
                            input=valid_texts,
                            model=self.model
                        ),
                        max_retries=3,
                        exceptions=(OpenAIError,)
                    )
                    
                    # Map embeddings back to original indices
                    batch_embeddings: List[Optional[List[float]]] = [None] * len(batch)
                    for embedding_idx, data in enumerate(response.data):
                        original_idx = valid_indices[embedding_idx]
                        batch_embeddings[original_idx] = data.embedding
                    
                    all_embeddings.extend(batch_embeddings)
                else:
                    all_embeddings.extend([None] * len(batch))
                
                self.logger.info(
                    f"Processed batch {i // batch_size + 1}, "
                    f"total: {len(all_embeddings)}/{len(texts)}"
                )
                
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                all_embeddings.extend([None] * len(batch))
        
        return all_embeddings
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: Raw input text.
            
        Returns:
            Cleaned and normalized text.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Strip whitespace and normalize newlines
        cleaned = text.strip().replace("\n", " ")
        
        # Remove multiple spaces
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        
        return cleaned
    
    def health_check(self) -> dict:
        """
        Check if the embedding service is operational.
        
        Returns:
            Health status dictionary.
        """
        base_health = super().health_check()
        
        try:
            # Try a simple embedding to verify connectivity
            test_embedding = self.embed_query("test")
            base_health["openai_connected"] = test_embedding is not None
            base_health["model"] = self.model
        except Exception as e:
            base_health["status"] = "unhealthy"
            base_health["error"] = str(e)
        
        return base_health
