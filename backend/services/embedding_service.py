"""
Embedding Service for ProductLens AI.
Handles generating embeddings using OpenAI's text-embedding-3-small model.
"""

from typing import List, Optional
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service class for generating text embeddings using OpenAI.
    
    Uses the text-embedding-3-small model which produces 1536-dimensional vectors.
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize the EmbeddingService.
        
        Args:
            api_key: OpenAI API key.
            model: Embedding model to use.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = 1536  # text-embedding-3-small output dimensions
        
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Input text to embed.
            
        Returns:
            List of floats representing the embedding vector, or None on error.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            # Clean the text
            text = text.strip().replace("\n", " ")
            
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def get_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process per API call.
            
        Returns:
            List of embedding vectors (or None for failed texts).
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Clean the batch
            cleaned_batch = [
                t.strip().replace("\n", " ") if t else ""
                for t in batch
            ]
            
            try:
                response = self.client.embeddings.create(
                    input=cleaned_batch,
                    model=self.model
                )
                
                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Processed batch {i // batch_size + 1}, "
                           f"total: {len(all_embeddings)}/{len(texts)}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                # Fill with None for failed batch
                all_embeddings.extend([None] * len(batch))
        
        return all_embeddings
    
    def embed_query(self, query: str) -> Optional[List[float]]:
        """
        Generate an embedding for a search query.
        
        This is a convenience method that's semantically clear for search operations.
        
        Args:
            query: Search query text.
            
        Returns:
            Embedding vector or None on error.
        """
        return self.get_embedding(query)
