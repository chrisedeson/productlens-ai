"""
Vector Service for ProductLens AI.
Handles Pinecone vector database operations.
"""

from typing import List, Dict, Optional, Any
import logging
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorService:
    """
    Service class for managing vectors in Pinecone.
    
    Handles:
    - Connecting to Pinecone index
    - Upserting product vectors
    - Querying for similar products
    """
    
    def __init__(
        self, 
        api_key: str, 
        index_name: str,
        host: str
    ):
        """
        Initialize the VectorService.
        
        Args:
            api_key: Pinecone API key.
            index_name: Name of the Pinecone index.
            host: Pinecone host URL.
        """
        self.api_key = api_key
        self.index_name = index_name
        self.host = host
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(name=index_name, host=host)
        
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    def upsert_vectors(
        self, 
        vectors: List[Dict[str, Any]], 
        batch_size: int = 100
    ) -> int:
        """
        Upsert vectors to Pinecone in batches.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'values', and 'metadata'.
            batch_size: Number of vectors to upsert per batch.
            
        Returns:
            Total number of vectors upserted.
        """
        total_upserted = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            try:
                self.index.upsert(vectors=batch)
                total_upserted += len(batch)
                logger.info(f"Upserted batch {i // batch_size + 1}, "
                           f"total: {total_upserted}/{len(vectors)}")
            except Exception as e:
                logger.error(f"Error upserting batch: {e}")
        
        return total_upserted
    
    def upsert_product(
        self, 
        product_id: str, 
        embedding: List[float], 
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Upsert a single product vector.
        
        Args:
            product_id: Unique identifier for the product.
            embedding: Vector representation of the product.
            metadata: Product metadata (stock_code, description, etc.).
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.index.upsert(vectors=[{
                "id": product_id,
                "values": embedding,
                "metadata": metadata
            }])
            return True
        except Exception as e:
            logger.error(f"Error upserting product {product_id}: {e}")
            return False
    
    def query(
        self, 
        query_vector: List[float], 
        top_k: int = 5,
        include_metadata: bool = True,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar products.
        
        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            include_metadata: Whether to include metadata in results.
            filter_dict: Optional metadata filter.
            
        Returns:
            List of matching products with scores.
        """
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                filter=filter_dict
            )
            
            # Format results
            matches = []
            for match in results.matches:
                matches.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata if include_metadata else {}
                })
            
            return matches
            
        except Exception as e:
            logger.error(f"Error querying vectors: {e}")
            return []
    
    def delete_all(self) -> bool:
        """
        Delete all vectors from the index.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.index.delete(delete_all=True)
            logger.info("Deleted all vectors from index")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary containing index statistics.
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
