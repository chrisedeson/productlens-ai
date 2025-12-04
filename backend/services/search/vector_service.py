"""
Vector Service for ProductLens AI.

Handles Pinecone vector database operations for semantic product search.
"""

from typing import List, Dict, Optional, Any
import logging

from pinecone import Pinecone, PineconeException

from core.base_service import BaseService, RetryMixin
from core.exceptions import ExternalAPIError, ValidationError, ConfigurationError

logging.basicConfig(level=logging.INFO)


class VectorService(BaseService, RetryMixin):
    """
    Service for managing vectors in Pinecone.
    
    Handles:
    - Connecting to Pinecone index
    - Upserting product vectors
    - Querying for similar products
    - Managing vector metadata
    
    The vector database stores product embeddings with metadata,
    enabling fast semantic similarity search.
    
    Attributes:
        index_name: Name of the Pinecone index.
        host: Pinecone host URL.
    
    Example:
        >>> service = VectorService(api_key="...", index_name="products", host="...")
        >>> results = service.query(embedding_vector, top_k=5)
    """
    
    DEFAULT_TOP_K = 5
    DEFAULT_BATCH_SIZE = 100
    
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
            
        Raises:
            ValidationError: If required parameters are missing.
            ConfigurationError: If connection to Pinecone fails.
        """
        super().__init__("VectorService")
        
        # Validate inputs
        if not api_key:
            raise ValidationError("api_key", "Pinecone API key is required")
        if not index_name:
            raise ValidationError("index_name", "Pinecone index name is required")
        if not host:
            raise ValidationError("host", "Pinecone host URL is required")
        
        self.index_name = index_name
        self.host = host
        
        try:
            self._pc = Pinecone(api_key=api_key)
            self._index = self._pc.Index(name=index_name, host=host)
            self._mark_initialized()
            self.logger.info(f"Connected to Pinecone index: {index_name}")
        except PineconeException as e:
            raise ConfigurationError(
                "pinecone_connection",
                f"Failed to connect to Pinecone: {str(e)}"
            )
    
    def query(
        self,
        query_vector: List[float],
        top_k: int = DEFAULT_TOP_K,
        include_metadata: bool = True,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar products.
        
        Args:
            query_vector: Query embedding vector (1536 dimensions).
            top_k: Number of results to return.
            include_metadata: Whether to include product metadata.
            filter_dict: Optional metadata filter (e.g., {"country": "UK"}).
            
        Returns:
            List of matching products with scores and metadata.
            
        Example:
            >>> results = service.query(embedding, top_k=10)
            >>> for r in results:
            ...     print(f"{r['metadata']['description']}: {r['score']:.3f}")
        """
        if not query_vector:
            self.logger.warning("Empty query vector provided")
            return []
        
        try:
            results = self._with_retry(
                lambda: self._index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=include_metadata,
                    filter=filter_dict
                ),
                max_retries=3,
                exceptions=(PineconeException,)
            )
            
            # Format results
            matches = []
            for match in results.matches:
                matches.append({
                    "id": match.id,
                    "score": float(match.score),
                    "metadata": dict(match.metadata) if include_metadata and match.metadata else {}
                })
            
            self._log_operation(
                "query",
                {"top_k": top_k, "results": len(matches)},
                "debug"
            )
            
            return matches
            
        except PineconeException as e:
            raise ExternalAPIError(
                api_name="Pinecone",
                message=f"Query failed: {str(e)}",
                original_error=e
            )
        except Exception as e:
            self.logger.error(f"Error querying vectors: {e}")
            return []
    
    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> int:
        """
        Upsert vectors to Pinecone in batches.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'values', and 'metadata'.
            batch_size: Number of vectors to upsert per batch.
            
        Returns:
            Total number of vectors upserted.
            
        Example:
            >>> vectors = [
            ...     {"id": "prod_1", "values": [...], "metadata": {"name": "Shoes"}}
            ... ]
            >>> count = service.upsert_vectors(vectors)
        """
        if not vectors:
            return 0
        
        total_upserted = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            try:
                self._with_retry(
                    lambda b=batch: self._index.upsert(vectors=b),
                    max_retries=3,
                    exceptions=(PineconeException,)
                )
                total_upserted += len(batch)
                
                self.logger.info(
                    f"Upserted batch {i // batch_size + 1}, "
                    f"total: {total_upserted}/{len(vectors)}"
                )
                
            except PineconeException as e:
                self.logger.error(f"Error upserting batch: {e}")
        
        return total_upserted
    
    def upsert_single(
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
            self._index.upsert(vectors=[{
                "id": product_id,
                "values": embedding,
                "metadata": metadata
            }])
            return True
        except PineconeException as e:
            self.logger.error(f"Error upserting product {product_id}: {e}")
            return False
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors by their IDs.
        
        Args:
            ids: List of vector IDs to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        if not ids:
            return True
        
        try:
            self._index.delete(ids=ids)
            self.logger.info(f"Deleted {len(ids)} vectors")
            return True
        except PineconeException as e:
            self.logger.error(f"Error deleting vectors: {e}")
            return False
    
    def delete_all(self) -> bool:
        """
        Delete all vectors from the index.
        
        WARNING: This operation is irreversible.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            self._index.delete(delete_all=True)
            self.logger.warning("All vectors deleted from index")
            return True
        except PineconeException as e:
            self.logger.error(f"Error deleting all vectors: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index stats (vector count, dimensions, etc.).
        """
        try:
            stats = self._index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
        except PineconeException as e:
            self.logger.error(f"Error getting index stats: {e}")
            return {}
    
    def health_check(self) -> dict:
        """
        Check if the vector service is operational.
        
        Returns:
            Health status dictionary.
        """
        base_health = super().health_check()
        
        try:
            stats = self.get_stats()
            base_health["pinecone_connected"] = True
            base_health["index_name"] = self.index_name
            base_health["vector_count"] = stats.get("total_vector_count", 0)
        except Exception as e:
            base_health["status"] = "unhealthy"
            base_health["error"] = str(e)
        
        return base_health
