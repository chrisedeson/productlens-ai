"""
Recommendation Service for ProductLens AI.
Handles product recommendation logic using embeddings and vector search.
"""

from typing import List, Dict, Any, Tuple
import logging

from .embedding_service import EmbeddingService
from .vector_service import VectorService
from .llm_service import LLMService
from models.product import Product
from utils.validators import QueryValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Service class for product recommendations.
    
    Orchestrates:
    - Query validation
    - Embedding generation
    - Vector similarity search
    - Natural language response generation
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_service: VectorService,
        llm_service: LLMService
    ):
        """
        Initialize the RecommendationService.
        
        Args:
            embedding_service: Service for generating embeddings.
            vector_service: Service for vector database operations.
            llm_service: Service for generating natural language responses.
        """
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.llm_service = llm_service
    
    def recommend(
        self, 
        query: str, 
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Get product recommendations for a natural language query.
        
        Args:
            query: User's natural language query.
            top_k: Number of products to return.
            
        Returns:
            Tuple of (products list, natural language response).
        """
        # Validate the query
        is_valid, error_message = QueryValidator.validate(query)
        if not is_valid:
            return [], error_message
        
        # Sanitize the query
        query = QueryValidator.sanitize(query)
        
        # Generate embedding for the query
        query_embedding = self.embedding_service.embed_query(query)
        if not query_embedding:
            return [], "Unable to process your query. Please try again."
        
        # Search for similar products
        matches = self.vector_service.query(
            query_vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        if not matches:
            return [], "No products found matching your query. Try different search terms."
        
        # Convert matches to product format
        products = []
        for match in matches:
            metadata = match.get("metadata", {})
            products.append({
                "stock_code": metadata.get("stock_code", ""),
                "description": metadata.get("description", ""),
                "unit_price": float(metadata.get("unit_price", 0)),
                "country": metadata.get("country", "Unknown"),
                "score": match.get("score", 0)
            })
        
        # Generate natural language response
        response = self.llm_service.generate_product_response(query, products)
        
        # Remove score from final output (internal use only)
        for product in products:
            product.pop("score", None)
        
        return products, response
    
    def recommend_from_text(
        self, 
        text: str, 
        top_k: int = 5,
        is_ocr: bool = False
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Get recommendations from extracted text (e.g., from OCR).
        
        Args:
            text: Extracted text to use as query.
            top_k: Number of products to return.
            is_ocr: Whether the text came from OCR.
            
        Returns:
            Tuple of (products list, natural language response).
        """
        products, response = self.recommend(text, top_k)
        
        if is_ocr and products:
            response = self.llm_service.generate_ocr_response(text, products)
        
        return products, response
    
    def recommend_from_class(
        self, 
        predicted_class: str, 
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Get recommendations based on a predicted product class.
        
        Args:
            predicted_class: Class name predicted by CNN.
            top_k: Number of products to return.
            
        Returns:
            Tuple of (products list, natural language response).
        """
        # Use the class name as the search query
        products, _ = self.recommend(predicted_class, top_k)
        
        # Generate appropriate response for image detection
        response = self.llm_service.generate_image_detection_response(
            predicted_class, 
            products
        )
        
        return products, response
