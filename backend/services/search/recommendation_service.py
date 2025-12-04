"""
Recommendation Service for ProductLens AI.

Orchestrates the complete product recommendation pipeline.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from core.base_service import BaseService
from core.exceptions import (
    ServiceError,
    ValidationError,
    ProductLensError
)
from .embedding_service import EmbeddingService
from .vector_service import VectorService
from .llm_service import LLMService

logging.basicConfig(level=logging.INFO)


@dataclass
class RecommendationResult:
    """
    Container for recommendation results.
    
    Attributes:
        success: Whether the recommendation was successful.
        query: Original user query.
        recommendations: List of recommended products.
        explanation: AI-generated explanation.
        metadata: Additional information about the request.
    """
    success: bool
    query: str
    recommendations: List[Dict[str, Any]]
    explanation: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "query": self.query,
            "recommendations": self.recommendations,
            "explanation": self.explanation,
            "metadata": self.metadata
        }


class RecommendationService(BaseService):
    """
    Orchestrates the product recommendation pipeline.
    
    This is the main entry point for semantic product search. It coordinates:
    1. Query embedding generation (EmbeddingService)
    2. Vector similarity search (VectorService)
    3. AI recommendation formatting (LLMService)
    
    The service uses dependency injection for all sub-services,
    making it easy to test and configure.
    
    Attributes:
        embedding_service: Service for generating embeddings.
        vector_service: Service for vector similarity search.
        llm_service: Service for AI recommendations.
    
    Example:
        >>> service = RecommendationService(
        ...     embedding_service=embedding_svc,
        ...     vector_service=vector_svc,
        ...     llm_service=llm_svc
        ... )
        >>> result = service.get_recommendations("gift for mom")
        >>> print(result.explanation)
    """
    
    DEFAULT_TOP_K = 5
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_service: VectorService,
        llm_service: LLMService
    ):
        """
        Initialize the Recommendation Service.
        
        Args:
            embedding_service: Service for generating embeddings.
            vector_service: Service for vector similarity search.
            llm_service: Service for AI recommendations.
            
        Raises:
            ValidationError: If any required service is None.
        """
        super().__init__("RecommendationService")
        
        # Validate dependencies
        if embedding_service is None:
            raise ValidationError("embedding_service", "EmbeddingService is required")
        if vector_service is None:
            raise ValidationError("vector_service", "VectorService is required")
        if llm_service is None:
            raise ValidationError("llm_service", "LLMService is required")
        
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.llm_service = llm_service
        
        self._mark_initialized()
        self.logger.info("RecommendationService initialized with all dependencies")
    
    def get_recommendations(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        include_ai_explanation: bool = True,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> RecommendationResult:
        """
        Get product recommendations for a search query.
        
        This is the main method that orchestrates the recommendation pipeline:
        1. Generate embedding for the query
        2. Search vector database for similar products
        3. Format results with AI explanations (optional)
        
        Args:
            query: User's search query.
            top_k: Number of recommendations to return.
            include_ai_explanation: Whether to generate AI explanations.
            filter_dict: Optional metadata filters for search.
            
        Returns:
            RecommendationResult with products and explanations.
            
        Example:
            >>> result = service.get_recommendations(
            ...     query="birthday gift for dad who likes gardening",
            ...     top_k=5,
            ...     filter_dict={"country": "UK"}
            ... )
            >>> for product in result.recommendations:
            ...     print(product["description"])
        """
        # Validate input
        if not query or not query.strip():
            raise ValidationError("query", "Search query cannot be empty")
        
        query = query.strip()
        
        self._log_operation(
            "get_recommendations",
            {"query": query, "top_k": top_k}
        )
        
        try:
            # Step 1: Generate query embedding
            embedding = self.embedding_service.create_embedding(query)
            
            if not embedding:
                return RecommendationResult(
                    success=False,
                    query=query,
                    recommendations=[],
                    explanation="Failed to process your search query. Please try again.",
                    metadata={"error": "embedding_failed"}
                )
            
            # Step 2: Search vector database
            matches = self.vector_service.query(
                query_vector=embedding,
                top_k=top_k,
                include_metadata=True,
                filter_dict=filter_dict
            )
            
            if not matches:
                return RecommendationResult(
                    success=True,
                    query=query,
                    recommendations=[],
                    explanation="No products found matching your search. "
                               "Try different keywords or browse our categories.",
                    metadata={"matched_count": 0}
                )
            
            # Extract product data from metadata
            products = []
            for match in matches:
                product_data = match.get("metadata", {})
                product_data["score"] = match.get("score", 0)
                product_data["id"] = match.get("id", "")
                products.append(product_data)
            
            # Step 3: Generate AI explanation (optional)
            explanation = ""
            if include_ai_explanation:
                try:
                    llm_result = self.llm_service.generate_recommendations(
                        query=query,
                        products=products
                    )
                    explanation = llm_result.get("explanation", "")
                except ProductLensError as e:
                    self.logger.warning(f"LLM generation failed: {e}")
                    explanation = self._generate_fallback_explanation(products)
            else:
                explanation = self._generate_fallback_explanation(products)
            
            return RecommendationResult(
                success=True,
                query=query,
                recommendations=products,
                explanation=explanation,
                metadata={
                    "matched_count": len(products),
                    "ai_generated": include_ai_explanation
                }
            )
            
        except ValidationError:
            raise
        except ProductLensError as e:
            self.logger.error(f"Recommendation pipeline error: {e}")
            raise ServiceError(
                service_name="RecommendationService",
                operation="get_recommendations",
                message=f"Failed to get recommendations: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in recommendation pipeline: {e}")
            raise ServiceError(
                service_name="RecommendationService",
                operation="get_recommendations",
                message="An unexpected error occurred"
            )
    
    def search_similar_products(
        self,
        product_id: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Find products similar to a given product.
        
        Useful for "customers also viewed" or "related products" features.
        
        Args:
            product_id: ID of the product to find similar items for.
            top_k: Number of similar products to return.
            
        Returns:
            List of similar products.
        """
        # TODO: Implement product-to-product similarity
        # This would require fetching the product's embedding first
        raise NotImplementedError(
            "Product similarity search not yet implemented"
        )
    
    def _generate_fallback_explanation(
        self,
        products: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a simple explanation when AI is unavailable.
        
        Args:
            products: List of matched products.
            
        Returns:
            Simple explanation string.
        """
        if not products:
            return "No products found."
        
        count = len(products)
        if count == 1:
            return "Here's a product that matches your search."
        return f"Here are {count} products that match your search."
    
    # ----- Backward Compatibility Methods -----
    # These methods maintain compatibility with the original API
    
    def recommend(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Get recommendations (backward compatible interface).
        
        This method provides a simple tuple-based return for compatibility
        with existing route handlers.
        
        Args:
            query: Search query.
            top_k: Number of recommendations.
            
        Returns:
            Tuple of (products list, response text).
        """
        result = self.get_recommendations(query=query, top_k=top_k)
        return result.recommendations, result.explanation
    
    def recommend_from_text(
        self,
        text: str,
        top_k: int = DEFAULT_TOP_K,
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
            try:
                ocr_response = self.llm_service.generate_ocr_response(text, products)
                if ocr_response:
                    response = ocr_response
            except Exception as e:
                self.logger.warning(f"Failed to generate OCR response: {e}")
        
        return products, response
    
    def recommend_from_class(
        self,
        predicted_class: str,
        top_k: int = DEFAULT_TOP_K
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Get recommendations based on a predicted product class.
        
        Args:
            predicted_class: Class name predicted by CNN.
            top_k: Number of products to return.
            
        Returns:
            Tuple of (products list, natural language response).
        """
        products, _ = self.recommend(predicted_class, top_k)
        
        # Generate appropriate response for image detection
        try:
            response = self.llm_service.generate_image_detection_response(
                predicted_class,
                products
            )
        except Exception as e:
            self.logger.warning(f"Failed to generate image response: {e}")
            response = f"Found products similar to: {predicted_class}"
        
        return products, response
    
    def health_check(self) -> dict:
        """
        Check the health of all dependent services.
        
        Returns:
            Comprehensive health status of the recommendation pipeline.
        """
        base_health = super().health_check()
        
        # Check all sub-services
        try:
            base_health["embedding_service"] = self.embedding_service.health_check()
        except Exception as e:
            base_health["embedding_service"] = {"status": "error", "error": str(e)}
        
        try:
            base_health["vector_service"] = self.vector_service.health_check()
        except Exception as e:
            base_health["vector_service"] = {"status": "error", "error": str(e)}
        
        try:
            base_health["llm_service"] = self.llm_service.health_check()
        except Exception as e:
            base_health["llm_service"] = {"status": "error", "error": str(e)}
        
        # Overall status
        all_healthy = all(
            svc.get("status") == "healthy"
            for svc in [
                base_health.get("embedding_service", {}),
                base_health.get("vector_service", {}),
                base_health.get("llm_service", {})
            ]
        )
        
        base_health["status"] = "healthy" if all_healthy else "degraded"
        
        return base_health


class RecommendationServiceFactory:
    """
    Factory for creating RecommendationService instances.
    
    Handles service initialization and dependency wiring.
    
    Example:
        >>> factory = RecommendationServiceFactory(
        ...     openai_api_key="...",
        ...     pinecone_api_key="...",
        ...     pinecone_index="products",
        ...     pinecone_host="..."
        ... )
        >>> service = factory.create()
    """
    
    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: str,
        pinecone_index: str,
        pinecone_host: str
    ):
        """
        Initialize the factory with API credentials.
        
        Args:
            openai_api_key: OpenAI API key.
            pinecone_api_key: Pinecone API key.
            pinecone_index: Pinecone index name.
            pinecone_host: Pinecone host URL.
        """
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index = pinecone_index
        self.pinecone_host = pinecone_host
    
    def create(self) -> RecommendationService:
        """
        Create a fully configured RecommendationService.
        
        Returns:
            Configured RecommendationService instance.
            
        Raises:
            ConfigurationError: If service initialization fails.
        """
        embedding_service = EmbeddingService(api_key=self.openai_api_key)
        vector_service = VectorService(
            api_key=self.pinecone_api_key,
            index_name=self.pinecone_index,
            host=self.pinecone_host
        )
        llm_service = LLMService(api_key=self.openai_api_key)
        
        return RecommendationService(
            embedding_service=embedding_service,
            vector_service=vector_service,
            llm_service=llm_service
        )
