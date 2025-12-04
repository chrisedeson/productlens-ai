"""
Unit Tests for RecommendationService.

Tests the complete recommendation pipeline.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestRecommendationService:
    """Tests for RecommendationService."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        mock = Mock()
        mock.create_embedding.return_value = [0.1] * 1536
        mock.health_check.return_value = {"status": "healthy"}
        return mock
    
    @pytest.fixture
    def mock_vector_service(self):
        """Create a mock vector service."""
        mock = Mock()
        mock.query.return_value = [
            {
                "id": "22384",
                "score": 0.89,
                "metadata": {
                    "stock_code": "22384",
                    "description": "LUNCH BAG PINK POLKADOT",
                    "unit_price": 1.65
                }
            },
            {
                "id": "22423",
                "score": 0.85,
                "metadata": {
                    "stock_code": "22423",
                    "description": "REGENCY CAKESTAND 3 TIER",
                    "unit_price": 12.75
                }
            }
        ]
        mock.health_check.return_value = {"status": "healthy"}
        return mock
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        mock = Mock()
        mock.generate_recommendations.return_value = {
            "explanation": "Here are some great products!",
            "recommendations": []
        }
        mock.generate_ocr_response.return_value = "I found these items from your list!"
        mock.generate_image_detection_response.return_value = "I identified this product!"
        mock.health_check.return_value = {"status": "healthy"}
        return mock
    
    @pytest.fixture
    def service(self, mock_embedding_service, mock_vector_service, mock_llm_service):
        """Create a RecommendationService with mocked dependencies."""
        from services.search.recommendation_service import RecommendationService
        
        return RecommendationService(
            embedding_service=mock_embedding_service,
            vector_service=mock_vector_service,
            llm_service=mock_llm_service
        )
    
    def test_recommend_success(
        self, 
        service, 
        mock_embedding_service, 
        mock_vector_service, 
        mock_llm_service
    ):
        """Test successful recommendation."""
        products, response = service.recommend("gift for mom", top_k=5)
        
        assert len(products) == 2
        assert products[0]["stock_code"] == "22384"
        mock_embedding_service.create_embedding.assert_called_once_with("gift for mom")
        mock_vector_service.query.assert_called_once()
    
    def test_recommend_empty_query(self, service):
        """Test that empty query raises ValidationError."""
        from core.exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            service.recommend("", top_k=5)
    
    def test_recommend_no_results(
        self, 
        service, 
        mock_vector_service
    ):
        """Test recommendation with no matching products."""
        mock_vector_service.query.return_value = []
        
        products, response = service.recommend("very specific query", top_k=5)
        
        assert products == []
        assert "No products found" in response or response != ""
    
    def test_recommend_embedding_failure(
        self, 
        service, 
        mock_embedding_service
    ):
        """Test handling of embedding service failure."""
        mock_embedding_service.create_embedding.return_value = None
        
        products, response = service.recommend("test query", top_k=5)
        
        # Should return empty results gracefully
        assert products == []
    
    def test_recommend_from_text(
        self, 
        service, 
        mock_llm_service
    ):
        """Test recommendation from OCR text."""
        products, response = service.recommend_from_text(
            "candles, picture frame, mug",
            top_k=5,
            is_ocr=True
        )
        
        assert len(products) == 2
        mock_llm_service.generate_ocr_response.assert_called_once()
    
    def test_recommend_from_class(
        self, 
        service, 
        mock_llm_service
    ):
        """Test recommendation from CNN class prediction."""
        products, response = service.recommend_from_class(
            "LUNCH_BAG_PINK_POLKADOT",
            top_k=5
        )
        
        assert len(products) == 2
        mock_llm_service.generate_image_detection_response.assert_called_once()
    
    def test_get_recommendations_returns_dataclass(self, service):
        """Test that get_recommendations returns RecommendationResult."""
        from services.search.recommendation_service import RecommendationResult
        
        result = service.get_recommendations("test query", top_k=5)
        
        assert isinstance(result, RecommendationResult)
        assert result.success is True
        assert result.query == "test query"
    
    def test_health_check(self, service):
        """Test health check includes all sub-services."""
        result = service.health_check()
        
        assert "status" in result
        assert "embedding_service" in result
        assert "vector_service" in result
        assert "llm_service" in result


class TestRecommendationServiceFactory:
    """Tests for RecommendationServiceFactory."""
    
    def test_factory_creates_service(self):
        """Test that factory creates a properly configured service."""
        with patch("services.search.recommendation_service.EmbeddingService"), \
             patch("services.search.recommendation_service.VectorService"), \
             patch("services.search.recommendation_service.LLMService"):
            
            from services.search.recommendation_service import RecommendationServiceFactory
            
            factory = RecommendationServiceFactory(
                openai_api_key="test-key",
                pinecone_api_key="test-key",
                pinecone_index="test-index",
                pinecone_host="test-host"
            )
            
            service = factory.create()
            
            assert service is not None


class TestRecommendationResult:
    """Tests for RecommendationResult dataclass."""
    
    def test_result_dataclass(self):
        """Test RecommendationResult attributes."""
        from services.search.recommendation_service import RecommendationResult
        
        result = RecommendationResult(
            success=True,
            query="test query",
            recommendations=[{"id": "1"}],
            explanation="Test explanation",
            metadata={"count": 1}
        )
        
        assert result.success is True
        assert result.query == "test query"
        assert len(result.recommendations) == 1
        assert result.explanation == "Test explanation"
        assert result.metadata["count"] == 1
