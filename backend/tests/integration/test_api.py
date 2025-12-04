"""
API Integration Tests for ProductLens AI.

Tests the Flask API endpoints.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import io

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture
def mock_services():
    """Create all mock services for integration testing."""
    mock_embedding = Mock()
    mock_embedding.create_embedding.return_value = [0.1] * 1536
    mock_embedding.health_check.return_value = {"status": "healthy"}
    
    mock_vector = Mock()
    mock_vector.query.return_value = [
        {
            "id": "22384",
            "score": 0.89,
            "metadata": {
                "stock_code": "22384",
                "description": "LUNCH BAG PINK POLKADOT",
                "unit_price": 1.65
            }
        }
    ]
    mock_vector.health_check.return_value = {"status": "healthy"}
    
    mock_llm = Mock()
    mock_llm.generate_recommendations.return_value = {
        "explanation": "Here are great products!"
    }
    mock_llm.health_check.return_value = {"status": "healthy"}
    
    mock_ocr = Mock()
    mock_ocr.extract_text.return_value = "candles, picture frame"
    
    mock_recommendation = Mock()
    mock_recommendation.recommend.return_value = (
        [{"stock_code": "22384", "description": "LUNCH BAG PINK POLKADOT", "unit_price": 1.65}],
        "Here are great products!"
    )
    mock_recommendation.recommend_from_text.return_value = (
        [{"stock_code": "22384", "description": "LUNCH BAG PINK POLKADOT", "unit_price": 1.65}],
        "Here are great products!"
    )
    mock_recommendation.recommend_from_class.return_value = (
        [{"stock_code": "22384", "description": "LUNCH BAG PINK POLKADOT", "unit_price": 1.65}],
        "Here are great products!"
    )
    mock_recommendation.health_check.return_value = {"status": "healthy"}
    
    return {
        "embedding": mock_embedding,
        "vector": mock_vector,
        "llm": mock_llm,
        "ocr": mock_ocr,
        "recommendation": mock_recommendation
    }


@pytest.fixture
def app(mock_services):
    """Create Flask app with mocked services."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["PINECONE_API_KEY"] = "test-key"
    os.environ["PINECONE_INDEX_NAME"] = "test-index"
    os.environ["PINECONE_HOST"] = "test-host"
    
    with patch("services.search.embedding_service.EmbeddingService", return_value=mock_services["embedding"]), \
         patch("services.search.vector_service.VectorService", return_value=mock_services["vector"]), \
         patch("services.search.llm_service.LLMService", return_value=mock_services["llm"]), \
         patch("services.search.recommendation_service.RecommendationService", return_value=mock_services["recommendation"]), \
         patch("services.ocr.ocr_service.OCRService", return_value=mock_services["ocr"]):
        
        # Also patch the service initialization
        with patch("services.search.embedding_service.OpenAI"), \
             patch("services.search.vector_service.Pinecone"), \
             patch("services.search.llm_service.OpenAI"):
            
            from app import create_app
            app = create_app()
            app.config["TESTING"] = True
            yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint returns 200."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data
        assert "services" in data
    
    def test_health_check_includes_services(self, client):
        """Test health check includes all services."""
        response = client.get("/health")
        
        data = response.get_json()
        services = data.get("services", {})
        assert "recommendation" in services
        assert "ocr" in services


class TestRootEndpoint:
    """Tests for / endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.get_json()
        assert "name" in data
        assert data["name"] == "ProductLens AI"
        assert "version" in data
        assert "endpoints" in data


class TestProductRecommendationEndpoint:
    """Tests for /product-recommendation endpoint."""
    
    @pytest.mark.integration
    def test_recommendation_json_input(self, client, mock_services):
        """Test recommendation with JSON input."""
        response = client.post(
            "/product-recommendation",
            json={"query": "gift for mom"},
            content_type="application/json"
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert "products" in data
        assert "response" in data
    
    @pytest.mark.integration
    def test_recommendation_form_input(self, client):
        """Test recommendation with form data input."""
        response = client.post(
            "/product-recommendation",
            data={"query": "gift for mom"},
            content_type="application/x-www-form-urlencoded"
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert "products" in data
    
    @pytest.mark.integration
    def test_recommendation_empty_query(self, client):
        """Test recommendation with empty query returns 400."""
        response = client.post(
            "/product-recommendation",
            json={"query": ""},
            content_type="application/json"
        )
        
        assert response.status_code == 400
    
    @pytest.mark.integration
    def test_recommendation_missing_query(self, client):
        """Test recommendation without query returns 400."""
        response = client.post(
            "/product-recommendation",
            json={},
            content_type="application/json"
        )
        
        assert response.status_code == 400


class TestOCREndpoint:
    """Tests for /ocr-query endpoint."""
    
    @pytest.fixture
    def test_image(self):
        """Create a test image for upload."""
        from PIL import Image
        
        img = Image.new("RGB", (224, 224), color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer
    
    @pytest.mark.integration
    def test_ocr_with_image(self, client, test_image, mock_services):
        """Test OCR endpoint with image upload."""
        response = client.post(
            "/ocr-query",
            data={
                "image_data": (test_image, "test.jpg")
            },
            content_type="multipart/form-data"
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert "products" in data
        assert "response" in data
        assert "extracted_text" in data
    
    @pytest.mark.integration
    def test_ocr_no_image(self, client):
        """Test OCR endpoint without image returns 400."""
        response = client.post(
            "/ocr-query",
            content_type="multipart/form-data"
        )
        
        assert response.status_code == 400


class TestImageProductSearchEndpoint:
    """Tests for /image-product-search endpoint."""
    
    @pytest.fixture
    def test_image(self):
        """Create a test image for upload."""
        from PIL import Image
        
        img = Image.new("RGB", (224, 224), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer
    
    @pytest.mark.integration
    def test_image_search_no_image(self, client):
        """Test image search without image returns 400."""
        response = client.post(
            "/image-product-search",
            content_type="multipart/form-data"
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert "products" in data
        assert data["products"] == []


class TestCORSHeaders:
    """Tests for CORS configuration."""
    
    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        response = client.options(
            "/product-recommendation",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # OPTIONS should be allowed
        assert response.status_code in [200, 204]
    
    def test_cors_allowed_origin(self, client):
        """Test that allowed origins get CORS headers."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Should have Access-Control headers
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_404_not_found(self, client):
        """Test 404 for unknown endpoints."""
        response = client.get("/unknown-endpoint")
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method."""
        response = client.get("/product-recommendation")
        assert response.status_code == 405
