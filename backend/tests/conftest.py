"""
Pytest Configuration and Fixtures for ProductLens AI Tests.

This module provides shared fixtures for all tests.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List
import os
import sys

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----- Mock Data Fixtures -----

@pytest.fixture
def sample_products() -> List[Dict[str, Any]]:
    """Sample product data for testing."""
    return [
        {
            "stock_code": "22384",
            "description": "LUNCH BAG PINK POLKADOT",
            "unit_price": 1.65,
            "country": "United Kingdom",
            "score": 0.89
        },
        {
            "stock_code": "22423",
            "description": "REGENCY CAKESTAND 3 TIER",
            "unit_price": 12.75,
            "country": "United Kingdom",
            "score": 0.85
        },
        {
            "stock_code": "20725",
            "description": "LUNCH BAG RED RETROSPOT",
            "unit_price": 1.65,
            "country": "United Kingdom",
            "score": 0.82
        }
    ]


@pytest.fixture
def sample_embedding() -> List[float]:
    """Sample embedding vector for testing."""
    return [0.1] * 1536  # OpenAI embedding dimension


@pytest.fixture
def sample_query() -> str:
    """Sample search query for testing."""
    return "gift for mom who likes gardening"


@pytest.fixture
def sample_ocr_text() -> str:
    """Sample OCR extracted text for testing."""
    return "1. Candles 2. Picture frame 3. Mug"


# ----- Mock Service Fixtures -----

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    
    # Mock embeddings
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
    mock_client.embeddings.create.return_value = mock_embedding_response
    
    # Mock chat completions
    mock_chat_response = Mock()
    mock_chat_response.choices = [
        Mock(message=Mock(content="Here are some great product recommendations!"))
    ]
    mock_client.chat.completions.create.return_value = mock_chat_response
    
    return mock_client


@pytest.fixture
def mock_pinecone_index():
    """Mock Pinecone index for testing."""
    mock_index = Mock()
    
    # Mock query response
    mock_index.query.return_value = Mock(
        matches=[
            Mock(
                id="22384",
                score=0.89,
                metadata={
                    "stock_code": "22384",
                    "description": "LUNCH BAG PINK POLKADOT",
                    "unit_price": 1.65
                }
            ),
            Mock(
                id="22423",
                score=0.85,
                metadata={
                    "stock_code": "22423",
                    "description": "REGENCY CAKESTAND 3 TIER",
                    "unit_price": 12.75
                }
            )
        ]
    )
    
    return mock_index


# ----- Service Instance Fixtures -----

@pytest.fixture
def embedding_service(mock_openai_client):
    """Create an EmbeddingService with mocked OpenAI client."""
    with patch("services.search.embedding_service.OpenAI", return_value=mock_openai_client):
        from services.search.embedding_service import EmbeddingService
        service = EmbeddingService(api_key="test-key")
        service._client = mock_openai_client
        return service


@pytest.fixture
def vector_service(mock_pinecone_index):
    """Create a VectorService with mocked Pinecone client."""
    with patch("services.search.vector_service.Pinecone") as mock_pc:
        mock_pc.return_value.Index.return_value = mock_pinecone_index
        from services.search.vector_service import VectorService
        service = VectorService(
            api_key="test-key",
            index_name="test-index",
            host="test-host"
        )
        return service


@pytest.fixture
def llm_service(mock_openai_client):
    """Create an LLMService with mocked OpenAI client."""
    with patch("services.search.llm_service.OpenAI", return_value=mock_openai_client):
        from services.search.llm_service import LLMService
        service = LLMService(api_key="test-key")
        service._client = mock_openai_client
        return service


# ----- Flask App Fixture -----

@pytest.fixture
def app():
    """Create a Flask app for testing."""
    # Set test environment variables
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["PINECONE_API_KEY"] = "test-key"
    os.environ["PINECONE_INDEX_NAME"] = "test-index"
    os.environ["PINECONE_HOST"] = "test-host"
    
    # Import and create app with testing config
    with patch("app.EmbeddingService"), \
         patch("app.VectorService"), \
         patch("app.LLMService"), \
         patch("app.RecommendationService"), \
         patch("app.OCRService"):
        from app import create_app
        app = create_app()
        app.config["TESTING"] = True
        return app


@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    return app.test_client()


# ----- Utility Fixtures -----

@pytest.fixture
def temp_image_file(tmp_path):
    """Create a temporary image file for testing."""
    from PIL import Image
    import io
    
    # Create a simple test image
    img = Image.new("RGB", (224, 224), color="red")
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)
    
    return img_path


@pytest.fixture
def temp_image_bytes():
    """Create image bytes for testing."""
    from PIL import Image
    import io
    
    img = Image.new("RGB", (224, 224), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    
    return buffer.read()


# ----- Configuration -----

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests as ML-specific tests"
    )
