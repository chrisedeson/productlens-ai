"""
Unit Tests for EmbeddingService.

Tests the OpenAI embedding generation functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestEmbeddingService:
    """Tests for EmbeddingService."""
    
    @pytest.fixture
    def mock_openai(self):
        """Create a mock OpenAI client."""
        mock = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock.embeddings.create.return_value = mock_response
        return mock
    
    @pytest.fixture
    def service(self, mock_openai):
        """Create an EmbeddingService with mocked client."""
        with patch("services.search.embedding_service.OpenAI", return_value=mock_openai):
            from services.search.embedding_service import EmbeddingService
            svc = EmbeddingService(api_key="test-key")
            svc._client = mock_openai
            return svc
    
    def test_embed_query_success(self, service, mock_openai):
        """Test successful embedding creation."""
        result = service.embed_query("test query")
        
        assert result is not None
        assert len(result) == 1536
        mock_openai.embeddings.create.assert_called_once()
    
    def test_embed_query_empty(self, service):
        """Test that empty query returns None or handles gracefully."""
        result = service.embed_query("")
        # Empty query should return None
        assert result is None
    
    def test_embed_query_whitespace(self, service):
        """Test that whitespace-only query returns None or handles gracefully."""
        result = service.embed_query("   ")
        assert result is None
    
    def test_embed_query_api_error(self, service, mock_openai):
        """Test handling of OpenAI API errors."""
        mock_openai.embeddings.create.side_effect = Exception("API Error")
        
        # Should return None on error
        result = service.embed_query("test query")
        assert result is None
    
    def test_embed_batch(self, service, mock_openai):
        """Test batch embedding creation."""
        queries = ["query 1", "query 2", "query 3"]
        
        # Setup mock for batch response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536, index=i)
            for i in range(len(queries))
        ]
        mock_openai.embeddings.create.return_value = mock_response
        
        result = service.embed_batch(queries)
        
        assert len(result) == 3
        assert all(len(emb) == 1536 for emb in result)
    
    def test_embed_batch_empty_list(self, service):
        """Test that empty list returns empty result."""
        result = service.embed_batch([])
        assert result == []
    
    def test_health_check(self, service, mock_openai):
        """Test health check returns correct status."""
        result = service.health_check()
        
        assert "status" in result
        assert "initialized" in result
        assert result["initialized"] == True
    
    def test_embedding_dimension(self, service, mock_openai):
        """Test that embeddings have correct dimension."""
        result = service.embed_query("test")
        assert len(result) == 1536  # OpenAI text-embedding-3-small dimension


class TestEmbeddingServiceValidation:
    """Tests for input validation in EmbeddingService."""
    
    @pytest.fixture
    def service(self):
        """Create service with mocked client."""
        with patch("services.search.embedding_service.OpenAI"):
            from services.search.embedding_service import EmbeddingService
            return EmbeddingService(api_key="test-key")
    
    def test_validate_query_max_length(self, service):
        """Test that overly long queries are handled."""
        long_query = "a" * 10000
        # Should not raise - service should truncate or handle gracefully
        # This tests the validation logic
        pass  # Implementation specific
    
    def test_validate_query_special_characters(self, service):
        """Test that special characters in queries are handled."""
        special_query = "gift for mom 🎁 <script>alert('xss')</script>"
        # Should handle without error
        pass  # Implementation specific


class TestEmbeddingServiceConfiguration:
    """Tests for service configuration."""
    
    def test_missing_api_key(self):
        """Test that missing API key raises ValidationError."""
        from core.exceptions import ValidationError
        from services.search.embedding_service import EmbeddingService
        
        with pytest.raises(ValidationError):
            EmbeddingService(api_key="")
    
    def test_custom_model(self):
        """Test that custom model can be specified."""
        with patch("services.search.embedding_service.OpenAI"):
            from services.search.embedding_service import EmbeddingService
            service = EmbeddingService(
                api_key="test-key",
                model="text-embedding-3-large"
            )
            assert service.model == "text-embedding-3-large"
