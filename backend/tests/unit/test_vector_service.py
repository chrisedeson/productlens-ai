"""
Unit Tests for VectorService.

Tests the Pinecone vector database operations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestVectorService:
    """Tests for VectorService."""
    
    @pytest.fixture
    def mock_pinecone(self):
        """Create a mock Pinecone client."""
        mock_pc = Mock()
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        return mock_pc, mock_index
    
    @pytest.fixture
    def service(self, mock_pinecone):
        """Create a VectorService with mocked client."""
        mock_pc, mock_index = mock_pinecone
        with patch("services.search.vector_service.Pinecone", return_value=mock_pc):
            from services.search.vector_service import VectorService
            svc = VectorService(
                api_key="test-key",
                index_name="test-index",
                host="test-host"
            )
            svc._index = mock_index
            return svc
    
    @pytest.fixture
    def sample_vector(self) -> List[float]:
        """Sample embedding vector."""
        return [0.1] * 1536
    
    @pytest.fixture
    def mock_query_response(self):
        """Mock Pinecone query response."""
        return Mock(
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
    
    def test_query_success(self, service, mock_pinecone, sample_vector, mock_query_response):
        """Test successful vector query."""
        _, mock_index = mock_pinecone
        mock_index.query.return_value = mock_query_response
        
        results = service.query(sample_vector, top_k=5)
        
        assert len(results) == 2
        assert results[0]["id"] == "22384"
        assert results[0]["score"] == 0.89
        mock_index.query.assert_called_once()
    
    def test_query_with_metadata(self, service, mock_pinecone, sample_vector, mock_query_response):
        """Test query includes metadata in results."""
        _, mock_index = mock_pinecone
        mock_index.query.return_value = mock_query_response
        
        results = service.query(sample_vector, top_k=5, include_metadata=True)
        
        assert "metadata" in results[0]
        assert results[0]["metadata"]["description"] == "LUNCH BAG PINK POLKADOT"
    
    def test_query_empty_results(self, service, mock_pinecone, sample_vector):
        """Test query with no matches."""
        _, mock_index = mock_pinecone
        mock_index.query.return_value = Mock(matches=[])
        
        results = service.query(sample_vector, top_k=5)
        
        assert results == []
    
    def test_query_with_filter(self, service, mock_pinecone, sample_vector, mock_query_response):
        """Test query with metadata filter."""
        _, mock_index = mock_pinecone
        mock_index.query.return_value = mock_query_response
        
        results = service.query(
            sample_vector,
            top_k=5,
            filter_dict={"country": "UK"}
        )
        
        # Verify filter was passed to Pinecone
        call_kwargs = mock_index.query.call_args[1]
        assert "filter" in call_kwargs
    
    def test_upsert_success(self, service, mock_pinecone):
        """Test successful vector upsert."""
        _, mock_index = mock_pinecone
        mock_index.upsert.return_value = Mock(upserted_count=1)
        
        vectors = [
            {
                "id": "test-1",
                "values": [0.1] * 1536,
                "metadata": {"description": "Test product"}
            }
        ]
        
        result = service.upsert(vectors)
        
        assert result is True
        mock_index.upsert.assert_called_once()
    
    def test_delete_success(self, service, mock_pinecone):
        """Test successful vector deletion."""
        _, mock_index = mock_pinecone
        mock_index.delete.return_value = None
        
        result = service.delete(ids=["test-1", "test-2"])
        
        assert result is True
        mock_index.delete.assert_called_once()
    
    def test_health_check(self, service, mock_pinecone):
        """Test health check returns correct status."""
        _, mock_index = mock_pinecone
        mock_index.describe_index_stats.return_value = Mock(
            total_vector_count=1000,
            dimension=1536
        )
        
        result = service.health_check()
        
        assert "status" in result
        assert result["status"] == "healthy"


class TestVectorServiceValidation:
    """Tests for input validation in VectorService."""
    
    @pytest.fixture
    def service(self):
        """Create service with mocked client."""
        with patch("services.search.vector_service.Pinecone"):
            from services.search.vector_service import VectorService
            return VectorService(
                api_key="test-key",
                index_name="test-index",
                host="test-host"
            )
    
    def test_query_invalid_vector_dimension(self, service):
        """Test that wrong vector dimension raises error."""
        from core.exceptions import ValidationError
        
        invalid_vector = [0.1] * 100  # Wrong dimension
        
        with pytest.raises(ValidationError):
            service.query(invalid_vector, top_k=5)
    
    def test_query_invalid_top_k(self, service):
        """Test that invalid top_k raises error."""
        from core.exceptions import ValidationError
        
        valid_vector = [0.1] * 1536
        
        with pytest.raises(ValidationError):
            service.query(valid_vector, top_k=0)
        
        with pytest.raises(ValidationError):
            service.query(valid_vector, top_k=-1)


class TestVectorServiceConfiguration:
    """Tests for service configuration."""
    
    def test_missing_api_key(self):
        """Test that missing API key raises ConfigurationError."""
        from core.exceptions import ConfigurationError
        
        with patch("services.search.vector_service.Pinecone"):
            from services.search.vector_service import VectorService
            
            with pytest.raises(ConfigurationError):
                VectorService(
                    api_key="",
                    index_name="test-index",
                    host="test-host"
                )
    
    def test_missing_index_name(self):
        """Test that missing index name raises ConfigurationError."""
        from core.exceptions import ConfigurationError
        
        with patch("services.search.vector_service.Pinecone"):
            from services.search.vector_service import VectorService
            
            with pytest.raises(ConfigurationError):
                VectorService(
                    api_key="test-key",
                    index_name="",
                    host="test-host"
                )
