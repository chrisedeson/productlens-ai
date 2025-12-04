# ProductLens AI - Test Suite

This directory contains the test suite for ProductLens AI.

## Structure

```
tests/
├── __init__.py
├── conftest.py          # Pytest fixtures
├── unit/                # Unit tests
│   ├── __init__.py
│   ├── test_embedding_service.py
│   ├── test_vector_service.py
│   ├── test_llm_service.py
│   ├── test_recommendation_service.py
│   └── test_ocr_service.py
├── integration/         # Integration tests
│   ├── __init__.py
│   └── test_api.py
└── ml/                  # ML-specific tests
    ├── __init__.py
    ├── test_data_cleaner.py
    ├── test_image_scraper.py
    └── test_image_classifier.py
```

## Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
pytest tests/unit/test_embedding_service.py -v

# Run specific test
pytest tests/unit/test_embedding_service.py::test_create_embedding -v
```

## Test Categories

- **Unit Tests**: Test individual components in isolation with mocked dependencies
- **Integration Tests**: Test API endpoints with real (or mocked) services
- **ML Tests**: Test ML pipeline components

## Writing Tests

Each test file follows this pattern:

```python
import pytest
from unittest.mock import Mock, patch

class TestServiceName:
    """Tests for ServiceName."""
    
    @pytest.fixture
    def service(self):
        """Create a service instance for testing."""
        return ServiceName(api_key="test")
    
    def test_method_success(self, service):
        """Test successful case."""
        result = service.method()
        assert result is not None
    
    def test_method_failure(self, service):
        """Test failure case."""
        with pytest.raises(ExpectedError):
            service.method(invalid_input)
```
