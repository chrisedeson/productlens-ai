"""
API Schemas Module for ProductLens AI.

Contains Pydantic models for API request/response validation.
"""

from .requests import (
    SearchRequest,
    OCRRequest,
    ClassificationRequest
)
from .responses import (
    SearchResponse,
    ProductResult,
    OCRResponse,
    ClassificationResponse,
    HealthResponse,
    ErrorResponse
)

__all__ = [
    # Requests
    "SearchRequest",
    "OCRRequest",
    "ClassificationRequest",
    # Responses
    "SearchResponse",
    "ProductResult",
    "OCRResponse",
    "ClassificationResponse",
    "HealthResponse",
    "ErrorResponse"
]
