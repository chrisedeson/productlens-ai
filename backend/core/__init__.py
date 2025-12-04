"""
Core module for ProductLens AI.

Contains base classes, interfaces, and common exceptions used across the application.
"""

from .base_service import BaseService
from .exceptions import (
    ProductLensError,
    ServiceError,
    ValidationError,
    ConfigurationError,
    ExternalAPIError,
    RateLimitError,
    ModelNotFoundError,
    ModelError,
    DataProcessingError,
    DataError,
    ImageProcessingError,
    OCRError,
)

__all__ = [
    "BaseService",
    "ProductLensError",
    "ServiceError",
    "ValidationError",
    "ConfigurationError",
    "ExternalAPIError",
    "RateLimitError",
    "ModelNotFoundError",
    "ModelError",
    "DataProcessingError",
    "DataError",
    "ImageProcessingError",
    "OCRError",
]
