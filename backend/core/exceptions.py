"""
Custom exceptions for ProductLens AI.

Provides a hierarchy of exceptions for different error scenarios,
enabling proper error handling and meaningful error messages.
"""

from typing import Optional, Dict, Any


class ProductLensError(Exception):
    """
    Base exception for all ProductLens AI errors.
    
    All custom exceptions should inherit from this class to enable
    catching all application-specific errors with a single except clause.
    
    Attributes:
        message: Human-readable error message.
        details: Optional dictionary with additional error context.
        original_error: Optional original exception that caused this error.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        result = {
            "error": self.__class__.__name__,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ServiceError(ProductLensError):
    """
    Exception raised when a service operation fails.
    
    Use for general service-level errors that don't fit more specific categories.
    """
    
    def __init__(
        self,
        service_name: str,
        operation: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.service_name = service_name
        self.operation = operation
        full_message = f"[{service_name}] {operation} failed: {message}"
        super().__init__(full_message, details, original_error)


class ValidationError(ProductLensError):
    """
    Exception raised when input validation fails.
    
    Use for user input validation, request body validation, etc.
    """
    
    def __init__(
        self,
        field: str,
        message: str,
        value: Any = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.field = field
        self.value = value
        full_message = f"Validation failed for '{field}': {message}"
        details = details or {}
        details["field"] = field
        if value is not None:
            details["provided_value"] = str(value)[:100]  # Truncate for safety
        super().__init__(full_message, details)


class ConfigurationError(ProductLensError):
    """
    Exception raised when configuration is invalid or missing.
    
    Use for missing environment variables, invalid config values, etc.
    """
    
    def __init__(
        self,
        config_key: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.config_key = config_key
        full_message = f"Configuration error for '{config_key}': {message}"
        details = details or {}
        details["config_key"] = config_key
        super().__init__(full_message, details)


class ExternalAPIError(ProductLensError):
    """
    Exception raised when an external API call fails.
    
    Use for OpenAI, Pinecone, or other third-party API errors.
    """
    
    def __init__(
        self,
        api_name: str,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.api_name = api_name
        self.status_code = status_code
        full_message = f"External API error [{api_name}]: {message}"
        details = details or {}
        details["api_name"] = api_name
        if status_code:
            details["status_code"] = status_code
        super().__init__(full_message, details, original_error)


class RateLimitError(ExternalAPIError):
    """
    Exception raised when an API rate limit is exceeded.
    
    Use for handling rate limiting from OpenAI, Pinecone, or other APIs.
    """
    
    def __init__(
        self,
        api_name: str,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.retry_after = retry_after
        details = details or {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(api_name, message, status_code=429, details=details, original_error=original_error)


class ModelNotFoundError(ProductLensError):
    """
    Exception raised when a ML model file is not found.
    
    Use when attempting to load a model that doesn't exist.
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "model",
        details: Optional[Dict[str, Any]] = None
    ):
        self.model_path = model_path
        self.model_type = model_type
        full_message = f"{model_type.title()} not found at: {model_path}"
        details = details or {}
        details["model_path"] = model_path
        details["model_type"] = model_type
        super().__init__(full_message, details)


# Alias for backward compatibility
ModelError = ModelNotFoundError


class DataProcessingError(ProductLensError):
    """
    Exception raised when data processing fails.
    
    Use for data cleaning, transformation, or preprocessing errors.
    """
    
    def __init__(
        self,
        operation: str,
        message: str,
        row_index: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.operation = operation
        self.row_index = row_index
        full_message = f"Data processing error during '{operation}': {message}"
        details = details or {}
        details["operation"] = operation
        if row_index is not None:
            details["row_index"] = row_index
        super().__init__(full_message, details, original_error)


# Alias for backward compatibility
DataError = DataProcessingError


class ImageProcessingError(ProductLensError):
    """
    Exception raised when image processing fails.
    
    Use for image loading, preprocessing, or classification errors.
    """
    
    def __init__(
        self,
        operation: str,
        message: str,
        image_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.operation = operation
        self.image_path = image_path
        full_message = f"Image processing error during '{operation}': {message}"
        details = details or {}
        details["operation"] = operation
        if image_path:
            details["image_path"] = image_path
        super().__init__(full_message, details, original_error)


class OCRError(ProductLensError):
    """
    Exception raised when OCR processing fails.
    
    Use for text extraction errors from images.
    """
    
    def __init__(
        self,
        message: str,
        ocr_engine: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.ocr_engine = ocr_engine
        full_message = f"OCR error [{ocr_engine}]: {message}"
        details = details or {}
        details["ocr_engine"] = ocr_engine
        super().__init__(full_message, details, original_error)
