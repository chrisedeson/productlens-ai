"""
Base service class for ProductLens AI.

Provides common functionality and interface for all services.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging


class BaseService(ABC):
    """
    Abstract base class for all services in ProductLens AI.
    
    Provides:
    - Consistent logging setup
    - Health check interface
    - Common utility methods
    
    All service classes should inherit from this base class
    to ensure consistent behavior and interface.
    """
    
    def __init__(self, service_name: Optional[str] = None):
        """
        Initialize the base service.
        
        Args:
            service_name: Optional custom name for the service.
                         Defaults to the class name.
        """
        self._service_name = service_name or self.__class__.__name__
        self._logger = logging.getLogger(self._service_name)
        self._is_initialized = False
    
    @property
    def service_name(self) -> str:
        """Get the service name."""
        return self._service_name
    
    @property
    def logger(self) -> logging.Logger:
        """Get the service logger."""
        return self._logger
    
    @property
    def is_initialized(self) -> bool:
        """Check if the service is properly initialized."""
        return self._is_initialized
    
    def _mark_initialized(self) -> None:
        """Mark the service as initialized."""
        self._is_initialized = True
        self._logger.info(f"{self._service_name} initialized successfully")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the service.
        
        Returns:
            Dictionary with health status and details.
        """
        return {
            "service": self._service_name,
            "status": "healthy" if self._is_initialized else "not_initialized",
            "initialized": self._is_initialized,
        }
    
    def _log_operation(
        self,
        operation: str,
        details: Optional[Dict[str, Any]] = None,
        level: str = "info"
    ) -> None:
        """
        Log an operation with consistent formatting.
        
        Args:
            operation: Name of the operation being performed.
            details: Optional details to include in the log.
            level: Log level (debug, info, warning, error).
        """
        message = f"[{operation}]"
        if details:
            message += f" {details}"
        
        log_method = getattr(self._logger, level, self._logger.info)
        log_method(message)
    
    def _handle_error(
        self,
        operation: str,
        error: Exception,
        reraise: bool = True
    ) -> None:
        """
        Handle an error with consistent logging.
        
        Args:
            operation: Name of the operation that failed.
            error: The exception that occurred.
            reraise: Whether to re-raise the exception.
        """
        self._logger.error(
            f"[{operation}] Error: {type(error).__name__}: {str(error)}"
        )
        if reraise:
            raise


class HealthCheckMixin:
    """
    Mixin class providing enhanced health check functionality.
    
    Use with services that need detailed health status reporting.
    """
    
    def detailed_health_check(self) -> Dict[str, Any]:
        """
        Perform a detailed health check.
        
        Override this method in subclasses to add service-specific checks.
        
        Returns:
            Dictionary with detailed health status.
        """
        return {
            "status": "healthy",
            "checks": {},
        }


class RetryMixin:
    """
    Mixin class providing retry functionality for external API calls.
    
    Use with services that make external API calls that may fail transiently.
    """
    
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0
    
    def _with_retry(
        self,
        operation: callable,
        max_retries: int = DEFAULT_MAX_RETRIES,
        delay: float = DEFAULT_RETRY_DELAY,
        exceptions: tuple = (Exception,)
    ):
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Callable to execute.
            max_retries: Maximum number of retry attempts.
            delay: Delay between retries in seconds.
            exceptions: Tuple of exception types to catch and retry.
            
        Returns:
            Result of the operation.
            
        Raises:
            The last exception if all retries fail.
        """
        import time
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                return operation()
            except exceptions as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
        
        raise last_exception
