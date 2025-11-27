"""
Input validation utilities for ProductLens AI.
Provides safeguards against bad queries and sensitive data exposure.
"""

import re
from typing import Tuple


class QueryValidator:
    """Validates and sanitizes user queries for security and quality."""
    
    # Patterns for potentially harmful content
    INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",  # SQL keywords
        r"(<script.*?>.*?</script>)",  # XSS attempts
        r"(\{\{.*?\}\})",  # Template injection
        r"(__import__|eval|exec|compile)",  # Python injection
    ]
    
    # Patterns for sensitive data that shouldn't be processed
    SENSITIVE_PATTERNS = [
        r"\b\d{16}\b",  # Credit card numbers
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN format
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email addresses
        r"\b\d{10,}\b",  # Long number sequences (phone, account numbers)
    ]
    
    # Minimum and maximum query lengths
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 500
    
    @classmethod
    def validate(cls, query: str) -> Tuple[bool, str]:
        """
        Validate a user query for safety and quality.
        
        Args:
            query: The user's input query string.
            
        Returns:
            Tuple of (is_valid, error_message).
            If valid, error_message will be empty string.
        """
        if not query or not query.strip():
            return False, "Query cannot be empty."
        
        query = query.strip()
        
        # Check length constraints
        if len(query) < cls.MIN_QUERY_LENGTH:
            return False, f"Query must be at least {cls.MIN_QUERY_LENGTH} characters."
        
        if len(query) > cls.MAX_QUERY_LENGTH:
            return False, f"Query cannot exceed {cls.MAX_QUERY_LENGTH} characters."
        
        # Check for injection attempts
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains invalid characters or patterns."
        
        # Check for sensitive data
        for pattern in cls.SENSITIVE_PATTERNS:
            if re.search(pattern, query):
                return False, "Please do not include personal or sensitive information in your query."
        
        return True, ""
    
    @classmethod
    def sanitize(cls, query: str) -> str:
        """
        Sanitize a query by removing potentially harmful content.
        
        Args:
            query: The user's input query string.
            
        Returns:
            Sanitized query string.
        """
        if not query:
            return ""
        
        # Strip whitespace
        query = query.strip()
        
        # Remove HTML tags
        query = re.sub(r"<[^>]+>", "", query)
        
        # Remove multiple spaces
        query = re.sub(r"\s+", " ", query)
        
        # Remove non-printable characters
        query = "".join(char for char in query if char.isprintable())
        
        return query
    
    @classmethod
    def is_product_related(cls, query: str) -> bool:
        """
        Check if a query appears to be product-related.
        
        Args:
            query: The sanitized query string.
            
        Returns:
            True if query seems product-related.
        """
        # Product-related keywords
        product_keywords = [
            "buy", "product", "item", "price", "cost", "looking for",
            "need", "want", "find", "search", "show", "recommend",
            "suggest", "similar", "like", "best", "cheap", "expensive",
            "gift", "kitchen", "home", "decor", "bag", "clock", "bottle",
            "tea", "cake", "ribbon", "storage", "lunch"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in product_keywords)
