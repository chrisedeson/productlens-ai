"""
Request Schemas for ProductLens AI API.

Pydantic models for validating incoming API requests.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """
    Request schema for semantic product search.
    
    Attributes:
        query: Search query string.
        top_k: Number of results to return.
        include_ai_explanation: Whether to generate AI explanations.
        country: Optional country filter.
    
    Example:
        {
            "query": "gift for mom who likes gardening",
            "top_k": 5,
            "include_ai_explanation": true
        }
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query string"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return"
    )
    include_ai_explanation: bool = Field(
        default=True,
        description="Whether to generate AI-powered explanations"
    )
    country: Optional[str] = Field(
        default=None,
        description="Filter by country"
    )
    
    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Validate query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "birthday gift for dad",
                    "top_k": 5,
                    "include_ai_explanation": True
                }
            ]
        }
    }


class OCRRequest(BaseModel):
    """
    Request schema for OCR text extraction.
    
    Note: The actual image is sent as multipart form data,
    not in the JSON body. This schema is for any additional
    options that may be passed.
    
    Attributes:
        parse_list: Whether to parse as shopping list.
        language: Expected language (default: English).
    """
    parse_list: bool = Field(
        default=True,
        description="Parse extracted text as a shopping list"
    )
    language: str = Field(
        default="en",
        description="Expected text language"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "parse_list": True,
                    "language": "en"
                }
            ]
        }
    }


class ClassificationRequest(BaseModel):
    """
    Request schema for image classification.
    
    Note: The actual image is sent as multipart form data.
    This schema is for classification options.
    
    Attributes:
        top_k: Number of top predictions to return.
        confidence_threshold: Minimum confidence to include.
    """
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top predictions to return"
    )
    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to include"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "top_k": 5,
                    "confidence_threshold": 0.1
                }
            ]
        }
    }


class BatchSearchRequest(BaseModel):
    """
    Request schema for batch product search.
    
    Used when searching for multiple queries at once,
    such as items from an OCR-parsed shopping list.
    
    Attributes:
        queries: List of search queries.
        top_k_per_query: Results per query.
        include_ai_explanation: Generate AI explanations.
    """
    queries: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of search queries"
    )
    top_k_per_query: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of results per query"
    )
    include_ai_explanation: bool = Field(
        default=False,
        description="Generate AI explanations (slower for batch)"
    )
    
    @field_validator("queries")
    @classmethod
    def queries_not_empty(cls, v: List[str]) -> List[str]:
        """Validate queries are not empty."""
        return [q.strip() for q in v if q.strip()]
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "queries": ["candles", "picture frame", "mug"],
                    "top_k_per_query": 3
                }
            ]
        }
    }
