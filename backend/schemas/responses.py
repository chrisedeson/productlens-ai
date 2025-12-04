"""
Response Schemas for ProductLens AI API.

Pydantic models for API response serialization.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ProductResult(BaseModel):
    """
    Schema for a product in search results.
    
    Attributes:
        stock_code: Product stock code.
        description: Product description.
        unit_price: Price per unit.
        country: Product origin country.
        similarity_score: Semantic similarity score.
    """
    stock_code: str = Field(description="Product stock code")
    description: str = Field(description="Product description")
    unit_price: Optional[float] = Field(
        default=None,
        description="Price per unit"
    )
    country: Optional[str] = Field(
        default=None,
        description="Country of origin"
    )
    similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "stock_code": "22384",
                    "description": "LUNCH BAG PINK POLKADOT",
                    "unit_price": 1.65,
                    "country": "United Kingdom",
                    "similarity_score": 0.89
                }
            ]
        }
    }


class SearchResponse(BaseModel):
    """
    Response schema for semantic product search.
    
    Attributes:
        success: Whether the search was successful.
        query: Original search query.
        recommendations: List of matching products.
        explanation: AI-generated explanation.
        metadata: Additional response metadata.
    """
    success: bool = Field(description="Whether search was successful")
    query: str = Field(description="Original search query")
    recommendations: List[ProductResult] = Field(
        default_factory=list,
        description="List of matching products"
    )
    explanation: str = Field(
        default="",
        description="AI-generated recommendation explanation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "query": "gift for mom",
                    "recommendations": [
                        {
                            "stock_code": "22384",
                            "description": "LUNCH BAG PINK POLKADOT",
                            "unit_price": 1.65,
                            "similarity_score": 0.89
                        }
                    ],
                    "explanation": "Here are some lovely gift ideas for mom...",
                    "metadata": {"matched_count": 5, "ai_generated": True}
                }
            ]
        }
    }


class OCRResponse(BaseModel):
    """
    Response schema for OCR text extraction.
    
    Attributes:
        success: Whether OCR was successful.
        text: Extracted text content.
        items: Parsed list items (if parse_list was True).
        confidence: OCR confidence score.
        method: OCR method used.
    """
    success: bool = Field(description="Whether OCR was successful")
    text: str = Field(default="", description="Extracted text content")
    items: List[str] = Field(
        default_factory=list,
        description="Parsed shopping list items"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="OCR confidence score"
    )
    method: str = Field(
        default="unknown",
        description="OCR method used (openai_vision, tesseract)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "text": "1. Candles 2. Picture frame 3. Mug",
                    "items": ["Candles", "Picture frame", "Mug"],
                    "confidence": 0.95,
                    "method": "openai_vision"
                }
            ]
        }
    }


class ClassificationPrediction(BaseModel):
    """
    Schema for a single classification prediction.
    
    Attributes:
        class_name: Raw class name from model.
        stock_code: Extracted product stock code.
        description: Extracted product description.
        confidence: Prediction confidence.
    """
    class_name: str = Field(description="Raw class name")
    stock_code: str = Field(description="Product stock code")
    description: str = Field(description="Product description")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Prediction confidence"
    )


class ClassificationResponse(BaseModel):
    """
    Response schema for image classification.
    
    Attributes:
        success: Whether classification was successful.
        predictions: List of predictions sorted by confidence.
        classifier_info: Information about the model used.
    """
    success: bool = Field(description="Whether classification was successful")
    predictions: List[ClassificationPrediction] = Field(
        default_factory=list,
        description="Classification predictions"
    )
    classifier_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model information"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "predictions": [
                        {
                            "class_name": "22384_LUNCH_BAG_PINK_POLKADOT",
                            "stock_code": "22384",
                            "description": "LUNCH BAG PINK POLKADOT",
                            "confidence": 0.92
                        }
                    ],
                    "classifier_info": {"num_classes": 10}
                }
            ]
        }
    }


class ServiceHealth(BaseModel):
    """
    Health status for a single service.
    
    Attributes:
        name: Service name.
        status: Health status.
        initialized: Whether service is initialized.
        details: Additional health details.
    """
    name: str = Field(description="Service name")
    status: str = Field(description="Health status")
    initialized: bool = Field(description="Whether initialized")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details"
    )


class HealthResponse(BaseModel):
    """
    Response schema for health check endpoint.
    
    Attributes:
        status: Overall health status.
        timestamp: Health check timestamp.
        version: API version.
        services: Health of individual services.
    """
    status: str = Field(description="Overall health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    version: str = Field(default="1.0.0", description="API version")
    services: Dict[str, ServiceHealth] = Field(
        default_factory=dict,
        description="Individual service health"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "version": "1.0.0",
                    "services": {
                        "embedding": {"name": "EmbeddingService", "status": "healthy"}
                    }
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """
    Response schema for API errors.
    
    Attributes:
        success: Always False for errors.
        error: Error type/code.
        message: Human-readable error message.
        details: Additional error details.
    """
    success: bool = Field(default=False, description="Always False")
    error: str = Field(description="Error type or code")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": False,
                    "error": "validation_error",
                    "message": "Query cannot be empty",
                    "details": {"field": "query"}
                }
            ]
        }
    }
