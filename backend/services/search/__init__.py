"""
Search Services Module for ProductLens AI.

This module contains services for semantic product search:
- EmbeddingService: Generate text embeddings using OpenAI
- VectorService: Manage and query Pinecone vector database
- LLMService: Generate AI-powered recommendations
- RecommendationService: Orchestrate the recommendation pipeline
- RecommendationServiceFactory: Factory for creating configured services
"""

from .embedding_service import EmbeddingService
from .vector_service import VectorService
from .llm_service import LLMService
from .recommendation_service import (
    RecommendationService,
    RecommendationServiceFactory,
    RecommendationResult
)

__all__ = [
    "EmbeddingService",
    "VectorService", 
    "LLMService",
    "RecommendationService",
    "RecommendationServiceFactory",
    "RecommendationResult"
]
