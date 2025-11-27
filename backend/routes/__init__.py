"""
Routes package for ProductLens AI.
Contains all API endpoint handlers.
"""

from .recommendation_routes import recommendation_bp
from .ocr_routes import ocr_bp
from .image_routes import image_bp

__all__ = ["recommendation_bp", "ocr_bp", "image_bp"]
