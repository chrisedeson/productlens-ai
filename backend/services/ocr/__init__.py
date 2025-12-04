"""
OCR Services Module for ProductLens AI.

This module contains services for optical character recognition:
- OCRService: Extract text from handwritten images using Tesseract
"""

from .ocr_service import OCRService

__all__ = ["OCRService"]