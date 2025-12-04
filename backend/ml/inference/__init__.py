"""
ML Inference Module for ProductLens AI.

Contains components for model inference:
- ImageClassifier: Product image classification using CNN
"""

from .image_classifier import ImageClassifier, ClassificationResult

__all__ = ["ImageClassifier", "ClassificationResult"]
