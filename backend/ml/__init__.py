"""
ML Module for ProductLens AI.

This module contains machine learning components:
- config: Configuration and hyperparameters
- data: Data processing and loading
- training: Model training pipelines
- inference: Production inference
- evaluation: Model evaluation metrics
"""

from .config import (
    MLConfig,
    ImageConfig,
    TrainingConfig,
    CNNModelConfig,
    DataPaths,
    HuggingFaceConfig,
    InferenceConfig,
    ProductCategories,
    get_config,
    create_config
)

__all__ = [
    "MLConfig",
    "ImageConfig",
    "TrainingConfig",
    "CNNModelConfig",
    "DataPaths",
    "HuggingFaceConfig",
    "InferenceConfig",
    "ProductCategories",
    "get_config",
    "create_config"
]
