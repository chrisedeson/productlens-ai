"""
ML Training Module for ProductLens AI.

Contains components for model training:
- CNNTrainer: Train product classification CNN
- DataPipeline: Data loading and augmentation
"""

from .cnn_trainer import CNNTrainer, TrainingResult
from .data_pipeline import DataPipeline, AugmentationConfig

__all__ = [
    "CNNTrainer",
    "TrainingResult",
    "DataPipeline",
    "AugmentationConfig"
]
